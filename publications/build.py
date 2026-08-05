#!/usr/bin/env python3
"""Build data/publications.json for the lamalab.org publications page.

Pipeline:  dois.txt -> metadata + abstracts (CrossRef / Semantic Scholar / arXiv)
           -> classify each paper into one or two research threads (pillars)
           -> data/publications.json  (committed; rendered by Hugo as a pillar map
              + filterable list, never recomputed at build time).

Classification degrades gracefully: a strong LLM reads the abstract
(ANTHROPIC_API_KEY) -> keyword heuristic. An explicit pillar in dois.txt always
wins. The figure is anchored on the pillars (no embeddings needed).

Human review = edit data/publications.json directly. Local annotations in dois.txt
remain authoritative. Paper inclusion normally comes from dois.txt; automated syncs
can instead supply lamalab-org/outputs/outputs/publications.yaml.

Usage:
    python build.py            # uses on-disk cache for network fetches
    python build.py --force    # ignore cache, re-fetch everything
    python build.py --outputs-file publications.yaml
                               # use lamalab-org/outputs as the publication list
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DOIS_FILE = HERE / "dois.txt"
CACHE_DIR = HERE / ".cache"
OUT_FILE = REPO / "data" / "publications.json"

MAILTO = "kevin.jablonka@uni-jena.de"
HEADERS = {"User-Agent": f"lamalab-publications/1.0 (mailto:{MAILTO})"}


LINKS_FILE = HERE / "links.json"
# DOI prefixes that identify preprint servers (arXiv, ChemRxiv, bioRxiv, Research
# Square, ChemRxiv legacy) — used to default `type` to "preprint".
PREPRINT_PREFIXES = ("10.48550/arxiv", "10.26434/chemrxiv", "10.1101/",
                     "10.21203/", "10.33774/", "10.31223/", "10.31224/")


def _is_preprint(doi: str) -> bool:
    return doi.lower().startswith(PREPRINT_PREFIXES)


def load_links() -> dict:
    """Optional curated media/news links per DOI: {doi: [{label, url}, ...]}."""
    if not LINKS_FILE.exists():
        return {}
    try:
        return {k.lower(): v for k, v in json.loads(LINKS_FILE.read_text()).items()}
    except Exception:                                      # noqa: BLE001
        return {}


def load_dotenv() -> None:
    """Load KEY=VALUE pairs from a `.env` (repo root or publications/) into the
    environment, without overriding variables already set. Keeps API keys out of
    the shell history and the repo (`.env` is git-ignored)."""
    for path in (REPO / ".env", HERE / ".env"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
ROLES = {"corresponding", "first", "coauthor"}
# the group's four research threads — these are the landscape's colours
PILLARS = ["perception", "reasoning", "evaluation", "action"]


def _parse_annotations(comment: str) -> dict:
    """Parse the `{...}` block of a dois.txt comment. Supports bare flags
    (a role, a pillar, or `highlight`) and `key=value` pairs (`venue=NeurIPS 2025`,
    `pillar=evaluation`). Comma-separated so values may contain spaces."""
    ann = {"role": None, "highlight": False, "pillars": [], "venue": None, "type": None}
    m = re.search(r"\{([^}]*)\}", comment)
    if not m:
        return ann
    for tok in m.group(1).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            key, _, val = tok.partition("=")
            key, val = key.strip().lower(), val.strip()
            if key == "pillar":
                ann["pillars"].append(val.lower())
            elif key in ann:
                ann[key] = val.lower() if key in ("role", "type") else val
            continue
        low = tok.lower()
        if low in ("highlight", "highlighted"):
            ann["highlight"] = True
        elif low in ROLES:
            ann["role"] = low
        elif low in PILLARS:
            ann["pillars"].append(low)
        elif low in ("community", "unclassified"):     # old alias maps to community
            ann["pillars"].append("community")
    seen: list[str] = []
    for p in ann["pillars"]:
        if p not in seen:
            seen.append(p)
    ann["pillars"] = seen[:2]
    return ann


def _parse_identifier(ident: str, comment: str = "") -> dict | None:
    """Normalize one DOI/arXiv identifier and its optional annotation comment."""
    ident = ident.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", ident, flags=re.I)
    if not doi:
        return None
    # arXiv entries: accept `arXiv:2505.12534`, an abs URL, or the arXiv DOI.
    arxiv = None
    m_ax = (re.match(r"(?:arxiv:|https?://arxiv\.org/abs/)(\d{4}\.\d{4,5}(?:v\d+)?)$", ident, re.I)
            or re.match(r"10\.48550/arxiv\.(\d{4}\.\d{4,5}(?:v\d+)?)$", doi, re.I))
    if m_ax:
        arxiv = m_ax.group(1)
        doi = f"10.48550/arXiv.{arxiv.split('v')[0]}"
    return {"doi": doi, "arxiv": arxiv, **_parse_annotations(comment)}


def read_dois(path: Path = DOIS_FILE) -> list[dict]:
    """Parse a curated DOI file. Each line is a DOI (or `arXiv:ID`) plus an optional
    `# comment` that may carry curated annotations inside braces, e.g.

        10.1038/s41557-025-01815-x   # ChemBench {evaluation, corresponding, highlight}
        arXiv:2505.12534             # ChemPile {reasoning, corresponding, venue=NeurIPS 2025}

    Recognised: a pillar (perception|reasoning|evaluation|action), a role
    (corresponding|first|coauthor), `highlight`, and `venue=...`. These come from
    human curation (the source of truth) and flow straight into the output."""
    if not path.exists():
        sys.exit(f"missing {path}")
    out, seen = [], set()
    for raw in path.read_text().splitlines():
        doi_part, _, comment = raw.partition("#")
        entry = _parse_identifier(doi_part, comment)
        if not entry:
            continue
        key = entry["doi"].lower()
        if key not in seen:
            seen.add(key)
            out.append(entry)
    return out


def read_outputs(path: Path) -> list[dict]:
    """Read the publications list from lamalab-org/outputs.

    Matching entries in the local dois.txt keep their curated pillar, venue, and
    type annotations. Upstream owns inclusion plus the default publication type
    and highlight flag.
    """
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is required when --outputs-file is used")

    try:
        document = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        sys.exit(f"could not read outputs publications file {path}: {e}")

    curated = {entry["doi"].lower(): entry for entry in read_dois()}
    type_map = {"paper": "peer-reviewed", "preprint": "preprint", "editorial": "editorial"}
    entries, seen = [], set()
    for output in document.get("outputs", []):
        ident = output.get("doi") or (f"arXiv:{output['arxiv']}" if output.get("arxiv") else "")
        entry = _parse_identifier(str(ident))
        if not entry:
            continue
        key = entry["doi"].lower()
        if key in seen:
            continue
        seen.add(key)

        local = curated.get(key, {})
        entry["pillars"] = local.get("pillars", [])
        entry["venue"] = local.get("venue") or output.get("venue")
        entry["type"] = local.get("type") or type_map.get(str(output.get("type", "")).lower())
        entry["role"] = local.get("role")
        entry["highlight"] = bool(local.get("highlight") or output.get("highlight"))
        entries.append(entry)
    return entries


# --------------------------------------------------------------------------- #
# metadata fetch
# --------------------------------------------------------------------------- #
def _cache_path(doi: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", doi)
    return CACHE_DIR / f"{safe}.json"


def _strip_jats(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)            # drop XML/JATS tags
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^abstract[:.\s]*", "", text, flags=re.I)
    return text


def _crossref(doi: str) -> dict:
    r = requests.get(f"https://api.crossref.org/works/{doi}",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    m = r.json()["message"]
    authors = []
    for a in m.get("author", []) or []:
        name = " ".join(x for x in [a.get("given"), a.get("family")] if x)
        if name:
            authors.append(name)
    venue = (m.get("short-container-title") or m.get("container-title") or [""])[0]
    year = None
    for key in ("published-print", "published-online", "issued", "created"):
        parts = (m.get(key) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            year = int(parts[0][0])
            break
    return {
        "title": (m.get("title") or [""])[0],
        "authors": authors,
        "venue": venue,
        "year": year,
        "abstract": _strip_jats(m.get("abstract")),
    }


def _semantic_scholar(sid: str) -> dict:
    """`sid` is a Semantic Scholar id, e.g. `DOI:10.x/y` or `arXiv:2505.12534`."""
    fields = "title,abstract,year,venue,citationCount,tldr,authors"
    r = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{sid}",
        params={"fields": fields}, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return {}
    m = r.json()
    tldr = (m.get("tldr") or {}).get("text") if m.get("tldr") else None
    return {
        "title": m.get("title"),
        "authors": [a.get("name") for a in (m.get("authors") or []) if a.get("name")],
        "venue": m.get("venue"),
        "year": m.get("year"),
        "abstract": m.get("abstract") or "",
        "tldr": tldr,
        "citations": m.get("citationCount"),
    }


_ATOM = "{http://www.w3.org/2005/Atom}"
_ARX = "{http://arxiv.org/schemas/atom}"


def _arxiv(arxiv_id: str) -> dict:
    import xml.etree.ElementTree as ET
    base = arxiv_id.split("v")[0]
    r = requests.get("http://export.arxiv.org/api/query",
                     params={"id_list": base}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    entry = ET.fromstring(r.text).find(f"{_ATOM}entry")
    if entry is None:
        return {}
    title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
    if title.lower().startswith("error"):
        return {}
    published = entry.findtext(f"{_ATOM}published") or ""
    return {
        "title": title,
        "abstract": " ".join((entry.findtext(f"{_ATOM}summary") or "").split()),
        "authors": [a.findtext(f"{_ATOM}name") for a in entry.findall(f"{_ATOM}author")],
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "venue": "arXiv",
        "published_doi": entry.findtext(f"{_ARX}doi"),     # set once a journal picks it up
    }


def fetch(entry: dict, force: bool) -> dict | None:
    doi = entry["doi"]
    cache = _cache_path(doi)
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    rec: dict = {"doi": doi, "url": f"https://doi.org/{doi}"}
    if entry.get("arxiv"):
        base = entry["arxiv"].split("v")[0]
        rec["url"] = f"https://arxiv.org/abs/{base}"
        sid = f"arXiv:{base}"
        try:
            primary = _arxiv(entry["arxiv"])
        except Exception as e:                              # noqa: BLE001
            print(f"  ! arXiv failed for {base}: {e}", file=sys.stderr)
            primary = {}
        if primary.get("published_doi"):                   # prefer the journal DOI once it exists
            rec["doi"] = primary["published_doi"]
            rec["url"] = f"https://doi.org/{primary['published_doi']}"
    else:
        sid = f"DOI:{doi}"
        try:
            primary = _crossref(doi)
        except Exception as e:                              # noqa: BLE001
            print(f"  ! CrossRef failed for {doi}: {e}", file=sys.stderr)
            primary = {}

    s2 = {}
    try:
        s2 = _semantic_scholar(sid)
        time.sleep(0.3)                                     # be polite to S2
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! Semantic Scholar failed for {sid}: {e}", file=sys.stderr)

    rec["title"] = primary.get("title") or s2.get("title") or doi
    rec["authors"] = primary.get("authors") or s2.get("authors") or []
    # explicit venue annotation wins (e.g. a NeurIPS paper posted on arXiv)
    rec["venue"] = entry.get("venue") or primary.get("venue") or s2.get("venue") or ""
    rec["year"] = primary.get("year") or s2.get("year")
    rec["abstract"] = (primary.get("abstract") or s2.get("abstract")
                       or s2.get("tldr") or "")
    rec["citations"] = s2.get("citations")

    if not rec["title"] or rec["title"] == doi:
        print(f"  ! no metadata resolved for {doi} — skipping", file=sys.stderr)
        return None

    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return rec


# --------------------------------------------------------------------------- #
# classification into the group's research threads (pillars)
# --------------------------------------------------------------------------- #
PILLAR_ORDER = PILLARS + ["community"]
PILLAR_COLOR = {
    "perception": "#2C6E91",   # blue
    "reasoning": "#8E5BA6",    # purple
    "evaluation": "#A43830",   # brand burgundy
    "action": "#5B8C5A",       # green
    "community": "#C98A3C",    # amber — perspectives, education, community-building
}
# one-line blurbs shown as a legend on the page
PILLAR_BLURB = {
    "perception": "understanding chemistry from what we actually measure — spectra and "
                  "characterization data, and the knowledge buried in the literature.",
    "reasoning": "combining formal rules with the tacit heuristics expert chemists use, "
                 "so models reason rather than pattern-match.",
    "evaluation": "honestly measuring what models and agents can really do — and where "
                  "they only look capable.",
    "action": "predictions experimentalists can act on, starting from the recipes and "
              "processing conditions they control.",
    "community": "perspectives, education, and community-building around AI for chemistry.",
}
# short descriptions (for embedding-based auto-suggestion) and keyword fallbacks
PILLAR_DESC = {
    "perception": "understanding chemistry through characterization data that chemists "
                  "actually measure — spectra, NMR, IR, Raman, mass spectrometry, XRD, "
                  "chromatograms; structure elucidation; foundation models over measurements; "
                  "extracting structured data from the literature.",
    "reasoning": "combining formal constraint systems with learned, tacit heuristics; "
                 "neurosymbolic methods; representations; training language models to reason "
                 "about chemistry; reasoning corpora and datasets.",
    "evaluation": "honest assessment of scientific capability; benchmarks; probing the "
                  "limitations of models and agents; whether systems are right for the right "
                  "reasons; robustness and predictability of evaluations.",
    "action": "predictions experimentalists can act on; recipe- and process-to-property or "
              "structure prediction; active learning and optimization for materials and "
              "molecules verified in the lab.",
}
PILLAR_KEYWORDS = {
    "perception": ["spectr", "nmr", "infrared", " ir ", "raman", "mass spec", "xrd",
                   "diffraction", "chromatograph", "characterization", "characterisation",
                   "structure elucidation", "data extraction", "multimodal", "perception"],
    "reasoning": ["reasoning", "neurosymbolic", "symbolic", "constraint", "representation",
                  "fine-tun", "instruct", "dataset", "corpus", "knowledge", "foundation model"],
    "evaluation": ["benchmark", "evaluat", "assessment", "limitation", "probing", "superhuman",
                   "robust", "wrong reason", "agent", "churn", "predictab"],
    "action": ["predict", "recipe", "process", "synthesis", "active learning", "optimization",
               "optimisation", "discovery", "design", "copolymer", "carbon capture",
               "gasification", "topology"],
}


def _keyword_pillars(text: str) -> list[str]:
    t = f" {text.lower()} "
    scores = {p: sum(t.count(k) for k in kws) for p, kws in PILLAR_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return [best] if scores[best] > 0 else ["community"]


def _dedupe(seq: list[str]) -> list[str]:
    out: list[str] = []
    for x in seq:
        if x in PILLAR_ORDER and x not in out:
            out.append(x)
    return out[:2]


def _llm_classify(papers: list[dict]) -> list[list[str]] | None:
    """Ask a strong LLM (Claude) to read each abstract and assign one or two
    research threads. Returns a list of 1–2 pillar names per paper, or None if no
    key/SDK is available (so the caller can fall back)."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    model = os.environ.get("PUBLICATIONS_LLM_MODEL", "claude-opus-4-8")
    pillar_block = "\n".join(f"- {p}: {PILLAR_DESC[p]}" for p in PILLARS)
    items = [{"i": i, "title": p["title"], "abstract": (p.get("abstract") or "")[:3000]}
             for i, p in enumerate(papers)]
    prompt = (
        "You assign scientific papers to a research group's threads. The four "
        f"research threads:\n{pillar_block}\n- community: non-research outputs "
        "(editorials, comments, education, perspectives).\n\n"
        "For each paper, pick ONE or TWO threads. Assign a second thread whenever "
        "the paper makes a genuine, substantive contribution to it — many of this "
        "group's papers legitimately span two threads (e.g. a benchmark that is "
        "also about reasoning, or a method that is both perception and action). "
        "Use a single thread when the paper is clearly about just one. Use "
        "'community' only for non-research outputs, and then alone.\n\n"
        f"Papers:\n{json.dumps(items, ensure_ascii=False)}\n\n"
        "Respond with ONLY a JSON array, no prose, each element "
        '{"i": <index>, "pillars": ["<thread>", ...]} using these exact names: '
        f"{PILLAR_ORDER}.")
    try:
        resp = anthropic.Anthropic().messages.create(
            model=model, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}])
        text = next(b.text for b in resp.content if b.type == "text")
        arr = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        by_i = {d["i"]: _dedupe(d.get("pillars", [])) for d in arr}
        print(f"  pillars: LLM classification ({model})")
        return [by_i.get(i) or ["community"] for i in range(len(papers))]
    except Exception as e:                                 # noqa: BLE001
        print(f"  pillars: LLM failed ({e.__class__.__name__}), falling back", file=sys.stderr)
        return None


def suggest_pillars(papers: list[dict], texts: list[str]) -> list[list[str]]:
    """Auto-suggest 1–2 pillars per paper: a strong LLM reading the abstract,
    else a keyword heuristic."""
    llm = _llm_classify(papers)
    if llm is not None:
        return llm
    print("  pillars: keyword heuristic")
    return [_keyword_pillars(t) for t in texts]


_PILLAR_CACHE = CACHE_DIR / "pillars.json"


def classify_pillars(papers: list[dict], texts: list[str], force: bool = False) -> None:
    """Fill in pillars for papers lacking an explicit annotation. Cached per DOI so
    rebuilds are stable — only genuinely new papers are (re)classified. Mutates in
    place."""
    cache = {}
    if not force and _PILLAR_CACHE.exists():
        try:
            cache = json.loads(_PILLAR_CACHE.read_text())
        except Exception:                                  # noqa: BLE001
            cache = {}
    need = []
    for i, p in enumerate(papers):
        if p.get("pillars"):                               # explicit annotation wins
            continue
        cached = cache.get(p["doi"].lower())
        if cached:
            p["pillars"], p["pillar_auto"] = cached, True
        else:
            need.append(i)
    if need:
        suggestions = suggest_pillars([papers[i] for i in need], [texts[i] for i in need])
        for idx, pillars in zip(need, suggestions):
            papers[idx]["pillars"] = pillars or ["community"]
            papers[idx]["pillar_auto"] = True
            cache[papers[idx]["doi"].lower()] = papers[idx]["pillars"]
        CACHE_DIR.mkdir(exist_ok=True)
        _PILLAR_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="ignore cache, re-fetch all metadata")
    ap.add_argument("--outputs-file", type=Path,
                    help="lamalab-org/outputs publications.yaml to use for inclusion")
    args = ap.parse_args()

    load_dotenv()
    entries = read_outputs(args.outputs_file) if args.outputs_file else read_dois()
    if not entries:
        sys.exit("publication input is empty — add at least one DOI or arXiv ID")
    print(f"fetching metadata for {len(entries)} DOIs ...")

    papers = []
    for entry in entries:
        rec = fetch(entry, args.force)
        if rec:
            rec["highlighted"] = entry["highlight"]
            rec["pillars"] = entry["pillars"] or None       # explicit annotation, else LLM fills
            rec["type"] = entry["type"] or ("preprint" if _is_preprint(rec["doi"]) else "peer-reviewed")
            print(f"  ✓ {rec['year']}  {rec['title'][:70]}")
            papers.append(rec)
    if not papers:
        sys.exit("no papers resolved — check the DOIs")

    links = load_links()                                   # curated media/news per DOI
    for p in papers:
        p["media"] = links.get(p["doi"].lower(), [])

    texts = [f"{p['title']}. {p.get('abstract', '')}".strip() for p in papers]
    print("classifying into research threads ...")
    classify_pillars(papers, texts, args.force)            # fills any missing pillars

    # threads are the group's pillars — canonical order, fixed colours + blurbs
    present = [name for name in PILLAR_ORDER
               if any(name in p["pillars"] for p in papers)]
    threads = [{"id": PILLAR_ORDER.index(name), "name": name,
                "color": PILLAR_COLOR[name], "description": PILLAR_BLURB[name]}
               for name in present]

    # the card list reads in this order: most recent first. (The figure re-orders
    # its rows by thread client-side, so the two views stay independent.)
    papers.sort(key=lambda p: (-(p["year"] or 0), p["title"].lower()))

    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps({"threads": threads, "papers": papers},
                                   indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT_FILE.relative_to(REPO)}: "
          f"{len(papers)} papers across {len(threads)} threads")


if __name__ == "__main__":
    main()
