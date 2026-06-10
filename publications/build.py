#!/usr/bin/env python3
"""Build data/publications.json for the lamalab.org publications page.

Pipeline:  dois.txt -> metadata + abstracts (CrossRef / Semantic Scholar)
           -> abstract embeddings -> 2D projection -> topic clusters + labels
           -> data/publications.json  (committed; rendered by Hugo, never recomputed
              at build time).

The backends degrade gracefully so the script runs anywhere:
  * embeddings: a hosted model over HTTPS (OPENAI_API_KEY -> text-embedding-3-large,
                or VOYAGE_API_KEY -> voyage-3.5; no torch) -> local
                sentence-transformers -> TF-IDF + TruncatedSVD
  * pillars:    a strong LLM (ANTHROPIC_API_KEY) -> embedding similarity -> keywords
  * 2D layout:  UMAP if installed, else PCA

Human review = edit data/publications.json directly. Re-runs preserve human-renamed
topics by matching each new cluster to the prior topic with the most DOIs in common.
Paper inclusion is driven solely by dois.txt.

Usage:
    python build.py            # uses on-disk cache for network fetches
    python build.py --force    # ignore cache, re-fetch everything
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DOIS_FILE = HERE / "dois.txt"
CACHE_DIR = HERE / ".cache"
OUT_FILE = REPO / "data" / "publications.json"

MAILTO = "kevin.jablonka@uni-jena.de"
HEADERS = {"User-Agent": f"lamalab-publications/1.0 (mailto:{MAILTO})"}


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
    ann = {"role": None, "highlight": False, "pillar": None, "venue": None}
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
            if key in ann:
                ann[key] = val.lower() if key in ("role", "pillar") else val
            continue
        low = tok.lower()
        if low in ("highlight", "highlighted"):
            ann["highlight"] = True
        elif low in ROLES:
            ann["role"] = low
        elif low in PILLARS or low == "unclassified":
            ann["pillar"] = low
    return ann


def read_dois() -> list[dict]:
    """Parse dois.txt. Each line is a DOI (or `arXiv:ID`) plus an optional
    `# comment` that may carry curated annotations inside braces, e.g.

        10.1038/s41557-025-01815-x   # ChemBench {evaluation, corresponding, highlight}
        arXiv:2505.12534             # ChemPile {reasoning, corresponding, venue=NeurIPS 2025}

    Recognised: a pillar (perception|reasoning|evaluation|action), a role
    (corresponding|first|coauthor), `highlight`, and `venue=...`. These come from
    human curation (the source of truth) and flow straight into the output."""
    if not DOIS_FILE.exists():
        sys.exit(f"missing {DOIS_FILE}")
    out, seen = [], set()
    for raw in DOIS_FILE.read_text().splitlines():
        doi_part, _, comment = raw.partition("#")
        ident = doi_part.strip()
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", ident, flags=re.I)
        if not doi:
            continue
        # arXiv entries: accept `arXiv:2505.12534`, an abs URL, or the arXiv DOI.
        arxiv = None
        m_ax = (re.match(r"(?:arxiv:|https?://arxiv\.org/abs/)(\d{4}\.\d{4,5}(?:v\d+)?)$", ident, re.I)
                or re.match(r"10\.48550/arxiv\.(\d{4}\.\d{4,5}(?:v\d+)?)$", doi, re.I))
        if m_ax:
            arxiv = m_ax.group(1)
            doi = f"10.48550/arXiv.{arxiv.split('v')[0]}"
        ann = _parse_annotations(comment)
        key = doi.lower()
        if key not in seen:
            seen.add(key)
            out.append({"doi": doi, "arxiv": arxiv, **ann})
    return out


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
# embeddings + 2D projection
# --------------------------------------------------------------------------- #
def _api_embed(texts: list[str]) -> np.ndarray | None:
    """Embed via a hosted model over plain HTTPS — no torch. Prefers OpenAI
    (`text-embedding-3-large`), then Voyage. Returns None if no key is set."""
    texts = [t[:8000] for t in texts]                      # stay under token limits
    if os.environ.get("OPENAI_API_KEY"):
        key = os.environ["OPENAI_API_KEY"]
        model = os.environ.get("PUBLICATIONS_EMBED_MODEL", "text-embedding-3-large")
        url, payload, prov = ("https://api.openai.com/v1/embeddings",
                              {"model": model, "input": texts}, "OpenAI")
    elif os.environ.get("VOYAGE_API_KEY"):
        key = os.environ["VOYAGE_API_KEY"]
        model = os.environ.get("PUBLICATIONS_EMBED_MODEL", "voyage-3.5")
        url, payload, prov = ("https://api.voyageai.com/v1/embeddings",
                              {"model": model, "input": texts, "input_type": "document"}, "Voyage")
    else:
        return None
    try:
        from sklearn.preprocessing import normalize
        r = requests.post(url, headers={"Authorization": f"Bearer {key}"},
                          json=payload, timeout=120)
        r.raise_for_status()
        rows = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        print(f"  embeddings: {prov} {model}")
        return normalize(np.asarray([d["embedding"] for d in rows], dtype=float))
    except Exception as e:                                  # noqa: BLE001
        print(f"  embeddings: {prov} API failed ({e.__class__.__name__}), falling back",
              file=sys.stderr)
        return None


def embed(texts: list[str]) -> np.ndarray:
    """Dense embeddings. Preference: hosted model (OpenAI/Voyage, no torch) →
    local sentence-transformers → TF-IDF + SVD."""
    api = _api_embed(texts)
    if api is not None:
        return api
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  embeddings: sentence-transformers/all-MiniLM-L6-v2")
        return np.asarray(model.encode(texts, normalize_embeddings=True))
    except Exception as e:                                  # noqa: BLE001
        print(f"  embeddings: TF-IDF fallback ({e.__class__.__name__})")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        tfidf = TfidfVectorizer(stop_words="english", max_features=4096,
                                ngram_range=(1, 2)).fit_transform(texts)
        dim = max(2, min(50, tfidf.shape[1] - 1, len(texts) - 1))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            svd = TruncatedSVD(n_components=dim, random_state=42)
            reduced = np.nan_to_num(svd.fit_transform(tfidf))
        return normalize(reduced)


def project_2d(emb: np.ndarray) -> np.ndarray:
    n = len(emb)
    if n < 3:                                               # too few for any reducer
        coords = np.zeros((n, 2))
        for i in range(n):
            coords[i] = [i, 0]
        return coords
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, n - 1),
                            min_dist=0.1, metric="cosine", random_state=42)
        print("  layout: UMAP")
        return reducer.fit_transform(emb)
    except Exception as e:                                  # noqa: BLE001
        print(f"  layout: PCA fallback ({e.__class__.__name__})")
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(emb)


# --------------------------------------------------------------------------- #
# classification into the group's four research threads (pillars)
# --------------------------------------------------------------------------- #
PILLAR_ORDER = PILLARS + ["unclassified"]
PILLAR_COLOR = {
    "perception": "#2C6E91",   # blue
    "reasoning": "#8E5BA6",    # purple
    "evaluation": "#A43830",   # brand burgundy
    "action": "#5B8C5A",       # green
    "unclassified": "#9AA0A6", # grey
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


def _keyword_pillar(text: str) -> str:
    t = f" {text.lower()} "
    scores = {p: sum(t.count(k) for k in kws) for p, kws in PILLAR_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unclassified"


def _llm_classify(papers: list[dict]) -> list[str] | None:
    """Classify papers into pillars with a strong LLM (Claude), reading the full
    abstract against the pillar descriptions. Returns None if the SDK or an API
    key is unavailable, so the caller can fall back."""
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
    schema = {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer"},
                        "pillar": {"type": "string", "enum": PILLAR_ORDER},
                    },
                    "required": ["i", "pillar"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["classifications"],
        "additionalProperties": False,
    }
    system = (
        "You classify scientific papers into a research group's four pillars. "
        "Read each paper's title and abstract and assign the single best-fitting "
        "pillar. Use 'unclassified' only when a paper (e.g. an editorial or "
        "community note) genuinely fits none. Return every paper exactly once."
    )
    user = (f"The four pillars:\n{pillar_block}\n\n"
            f"Classify these papers by their index `i`:\n{json.dumps(items, ensure_ascii=False)}")
    try:
        resp = anthropic.Anthropic().messages.create(
            model=model, max_tokens=8000, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}})
        text = next(b.text for b in resp.content if b.type == "text")
        by_i = {c["i"]: c["pillar"] for c in json.loads(text)["classifications"]}
        print(f"  pillars: LLM classification ({model})")
        return [by_i.get(i, "unclassified") for i in range(len(papers))]
    except Exception as e:                                 # noqa: BLE001
        print(f"  pillars: LLM failed ({e.__class__.__name__}), falling back", file=sys.stderr)
        return None


def suggest_pillars(papers: list[dict], texts: list[str]) -> list[str]:
    """Auto-suggest a pillar per paper. Preference order: a strong LLM reading the
    abstract → transformer similarity to the pillar descriptions → keyword heuristic."""
    llm = _llm_classify(papers)
    if llm is not None:
        return llm
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        pe = model.encode([PILLAR_DESC[p] for p in PILLARS], normalize_embeddings=True)
        te = model.encode(texts, normalize_embeddings=True)
        sims = np.asarray(te) @ np.asarray(pe).T
        print("  pillars: embedding similarity (sentence-transformers)")
        return [PILLARS[i] for i in sims.argmax(1)]
    except Exception:                                      # noqa: BLE001
        print("  pillars: keyword heuristic")
        return [_keyword_pillar(t) for t in texts]


def classify_pillars(papers: list[dict], texts: list[str]) -> None:
    """Fill in a pillar for every paper: explicit annotation wins, otherwise
    auto-suggest. Mutates papers in place."""
    need = [i for i, p in enumerate(papers) if not p.get("pillar")]
    if need:
        suggestions = suggest_pillars([papers[i] for i in need], [texts[i] for i in need])
        for idx, pil in zip(need, suggestions):
            papers[idx]["pillar"] = pil
            papers[idx]["pillar_auto"] = True              # flag for human review


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="ignore cache, re-fetch all metadata")
    args = ap.parse_args()

    load_dotenv()
    entries = read_dois()
    if not entries:
        sys.exit("dois.txt is empty — add at least one DOI")
    print(f"fetching metadata for {len(entries)} DOIs ...")

    papers = []
    for entry in entries:
        rec = fetch(entry, args.force)
        if rec:
            rec["role"] = entry["role"]
            rec["highlighted"] = entry["highlight"]
            rec["pillar"] = entry["pillar"]                 # explicit annotation (may be None)
            print(f"  ✓ {rec['year']}  {rec['title'][:70]}")
            papers.append(rec)
    if not papers:
        sys.exit("no papers resolved — check the DOIs")

    texts = [f"{p['title']}. {p.get('abstract', '')}".strip() for p in papers]
    print("embedding abstracts ...")
    emb = embed(texts)
    coords = project_2d(emb)

    print("classifying into research threads ...")
    classify_pillars(papers, texts)                        # fills any missing pillar

    # the four pillars are the landscape's topics — fixed colours, canonical order
    present = [p for p in PILLAR_ORDER if any(pp["pillar"] == p for pp in papers)]
    pid = {name: PILLAR_ORDER.index(name) for name in PILLAR_ORDER}
    topics = [{"id": pid[name], "name": name, "color": PILLAR_COLOR[name]}
              for name in present]

    for p, (x, y) in zip(papers, coords):
        p["topic"] = pid[p["pillar"]]
        p["x"] = round(float(x), 4)
        p["y"] = round(float(y), 4)

    papers.sort(key=lambda p: (-(p["year"] or 0), p["title"].lower()))

    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps({"topics": topics, "papers": papers},
                                   indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT_FILE.relative_to(REPO)}: "
          f"{len(papers)} papers across {len(topics)} threads")


if __name__ == "__main__":
    main()
