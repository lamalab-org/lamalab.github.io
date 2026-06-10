#!/usr/bin/env python3
"""Build data/publications.json for the lamalab.org publications page.

Pipeline:  dois.txt -> metadata + abstracts (CrossRef / Semantic Scholar)
           -> abstract embeddings -> 2D projection -> topic clusters + labels
           -> data/publications.json  (committed; rendered by Hugo, never recomputed
              at build time).

The expensive backends degrade gracefully so the script runs anywhere:
  * embeddings: sentence-transformers if installed, else TF-IDF + TruncatedSVD
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
import math
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

# brand colour first, then a spread of distinct hues for the remaining topics
PALETTE = [
    "#A43830", "#2C6E91", "#E2A33C", "#5B8C5A", "#8E5BA6",
    "#C0607A", "#3F8C8C", "#B5793A", "#6B7280", "#7A9E3F",
]
MAILTO = "kevin.jablonka@uni-jena.de"
HEADERS = {"User-Agent": f"lamalab-publications/1.0 (mailto:{MAILTO})"}


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
ROLES = {"corresponding", "first", "coauthor"}


def read_dois() -> list[dict]:
    """Parse dois.txt. Each line is a DOI plus an optional `# comment` that may
    carry curated annotations inside braces, e.g.

        10.1038/s41557-025-01815-x   # ChemBench {corresponding, highlight}

    Recognised tokens: a role (corresponding|first|coauthor) and `highlight`.
    These come from human curation (the source of truth) and flow straight into
    the output, so they are not re-derived or preserved across runs."""
    if not DOIS_FILE.exists():
        sys.exit(f"missing {DOIS_FILE}")
    out, seen = [], set()
    for raw in DOIS_FILE.read_text().splitlines():
        doi_part, _, comment = raw.partition("#")
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi_part.strip(), flags=re.I)
        if not doi:
            continue
        role, highlight = None, False
        m = re.search(r"\{([^}]*)\}", comment)
        if m:
            tokens = {t.strip().lower() for t in re.split(r"[,\s]+", m.group(1)) if t.strip()}
            highlight = "highlight" in tokens or "highlighted" in tokens
            role = next((t for t in tokens if t in ROLES), None)
        key = doi.lower()
        if key not in seen:
            seen.add(key)
            out.append({"doi": doi, "role": role, "highlight": highlight})
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


def _semantic_scholar(doi: str) -> dict:
    fields = "title,abstract,year,venue,citationCount,tldr,authors"
    r = requests.get(
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
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


def fetch(doi: str, force: bool) -> dict | None:
    cache = _cache_path(doi)
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    rec: dict = {"doi": doi, "url": f"https://doi.org/{doi}"}
    try:
        cr = _crossref(doi)
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! CrossRef failed for {doi}: {e}", file=sys.stderr)
        cr = {}
    s2 = {}
    try:
        s2 = _semantic_scholar(doi)
        time.sleep(0.3)                                     # be polite to S2
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! Semantic Scholar failed for {doi}: {e}", file=sys.stderr)

    rec["title"] = cr.get("title") or s2.get("title") or doi
    rec["authors"] = cr.get("authors") or s2.get("authors") or []
    rec["venue"] = cr.get("venue") or s2.get("venue") or ""
    rec["year"] = cr.get("year") or s2.get("year")
    rec["abstract"] = (cr.get("abstract") or s2.get("abstract")
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
def embed(texts: list[str]) -> np.ndarray:
    """Dense embeddings; transformer model if available, else TF-IDF + SVD."""
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
# clustering + topic labels
# --------------------------------------------------------------------------- #
def cluster(emb: np.ndarray) -> np.ndarray:
    n = len(emb)
    k = max(2, min(8, round(math.sqrt(n / 2)), n))
    if n <= 2:
        return np.zeros(n, dtype=int)
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emb)


def label_topics(texts: list[str], labels: np.ndarray) -> dict[int, str]:
    """Class-based TF-IDF: pool each cluster's text, take its top keywords."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    ids = sorted(set(int(x) for x in labels))
    pooled = [" ".join(t for t, l in zip(texts, labels) if l == cid) for cid in ids]
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=2048,
                              ngram_range=(1, 2))
        X = vec.fit_transform(pooled)
        terms = np.array(vec.get_feature_names_out())
        names = {}
        for row, cid in enumerate(ids):
            top = X[row].toarray().ravel().argsort()[::-1][:3]
            names[cid] = " / ".join(terms[i] for i in top if X[row, i] > 0) or f"topic {cid}"
        return names
    except Exception:                                      # noqa: BLE001
        return {cid: f"topic {cid}" for cid in ids}


def carry_forward_names(new_topics: dict[int, list[str]],
                        auto_names: dict[int, str]) -> dict[int, str]:
    """Preserve human-edited topic names from an existing data/publications.json by
    matching each new cluster to the prior topic sharing the most DOIs."""
    if not OUT_FILE.exists():
        return auto_names
    try:
        prev = json.loads(OUT_FILE.read_text())
    except Exception:                                      # noqa: BLE001
        return auto_names
    prev_by_topic: dict[int, set[str]] = {}
    prev_name: dict[int, str] = {}
    for t in prev.get("topics", []):
        prev_name[t["id"]] = t["name"]
        prev_by_topic[t["id"]] = set()
    for p in prev.get("papers", []):
        prev_by_topic.setdefault(p.get("topic"), set()).add(p["doi"].lower())

    names = dict(auto_names)
    for cid, dois in new_topics.items():
        cur = {d.lower() for d in dois}
        best, best_overlap = None, 0
        for pid, pdois in prev_by_topic.items():
            ov = len(cur & pdois)
            if ov > best_overlap:
                best, best_overlap = pid, ov
        # carry the prior name when the clusters clearly correspond
        if best is not None and best_overlap >= max(1, len(cur) // 2):
            names[cid] = prev_name.get(best, names[cid])
    return names


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="ignore cache, re-fetch all metadata")
    args = ap.parse_args()

    entries = read_dois()
    if not entries:
        sys.exit("dois.txt is empty — add at least one DOI")
    print(f"fetching metadata for {len(entries)} DOIs ...")

    papers = []
    for entry in entries:
        rec = fetch(entry["doi"], args.force)
        if rec:
            rec["role"] = entry["role"]
            rec["highlighted"] = entry["highlight"]
            print(f"  ✓ {rec['year']}  {rec['title'][:70]}")
            papers.append(rec)
    if not papers:
        sys.exit("no papers resolved — check the DOIs")

    texts = [f"{p['title']}. {p.get('abstract', '')}".strip() for p in papers]
    print("embedding abstracts ...")
    emb = embed(texts)
    coords = project_2d(emb)
    labels = cluster(emb)

    auto_names = label_topics(texts, labels)
    new_topics: dict[int, list[str]] = {}
    for p, l in zip(papers, labels):
        new_topics.setdefault(int(l), []).append(p["doi"])
    names = carry_forward_names(new_topics, auto_names)

    topic_ids = sorted(set(int(l) for l in labels))
    topics = [{"id": cid, "name": names[cid],
               "color": PALETTE[i % len(PALETTE)]}
              for i, cid in enumerate(topic_ids)]

    for p, (x, y), l in zip(papers, coords, labels):
        p["topic"] = int(l)
        p["x"] = round(float(x), 4)
        p["y"] = round(float(y), 4)

    papers.sort(key=lambda p: (-(p["year"] or 0), p["title"].lower()))

    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps({"topics": topics, "papers": papers},
                                   indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT_FILE.relative_to(REPO)}: "
          f"{len(papers)} papers, {len(topics)} topics")


if __name__ == "__main__":
    main()
