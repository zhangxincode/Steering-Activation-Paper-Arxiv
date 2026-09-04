#!/usr/bin/env python3
"""Collect arXiv papers about activation steering for language models.

The script queries arXiv, deduplicates papers, stores a machine-readable JSON
database, and regenerates README tables. It is intentionally dependency-free so
it can run both locally and in GitHub Actions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "queries.json"
DATA_PATH = ROOT / "docs" / "arxiv-daily.json"
README_PATH = ROOT / "README.md"
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def paper_id_from_entry(entry: ET.Element) -> str:
    raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
    return strip_version(raw_id.rsplit("/", 1)[-1])


def parse_arxiv_datetime(value: str) -> str:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()


def category_query(categories: list[str]) -> str:
    return " OR ".join(f"cat:{category}" for category in categories)


def build_search_query(query: str, categories: list[str]) -> str:
    return f"({query}) AND ({category_query(categories)})"


def fetch_feed(query: str, start: int, max_results: int) -> ET.Element:
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "steering-activation-paper-arxiv/1.0 (mailto:example@example.com)"
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return ET.fromstring(response.read())


def feed_total_results(feed: ET.Element) -> int:
    total = feed.findtext(
        "{http://a9.com/-/spec/opensearch/1.1/}totalResults",
        default="0",
    )
    return int(total)


def extract_entry(entry: ET.Element, source_query: str) -> dict[str, Any]:
    arxiv_id = paper_id_from_entry(entry)
    title = normalize_space(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
    abstract = normalize_space(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
    authors = [
        normalize_space(author.findtext("atom:name", default="", namespaces=ATOM_NS))
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    categories = [
        category.attrib.get("term", "")
        for category in entry.findall("atom:category", ATOM_NS)
        if category.attrib.get("term")
    ]
    published = parse_arxiv_datetime(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
    updated = parse_arxiv_datetime(entry.findtext("atom:updated", default="", namespaces=ATOM_NS))

    return {
        "id": arxiv_id,
        "title": title,
        "authors": authors,
        "published": published,
        "updated": updated,
        "categories": categories,
        "abstract": abstract,
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "primary_category": categories[0] if categories else "",
        "source_queries": [source_query],
        "code_urls": extract_code_urls(abstract),
    }


def extract_code_urls(text: str) -> list[str]:
    url_pattern = re.compile(
        r"https?://(?:www\.)?(?:github|gitlab)\.com/[^\s)\]}>,.;]+",
        re.I,
    )
    return sorted(set(match.group(0) for match in url_pattern.finditer(text)))


def match_relevance(paper: dict[str, Any], config: dict[str, Any]) -> bool:
    haystack = f"{paper['title']} {paper['abstract']}".lower()
    strong_terms = [
        term.lower()
        for term in config.get("strong_topic_terms", config.get("must_match_any", []))
    ]
    weak_terms = [term.lower() for term in config.get("weak_topic_terms", [])]
    steering_words = [term.lower() for term in config.get("steering_words", [])]
    model_terms = [term.lower() for term in config.get("model_terms", [])]
    has_model = any(term in haystack for term in model_terms)
    has_strong_topic = any(term in haystack for term in strong_terms)
    has_weak_topic = any(term in haystack for term in weak_terms)
    has_steering_word = any(
        re.search(rf"\b{re.escape(term)}\w*\b", haystack)
        for term in steering_words
    )
    has_activation_context = any(
        term in haystack
        for term in ("activation", "hidden state", "internal representation", "latent representation")
    )
    return has_model and (has_strong_topic or (has_weak_topic and has_steering_word and has_activation_context))


def merge_papers(existing: list[dict[str, Any]], new_papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {paper["id"]: paper for paper in existing}
    for paper in new_papers:
        if paper["id"] not in merged:
            merged[paper["id"]] = paper
            continue
        old = merged[paper["id"]]
        old["updated"] = max(old.get("updated", ""), paper.get("updated", ""))
        old["categories"] = sorted(set(old.get("categories", []) + paper.get("categories", [])))
        old["source_queries"] = sorted(set(old.get("source_queries", []) + paper.get("source_queries", [])))
        old["code_urls"] = sorted(set(old.get("code_urls", []) + paper.get("code_urls", [])))
    return sorted(merged.values(), key=lambda item: (item["published"], item["updated"], item["id"]), reverse=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect(config: dict[str, Any], sleep_seconds: float) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    categories = config["arxiv_categories"]
    page_size = int(config.get("max_results_per_page", 100))
    max_per_query = int(config.get("max_results_per_query", page_size))
    for index, query_config in enumerate(config["queries"], start=1):
        query = build_search_query(query_config["query"], categories)
        print(f"[{index}/{len(config['queries'])}] {query_config['name']}")
        start = 0
        while start < max_per_query:
            batch_size = min(page_size, max_per_query - start)
            feed = fetch_feed(query, start=start, max_results=batch_size)
            entries = feed.findall("atom:entry", ATOM_NS)
            if not entries:
                break
            total_results = min(feed_total_results(feed), max_per_query)
            for entry in entries:
                paper = extract_entry(entry, query_config["name"])
                if match_relevance(paper, config):
                    papers.append(paper)
            start += len(entries)
            if start >= total_results:
                break
            time.sleep(sleep_seconds)
    return papers


def authors_short(authors: list[str], limit: int = 4) -> str:
    if len(authors) <= limit:
        return ", ".join(authors)
    return ", ".join(authors[:limit]) + ", et al."


def markdown_cell(value: str) -> str:
    return normalize_space(value).replace("|", "\\|")


def markdown_table(papers: list[dict[str, Any]], limit: int | None = None) -> str:
    rows = [
        "| Date | Paper | Authors | Categories |",
        "| --- | --- | --- | --- |",
    ]
    shown = papers[:limit] if limit else papers
    for paper in shown:
        title = markdown_cell(paper["title"])
        authors = markdown_cell(authors_short(paper.get("authors", [])))
        categories = markdown_cell(", ".join(paper.get("categories", [])[:4]))
        rows.append(
            f"| {paper['published']} | [{title}]({paper['arxiv_url']}) / [pdf]({paper['pdf_url']}) | {authors} | {categories} |"
        )
    return "\n".join(rows)


def render_readme(papers: list[dict[str, Any]], generated_at: str) -> str:
    years = sorted({paper["published"][:4] for paper in papers}, reverse=True)
    lines = [
        "# Steering Activation Paper Arxiv",
        "",
        "A daily arXiv collector for papers about steering activations and internal representations in LLMs, VLMs, diffusion language models, multimodal foundation models, and related language-model systems.",
        "",
        f"Last updated: {generated_at}",
        "",
        f"Total papers: **{len(papers)}**",
        "",
        "## Papers By Year",
        "",
    ]
    for year in years:
        year_papers = [paper for paper in papers if paper["published"].startswith(year)]
        lines.extend([f"### {year}", "", markdown_table(year_papers), ""])
    lines.extend(
        [
            "## Collection Scope",
            "",
            "The collector searches for activation steering, activation addition, steering vectors, representation engineering, activation intervention, latent or feature steering, activation patching/editing, and related mechanistic-interpretability terms across language, multimodal, vision-language, diffusion-language, and foundation-model papers.",
            "",
            "## Update",
            "",
            "Run locally:",
            "",
            "```bash",
            "python3 daily_arxiv.py",
            "```",
            "",
            "The GitHub Action in `.github/workflows/daily_arxiv.yml` can run the same update automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--readme", type=Path, default=README_PATH)
    parser.add_argument("--sleep", type=float, default=3.0, help="Pause between arXiv API requests.")
    parser.add_argument("--no-fetch", action="store_true", help="Only regenerate README from existing JSON.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing JSON and rebuild from fetched results.")
    args = parser.parse_args()

    config = load_json(args.config, {})
    data = load_json(args.data, {"papers": []})
    existing = [] if args.rebuild else data.get("papers", [])
    new_papers = [] if args.no_fetch else collect(config, sleep_seconds=args.sleep)
    papers = merge_papers(existing, new_papers)
    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    save_json(args.data, {"generated_at": generated_at, "papers": papers})
    args.readme.write_text(render_readme(papers, generated_at), encoding="utf-8")
    print(f"Saved {len(papers)} papers to {args.data}")


if __name__ == "__main__":
    main()
