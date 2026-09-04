#!/usr/bin/env python3
"""Collect papers about activation steering for language-model systems."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import time
import urllib.error
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
DBLP_API = "https://dblp.org/search/publ/api"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API = "https://api.openalex.org/works"
ACL_BIB_URL = "https://aclanthology.org/anthology+abstracts.bib.gz"
OPENREVIEW_SEARCH_API = "https://api2.openreview.net/notes/search"

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

RETRYABLE_ERRORS = (
    ConnectionResetError,
    TimeoutError,
    http.client.HTTPException,
    http.client.IncompleteRead,
    socket.timeout,
    ssl.SSLError,
    urllib.error.URLError,
)


def normalize_space(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_title(value: str | None) -> str:
    value = normalize_space(value).lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def extract_arxiv_id(text: str | None) -> str:
    for pattern in (
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?",
        r"\barxiv:([0-9]{4}\.[0-9]{4,5})(?:v\d+)?\b",
    ):
        match = re.search(pattern, text or "", re.I)
        if match:
            return strip_version(match.group(1))
    return ""


def parse_arxiv_datetime(value: str) -> str:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()


def parse_year(value: Any) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", str(value or ""))
    return match.group(0) if match else ""


def date_from_year(value: Any) -> str:
    year = parse_year(value)
    return f"{year}-01-01" if year else "0000-01-01"


def markdown_cell(value: str) -> str:
    return normalize_space(value).replace("|", "\\|")


def urlopen_with_retry(
    request: urllib.request.Request,
    retries: int = 5,
    timeout: int = 60,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except RETRYABLE_ERRORS as error:
            last_error = error
            if attempt == retries:
                break
            wait_seconds = min(60, 2**attempt)
            print(f"Request failed on attempt {attempt}/{retries}: {error}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}") from last_error


def fetch_json(
    url: str,
    params: dict[str, str],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers=headers or {"User-Agent": "steering-activation-paper-arxiv/1.0"},
    )
    return json.loads(urlopen_with_retry(request).decode("utf-8"))


def fetch_xml(url: str, params: dict[str, str]) -> ET.Element:
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": "steering-activation-paper-arxiv/1.0"},
    )
    return ET.fromstring(urlopen_with_retry(request))


def extract_code_urls(text: str | None) -> list[str]:
    url_pattern = re.compile(
        r"https?://(?:www\.)?(?:github|gitlab)\.com/[^\s)\]}>,.;]+",
        re.I,
    )
    return sorted(set(match.group(0) for match in url_pattern.finditer(text or "")))


def stable_id(source: str, title: str, url: str = "", doi: str = "") -> str:
    arxiv_id = extract_arxiv_id(f"{title} {url}")
    if arxiv_id:
        return arxiv_id
    if doi:
        return f"doi:{doi.lower()}"
    digest = hashlib.sha1(f"{normalize_title(title)}|{url}".encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def relevance_score(paper: dict[str, Any], config: dict[str, Any]) -> int:
    haystack = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    score = 0
    score += 4 * sum(1 for term in config.get("strong_topic_terms", []) if term.lower() in haystack)
    score += 2 * sum(1 for term in config.get("weak_topic_terms", []) if term.lower() in haystack)
    score += 2 * sum(1 for term in config.get("model_terms", []) if term.lower() in haystack)
    score += sum(1 for term in config.get("adjacent_terms", []) if term.lower() in haystack)
    return score


def match_relevance(paper: dict[str, Any], config: dict[str, Any]) -> bool:
    haystack = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    strong_terms = [term.lower() for term in config.get("strong_topic_terms", [])]
    weak_terms = [term.lower() for term in config.get("weak_topic_terms", [])]
    model_terms = [term.lower() for term in config.get("model_terms", [])]
    steering_words = [term.lower() for term in config.get("steering_words", [])]

    has_model = any(term in haystack for term in model_terms)
    has_strong_topic = any(term in haystack for term in strong_terms)
    has_weak_topic = any(term in haystack for term in weak_terms)
    has_steering_word = any(
        re.search(rf"\b{re.escape(term)}\w*\b", haystack)
        for term in steering_words
    )
    return has_strong_topic or (has_model and has_weak_topic and has_steering_word)


def category_query(categories: list[str]) -> str:
    return " OR ".join(f"cat:{category}" for category in categories)


def build_arxiv_query(query: str, categories: list[str]) -> str:
    return f"({query}) AND ({category_query(categories)})" if categories else query


def feed_total_results(feed: ET.Element) -> int:
    total = feed.findtext(
        "{http://a9.com/-/spec/opensearch/1.1/}totalResults",
        default="0",
    )
    return int(total)


def paper_id_from_arxiv_entry(entry: ET.Element) -> str:
    raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
    return strip_version(raw_id.rsplit("/", 1)[-1])


def arxiv_paper(entry: ET.Element, source_query: str) -> dict[str, Any]:
    arxiv_id = paper_id_from_arxiv_entry(entry)
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
    return {
        "id": arxiv_id,
        "source": "arxiv",
        "sources": ["arxiv"],
        "title": title,
        "authors": authors,
        "published": parse_arxiv_datetime(entry.findtext("atom:published", default="", namespaces=ATOM_NS)),
        "updated": parse_arxiv_datetime(entry.findtext("atom:updated", default="", namespaces=ATOM_NS)),
        "categories": categories,
        "abstract": abstract,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "primary_category": categories[0] if categories else "",
        "source_queries": [source_query],
        "code_urls": extract_code_urls(abstract),
    }


def collect_arxiv(config: dict[str, Any], sleep_seconds: float) -> list[dict[str, Any]]:
    provider = config.get("providers", {}).get("arxiv", {})
    if not provider.get("enabled", True):
        print("[arxiv] skipped")
        return []

    papers: list[dict[str, Any]] = []
    categories = config.get("arxiv_categories", [])
    page_size = int(provider.get("max_results_per_page", config.get("max_results_per_page", 50)))
    max_per_query = int(provider.get("max_results_per_query", config.get("max_results_per_query", 500)))
    queries = config.get("queries", [])

    for index, query_config in enumerate(queries, start=1):
        print(f"[arxiv {index}/{len(queries)}] {query_config['name']}")
        query = build_arxiv_query(query_config["query"], categories)
        start = 0
        while start < max_per_query:
            params = {
                "search_query": query,
                "start": str(start),
                "max_results": str(min(page_size, max_per_query - start)),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            try:
                feed = fetch_xml(ARXIV_API, params)
            except RuntimeError as error:
                print(f"[arxiv] skip remaining pages for {query_config['name']} at start={start}: {error}")
                break
            entries = feed.findall("atom:entry", ATOM_NS)
            if not entries:
                break
            total_results = min(feed_total_results(feed), max_per_query)
            for entry in entries:
                paper = arxiv_paper(entry, query_config["name"])
                if match_relevance(paper, config):
                    paper["relevance_score"] = relevance_score(paper, config)
                    papers.append(paper)
            start += len(entries)
            if start >= total_results:
                break
            time.sleep(sleep_seconds)
    return papers


def parse_dblp_authors(authors_value: Any) -> list[str]:
    if isinstance(authors_value, dict):
        author = authors_value.get("author", [])
    else:
        author = authors_value or []
    if isinstance(author, dict):
        return [normalize_space(author.get("text", ""))]
    return [normalize_space(item.get("text", "")) for item in author if isinstance(item, dict)]


def dblp_paper(hit: dict[str, Any], source_query: str) -> dict[str, Any]:
    info = hit.get("info", {})
    title = normalize_space(info.get("title", ""))
    url = info.get("ee") or info.get("url", "")
    year = date_from_year(info.get("year"))
    return {
        "id": stable_id("dblp", title, url),
        "source": "dblp",
        "sources": ["dblp"],
        "title": title,
        "authors": parse_dblp_authors(info.get("authors")),
        "published": year,
        "updated": year,
        "categories": [category for category in ["DBLP", info.get("venue", ""), info.get("type", "")] if category],
        "abstract": "",
        "url": url,
        "pdf_url": f"https://arxiv.org/pdf/{extract_arxiv_id(url)}" if extract_arxiv_id(url) else "",
        "primary_category": "DBLP",
        "source_queries": [source_query],
        "code_urls": [],
        "dblp": {"key": info.get("key"), "url": info.get("url")},
    }


def collect_dblp(config: dict[str, Any], sleep_seconds: float) -> list[dict[str, Any]]:
    provider = config.get("providers", {}).get("dblp", {})
    if not provider.get("enabled", True):
        print("[dblp] skipped")
        return []

    papers: list[dict[str, Any]] = []
    page_size = int(provider.get("page_size", 100))
    max_per_query = int(provider.get("max_results_per_query", 300))
    queries = provider.get("queries") or config.get("queries", [])
    for index, query_config in enumerate(queries, start=1):
        print(f"[dblp {index}/{len(queries)}] {query_config['name']}")
        start = 0
        while start < max_per_query:
            try:
                payload = fetch_json(
                    DBLP_API,
                    {
                        "q": query_config["query"],
                        "format": "json",
                        "h": str(min(page_size, max_per_query - start)),
                        "f": str(start),
                    },
                )
            except RuntimeError as error:
                print(f"[dblp] skip remaining pages for {query_config['name']} at start={start}: {error}")
                break
            hits = payload.get("result", {}).get("hits", {}).get("hit", [])
            if not hits:
                break
            for hit in hits:
                paper = dblp_paper(hit, query_config["name"])
                if match_relevance(paper, config):
                    paper["relevance_score"] = relevance_score(paper, config)
                    papers.append(paper)
            if len(hits) < page_size:
                break
            start += len(hits)
            time.sleep(sleep_seconds)
    return papers


def semantic_paper(result: dict[str, Any], source_query: str) -> dict[str, Any]:
    title = normalize_space(result.get("title", ""))
    abstract = normalize_space(result.get("abstract", ""))
    external_ids = result.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv") or extract_arxiv_id(f"{title} {abstract} {result.get('url', '')}")
    doi = external_ids.get("DOI", "")
    year = date_from_year(result.get("year"))
    url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else result.get("url", "")
    return {
        "id": arxiv_id or stable_id("semantic-scholar", title, url, doi),
        "source": "semantic-scholar",
        "sources": ["semantic-scholar"],
        "title": title,
        "authors": [normalize_space(author.get("name", "")) for author in result.get("authors", []) if author.get("name")],
        "published": year,
        "updated": year,
        "categories": [category for category in ["Semantic Scholar", result.get("venue", "")] if category],
        "abstract": abstract,
        "url": url,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
        "primary_category": "Semantic Scholar",
        "source_queries": [source_query],
        "code_urls": extract_code_urls(abstract),
        "semantic_scholar": {
            "paper_id": result.get("paperId"),
            "citation_count": result.get("citationCount"),
            "external_ids": external_ids,
        },
    }


def collect_semantic_scholar(config: dict[str, Any], sleep_seconds: float) -> list[dict[str, Any]]:
    provider = config.get("providers", {}).get("semantic_scholar", {})
    if not provider.get("enabled", True):
        print("[semantic-scholar] skipped")
        return []

    papers: list[dict[str, Any]] = []
    limit = int(provider.get("limit", 100))
    max_per_query = int(provider.get("max_results_per_query", 300))
    queries = provider.get("queries") or config.get("queries", [])
    fields = "paperId,title,abstract,year,authors,venue,url,externalIds,citationCount"
    headers = {"User-Agent": "steering-activation-paper-arxiv/1.0"}
    api_key = provider.get("api_key_env") and os.environ.get(provider["api_key_env"], "")
    if api_key:
        headers["x-api-key"] = api_key

    for index, query_config in enumerate(queries, start=1):
        print(f"[semantic-scholar {index}/{len(queries)}] {query_config['name']}")
        offset = 0
        while offset < max_per_query:
            request = urllib.request.Request(
                f"{SEMANTIC_SCHOLAR_API}?{urllib.parse.urlencode({'query': query_config['query'], 'offset': str(offset), 'limit': str(min(limit, max_per_query - offset)), 'fields': fields})}",
                headers=headers,
            )
            try:
                payload = json.loads(urlopen_with_retry(request).decode("utf-8"))
            except RuntimeError as error:
                print(f"[semantic-scholar] skip remaining pages for {query_config['name']} at offset={offset}: {error}")
                break
            results = payload.get("data", [])
            if not results:
                break
            for result in results:
                paper = semantic_paper(result, query_config["name"])
                if match_relevance(paper, config):
                    paper["relevance_score"] = relevance_score(paper, config)
                    papers.append(paper)
            if len(results) < limit:
                break
            offset += len(results)
            time.sleep(sleep_seconds)
    return papers


def openalex_abstract(work: dict[str, Any]) -> str:
    inverted = work.get("abstract_inverted_index") or {}
    words: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        for position in positions:
            words.append((position, word))
    return normalize_space(" ".join(word for _, word in sorted(words)))


def openalex_paper(work: dict[str, Any], source_query: str) -> dict[str, Any]:
    title = normalize_space(work.get("title", ""))
    abstract = openalex_abstract(work)
    primary_location = work.get("primary_location") or {}
    landing_url = primary_location.get("landing_page_url") or work.get("id", "")
    pdf_url = primary_location.get("pdf_url") or ""
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    arxiv_id = extract_arxiv_id(f"{title} {abstract} {landing_url} {pdf_url}")
    concepts = [concept.get("display_name", "") for concept in work.get("concepts", [])[:4]]
    return {
        "id": arxiv_id or stable_id("openalex", title, landing_url, doi),
        "source": "openalex",
        "sources": ["openalex"],
        "title": title,
        "authors": [
            normalize_space((authorship.get("author") or {}).get("display_name", ""))
            for authorship in work.get("authorships", [])
            if (authorship.get("author") or {}).get("display_name")
        ],
        "published": work.get("publication_date") or date_from_year(work.get("publication_year")),
        "updated": (work.get("updated_date") or "")[:10] or work.get("publication_date") or "",
        "categories": [category for category in ["OpenAlex", *concepts] if category],
        "abstract": abstract,
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else landing_url,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else pdf_url,
        "primary_category": "OpenAlex",
        "source_queries": [source_query],
        "code_urls": extract_code_urls(abstract),
        "openalex": {"id": work.get("id"), "doi": doi, "cited_by_count": work.get("cited_by_count")},
    }


def collect_openalex(config: dict[str, Any], sleep_seconds: float) -> list[dict[str, Any]]:
    provider = config.get("providers", {}).get("openalex", {})
    if not provider.get("enabled", True):
        print("[openalex] skipped")
        return []

    papers: list[dict[str, Any]] = []
    per_page = int(provider.get("per_page", 100))
    max_pages = int(provider.get("max_pages_per_query", 3))
    queries = provider.get("queries") or config.get("queries", [])
    mailto = os.environ.get("OPENALEX_MAILTO", provider.get("mailto", ""))
    for index, query_config in enumerate(queries, start=1):
        print(f"[openalex {index}/{len(queries)}] {query_config['name']}")
        cursor = "*"
        for _page in range(max_pages):
            params = {
                "search": query_config["query"],
                "per-page": str(per_page),
                "cursor": cursor,
                "sort": "publication_date:desc",
            }
            if mailto:
                params["mailto"] = mailto
            try:
                payload = fetch_json(OPENALEX_API, params)
            except RuntimeError as error:
                print(f"[openalex] skip remaining pages for {query_config['name']}: {error}")
                break
            for work in payload.get("results", []):
                paper = openalex_paper(work, query_config["name"])
                if match_relevance(paper, config):
                    paper["relevance_score"] = relevance_score(paper, config)
                    papers.append(paper)
            cursor = payload.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(sleep_seconds)
    return papers


def parse_bib_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for block in re.split(r"\n@", text):
        block = block if block.startswith("@") else "@" + block
        title = re.search(r"\btitle\s*=\s*[{\"](.+?)[}\"]\s*,?\n", block, re.I | re.S)
        year = re.search(r"\byear\s*=\s*[{\"]?(\d{4})", block, re.I)
        url = re.search(r"\burl\s*=\s*[{\"](.+?)[}\"]\s*,?\n", block, re.I | re.S)
        abstract = re.search(r"\babstract\s*=\s*[{\"](.+?)[}\"]\s*,?\n", block, re.I | re.S)
        author = re.search(r"\bauthor\s*=\s*[{\"](.+?)[}\"]\s*,?\n", block, re.I | re.S)
        if title:
            entries.append(
                {
                    "title": normalize_space(title.group(1)),
                    "year": year.group(1) if year else "",
                    "url": normalize_space(url.group(1)) if url else "",
                    "abstract": normalize_space(abstract.group(1)) if abstract else "",
                    "author": normalize_space(author.group(1)) if author else "",
                }
            )
    return entries


def acl_paper(entry: dict[str, str], source_query: str) -> dict[str, Any]:
    title = entry["title"]
    url = entry.get("url", "")
    arxiv_id = extract_arxiv_id(f"{title} {url}")
    published = date_from_year(entry.get("year"))
    return {
        "id": arxiv_id or stable_id("acl-anthology", title, url),
        "source": "acl-anthology",
        "sources": ["acl-anthology"],
        "title": title,
        "authors": [normalize_space(author) for author in re.split(r"\s+and\s+", entry.get("author", "")) if author.strip()],
        "published": published,
        "updated": published,
        "categories": ["ACL Anthology"],
        "abstract": entry.get("abstract", ""),
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
        "primary_category": "ACL Anthology",
        "source_queries": [source_query],
        "code_urls": extract_code_urls(entry.get("abstract", "")),
    }


def collect_acl_anthology(config: dict[str, Any], _sleep_seconds: float) -> list[dict[str, Any]]:
    provider = config.get("providers", {}).get("acl_anthology", {})
    if not provider.get("enabled", True):
        print("[acl-anthology] skipped")
        return []

    print("[acl-anthology] downloading bibliography")
    try:
        request = urllib.request.Request(ACL_BIB_URL, headers={"User-Agent": "steering-activation-paper-arxiv/1.0"})
        raw = gzip.decompress(urlopen_with_retry(request, timeout=120)).decode("utf-8", errors="replace")
    except (RuntimeError, OSError) as error:
        print(f"[acl-anthology] skipped: {error}")
        return []

    papers: list[dict[str, Any]] = []
    queries = provider.get("queries") or config.get("queries", [])
    entries = parse_bib_entries(raw)
    max_matches = int(provider.get("max_matches", 500))
    for query_config in queries:
        terms = [
            term.lower().strip('"')
            for term in re.findall(r'"[^"]+"|[A-Za-z][A-Za-z-]+', query_config["query"])
        ]
        for entry in entries:
            haystack = f"{entry.get('title', '')} {entry.get('abstract', '')}".lower()
            if any(term and term in haystack for term in terms):
                paper = acl_paper(entry, query_config["name"])
                if match_relevance(paper, config):
                    paper["relevance_score"] = relevance_score(paper, config)
                    papers.append(paper)
                    if len(papers) >= max_matches:
                        return papers
    return papers


def openreview_paper(note: dict[str, Any], source_query: str) -> dict[str, Any]:
    content = note.get("content", {})
    title_value = content.get("title", {})
    abstract_value = content.get("abstract", {})
    authors_value = content.get("authors", {})
    title = normalize_space(title_value.get("value") if isinstance(title_value, dict) else title_value)
    abstract = normalize_space(abstract_value.get("value") if isinstance(abstract_value, dict) else abstract_value)
    authors = authors_value.get("value") if isinstance(authors_value, dict) else authors_value
    if not isinstance(authors, list):
        authors = []
    venue = content.get("venue", {})
    venue_value = venue.get("value") if isinstance(venue, dict) else venue
    cdate = note.get("cdate") or note.get("tcdate") or 0
    year = dt.datetime.utcfromtimestamp(cdate / 1000).year if cdate else ""
    url = f"https://openreview.net/forum?id={note.get('forum') or note.get('id')}"
    arxiv_id = extract_arxiv_id(f"{title} {abstract}")
    return {
        "id": arxiv_id or stable_id("openreview", title, url),
        "source": "openreview",
        "sources": ["openreview"],
        "title": title,
        "authors": [normalize_space(author) for author in authors],
        "published": date_from_year(year),
        "updated": date_from_year(year),
        "categories": [category for category in ["OpenReview", venue_value] if category],
        "abstract": abstract,
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
        "primary_category": "OpenReview",
        "source_queries": [source_query],
        "code_urls": extract_code_urls(abstract),
        "openreview": {"id": note.get("id"), "forum": note.get("forum")},
    }


def collect_openreview(config: dict[str, Any], sleep_seconds: float) -> list[dict[str, Any]]:
    provider = config.get("providers", {}).get("openreview", {})
    if not provider.get("enabled", True):
        print("[openreview] skipped")
        return []

    papers: list[dict[str, Any]] = []
    limit = int(provider.get("limit", 50))
    max_per_query = int(provider.get("max_results_per_query", 150))
    queries = provider.get("queries") or config.get("queries", [])
    for index, query_config in enumerate(queries, start=1):
        print(f"[openreview {index}/{len(queries)}] {query_config['name']}")
        offset = 0
        while offset < max_per_query:
            try:
                payload = fetch_json(
                    OPENREVIEW_SEARCH_API,
                    {"term": query_config["query"], "limit": str(min(limit, max_per_query - offset)), "offset": str(offset)},
                )
            except RuntimeError as error:
                print(f"[openreview] skip remaining pages for {query_config['name']} at offset={offset}: {error}")
                break
            notes = payload.get("notes") or payload.get("results") or []
            if not notes:
                break
            for note in notes:
                paper = openreview_paper(note, query_config["name"])
                if match_relevance(paper, config):
                    paper["relevance_score"] = relevance_score(paper, config)
                    papers.append(paper)
            if len(notes) < limit:
                break
            offset += len(notes)
            time.sleep(sleep_seconds)
    return papers


PROVIDERS = {
    "arxiv": collect_arxiv,
    "dblp": collect_dblp,
    "semantic_scholar": collect_semantic_scholar,
    "openalex": collect_openalex,
    "acl_anthology": collect_acl_anthology,
    "openreview": collect_openreview,
}


def dedup_keys(paper: dict[str, Any]) -> list[str]:
    keys = [paper.get("id", "")]
    arxiv_id = extract_arxiv_id(f"{paper.get('url', '')} {paper.get('pdf_url', '')} {paper.get('abstract', '')}")
    if arxiv_id:
        keys.append(arxiv_id)
    doi = (
        (paper.get("semantic_scholar") or {}).get("external_ids", {}).get("DOI")
        or (paper.get("openalex") or {}).get("doi", "")
    )
    if doi:
        keys.append(f"doi:{doi.lower()}")
    title_key = "title:" + normalize_title(paper.get("title", ""))
    if len(title_key) > 20:
        keys.append(title_key)
    return [key for key in keys if key]


def merge_record(old: dict[str, Any], paper: dict[str, Any]) -> dict[str, Any]:
    old["updated"] = max(old.get("updated", ""), paper.get("updated", ""))
    old["categories"] = sorted(set(old.get("categories", []) + paper.get("categories", [])))
    old["source_queries"] = sorted(set(old.get("source_queries", []) + paper.get("source_queries", [])))
    old["sources"] = sorted(set(old.get("sources", [old.get("source", "")]) + paper.get("sources", [paper.get("source", "")])))
    old["code_urls"] = sorted(set(old.get("code_urls", []) + paper.get("code_urls", [])))
    old["relevance_score"] = max(old.get("relevance_score", 0), paper.get("relevance_score", 0))
    for key in ("title", "authors", "published", "abstract", "url", "pdf_url", "primary_category"):
        if (not old.get(key) or old.get(key) == "0000-01-01") and paper.get(key):
            old[key] = paper[key]
    for key in ("dblp", "semantic_scholar", "openalex", "openreview"):
        if paper.get(key) and not old.get(key):
            old[key] = paper[key]
    return old


def merge_papers(existing: list[dict[str, Any]], new_papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    key_to_id: dict[str, str] = {}

    for paper in existing + new_papers:
        if "url" not in paper and paper.get("arxiv_url"):
            paper["url"] = paper["arxiv_url"]
        paper.setdefault("source", "arxiv")
        paper.setdefault("sources", [paper.get("source", "")])
        paper.setdefault("relevance_score", 0)
        keys = dedup_keys(paper)
        target_id = next((key_to_id[key] for key in keys if key in key_to_id), paper.get("id") or keys[0])
        if target_id in merged:
            merged[target_id] = merge_record(merged[target_id], paper)
        else:
            merged[target_id] = paper
        for key in keys:
            key_to_id[key] = target_id

    return sorted(
        merged.values(),
        key=lambda item: (item.get("published", ""), item.get("updated", ""), item.get("id", "")),
        reverse=True,
    )


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect(config: dict[str, Any], sleep_seconds: float, only_sources: set[str] | None) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    providers = config.get("providers", {})
    for name, collector in PROVIDERS.items():
        if only_sources and name not in only_sources:
            continue
        if not providers.get(name, {}).get("enabled", True):
            print(f"[{name}] skipped")
            continue
        try:
            papers.extend(collector(config, sleep_seconds))
        except Exception as error:
            print(f"[{name}] skipped after unexpected error: {error}")
    return papers


def authors_short(authors: list[str], limit: int = 4) -> str:
    if len(authors) <= limit:
        return ", ".join(authors)
    return ", ".join(authors[:limit]) + ", et al."


def paper_link(paper: dict[str, Any]) -> str:
    title = markdown_cell(paper.get("title", ""))
    url = paper.get("url") or paper.get("arxiv_url", "")
    pdf_url = paper.get("pdf_url", "")
    link = f"[{title}]({url})" if url else title
    if pdf_url:
        link += f" / [pdf]({pdf_url})"
    return link


def markdown_table(papers: list[dict[str, Any]]) -> str:
    rows = [
        "| Date | Paper | Authors | Categories |",
        "| --- | --- | --- | --- |",
    ]
    for paper in papers:
        authors = markdown_cell(authors_short(paper.get("authors", [])))
        categories = markdown_cell(", ".join(paper.get("categories", [])[:4]))
        rows.append(f"| {paper.get('published', '')} | {paper_link(paper)} | {authors} | {categories} |")
    return "\n".join(rows)


def source_counts(papers: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for paper in papers:
        for source in paper.get("sources", [paper.get("source", "unknown")]):
            if source:
                counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def render_readme(papers: list[dict[str, Any]], generated_at: str) -> str:
    years = sorted({paper.get("published", "")[:4] for paper in papers if paper.get("published")}, reverse=True)
    counts = source_counts(papers)
    lines = [
        "# Steering Activation Paper Arxiv",
        "",
        "A broad, additive collector for activation steering and internal-representation control in LLMs, VLMs, diffusion language models, multimodal foundation models, and related language-model systems.",
        "",
        f"Last updated: {generated_at}",
        "",
        f"Total papers: **{len(papers)}**",
        "",
        "Sources: " + ", ".join(f"{source} ({count})" for source, count in counts.items()),
        "",
        "## Papers By Year",
        "",
    ]
    for year in years:
        year_papers = [paper for paper in papers if paper.get("published", "").startswith(year)]
        lines.extend([f"### {year}", "", markdown_table(year_papers), ""])
    lines.extend(
        [
            "## Collection Scope",
            "",
            "The collector searches arXiv, DBLP, Semantic Scholar, OpenAlex, ACL Anthology, and OpenReview. Updates are additive by default; existing papers are kept unless `--rebuild` is explicitly used.",
            "",
            "## Update",
            "",
            "Run locally:",
            "",
            "```bash",
            "python3 daily_arxiv.py",
            "```",
            "",
            "Run only selected sources:",
            "",
            "```bash",
            "python3 daily_arxiv.py --sources arxiv,dblp,semantic_scholar,openalex,acl_anthology,openreview",
            "```",
            "",
            "Run a deeper backfill by raising the shared provider limits:",
            "",
            "```bash",
            "python3 daily_arxiv.py --max-results-per-query 500 --max-results-per-page 50 --sleep 8",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--readme", type=Path, default=README_PATH)
    parser.add_argument("--sleep", type=float, default=5.0, help="Pause between paginated provider requests.")
    parser.add_argument("--sources", help="Comma-separated providers to run.")
    parser.add_argument("--max-results-per-query", type=int, help="Override per-query result limits for all providers.")
    parser.add_argument("--max-results-per-page", type=int, help="Override per-page result limits for all providers.")
    parser.add_argument("--no-fetch", action="store_true", help="Only regenerate README from existing JSON.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore existing JSON and rebuild from fetched results.")
    args = parser.parse_args()

    config = load_json(args.config, {})
    providers = config.setdefault("providers", {})
    if args.max_results_per_query is not None:
        config["max_results_per_query"] = args.max_results_per_query
        for provider in providers.values():
            if "max_results_per_query" in provider:
                provider["max_results_per_query"] = args.max_results_per_query
            if "max_pages_per_query" in provider:
                provider["max_pages_per_query"] = 1
            if "max_matches" in provider:
                provider["max_matches"] = args.max_results_per_query
    if args.max_results_per_page is not None:
        config["max_results_per_page"] = args.max_results_per_page
        for provider in providers.values():
            for key in ("max_results_per_page", "page_size", "limit", "per_page"):
                if key in provider:
                    provider[key] = args.max_results_per_page
    only_sources = {name.strip() for name in args.sources.split(",")} if args.sources else None
    data = load_json(args.data, {"papers": []})
    existing = [] if args.rebuild else data.get("papers", [])
    new_papers = [] if args.no_fetch else collect(config, sleep_seconds=args.sleep, only_sources=only_sources)
    papers = merge_papers(existing, new_papers)
    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    save_json(args.data, {"generated_at": generated_at, "papers": papers})
    args.readme.write_text(render_readme(papers, generated_at), encoding="utf-8")
    print(f"Saved {len(papers)} papers to {args.data}")


if __name__ == "__main__":
    main()
