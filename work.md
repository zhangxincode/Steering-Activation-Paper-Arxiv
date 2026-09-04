# Work Commands

## Enter Project

```bash
cd Steering-activation-paper-arxiv
```

## Daily Update

Run all configured sources with default limits. This is additive and keeps existing papers.

```bash
python3 daily_arxiv.py
```

If network connections are unstable, use a slower interval:

```bash
python3 daily_arxiv.py --sleep 8
```

## Run Selected Sources

Run all supported sources explicitly:

```bash
python3 daily_arxiv.py --sources arxiv,dblp,semantic_scholar,openalex,acl_anthology,openreview
```

Run only arXiv:

```bash
python3 daily_arxiv.py --sources arxiv
```

Run only DBLP:

```bash
python3 daily_arxiv.py --sources dblp
```

Run only Semantic Scholar:

```bash
python3 daily_arxiv.py --sources semantic_scholar
```

Run only OpenAlex:

```bash
python3 daily_arxiv.py --sources openalex
```

Run only ACL Anthology:

```bash
python3 daily_arxiv.py --sources acl_anthology
```

Run only OpenReview:

```bash
python3 daily_arxiv.py --sources openreview
```

## Deep Backfill

Deep backfill means fetching more historical results per query. It is slower, but helps recover older papers.

```bash
python3 daily_arxiv.py --max-results-per-query 500 --max-results-per-page 50 --sleep 8
```

Deep backfill for only arXiv:

```bash
python3 daily_arxiv.py --sources arxiv --max-results-per-query 1500 --max-results-per-page 50 --sleep 8
```

## Regenerate README Only

Use existing `docs/arxiv-daily.json` and regenerate `README.md` without fetching from the web.

```bash
python3 daily_arxiv.py --no-fetch
```

## Rebuild From Scratch

This ignores the existing JSON and rebuilds the paper database from fetched results. Use carefully because this can remove papers that are not returned in the current run.

```bash
python3 daily_arxiv.py --rebuild
```

## Checks

Check Python syntax:

```bash
PYTHONPYCACHEPREFIX=/tmp/steering_activation_pycache python3 -m py_compile daily_arxiv.py
```

Check JSON config syntax:

```bash
python3 -m json.tool config/queries.json
```
