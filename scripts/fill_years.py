"""Fill null first_publish_year in a chunks JSONL from Open Library's search index.

    python fill_years.py data/chunks/chunks.jsonl data/chunks/chunks_filled_years.jsonl

Only the `first_publish_year` field is changed; `text` and everything else are
written back byte-for-byte as parsed JSON. Works that already have a year are left
alone. The year comes from Open Library's search index, which computes it from the
work's editions (the same "First published in" shown on openlibrary.org).
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict

API = "https://openlibrary.org/search.json"
BATCH = 50
HEADERS = {"User-Agent": "booksbot-year-filler/1.0 (contact: you@example.com)"}  # OL asks for one


def fetch_years(keys: list[str]) -> dict[str, int]:
    q = "key:(" + " OR ".join(f"/works/{k}" for k in keys) + ")"
    url = API + "?" + urllib.parse.urlencode(
        {"q": q, "fields": "key,first_publish_year", "limit": len(keys)})
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                docs = json.load(r).get("docs", [])
            return {d["key"].rsplit("/", 1)[-1]: d["first_publish_year"]
                    for d in docs if isinstance(d.get("first_publish_year"), int)}
        except Exception as e:
            wait = 2 ** attempt * 5
            print(f"  request failed ({e}); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    print(f"  giving up on batch starting {keys[0]}", file=sys.stderr)
    return {}


def main(src: str, dst: str):
    rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]

    by_work = defaultdict(list)
    for r in rows:
        by_work[r["work_id"]].append(r)
    missing = [w for w, rs in by_work.items()
               if all(r.get("first_publish_year") is None for r in rs)]
    print(f"{len(by_work)} works, {len(missing)} with no year; looking them up…")

    found: dict[str, int] = {}
    for i in range(0, len(missing), BATCH):
        found.update(fetch_years(missing[i:i + BATCH]))
        print(f"  {min(i + BATCH, len(missing))}/{len(missing)} checked, {len(found)} found")
        time.sleep(1)  # be polite to openlibrary.org

    chunks_updated = 0
    for w, year in found.items():
        for r in by_work[w]:
            r["first_publish_year"] = year
            chunks_updated += 1

    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    works_null = sum(all(r.get("first_publish_year") is None for r in rs)
                     for rs in by_work.values())
    chunks_null = sum(r.get("first_publish_year") is None for r in rows)
    print(f"\nUpdated: {len(found)} works ({chunks_updated} chunks)")
    print(f"Still null: {works_null} works ({chunks_null} of {len(rows)} chunks)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python fill_years.py IN.jsonl OUT.jsonl")
    main(sys.argv[1], sys.argv[2])