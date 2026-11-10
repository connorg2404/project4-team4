"""run_eval.py - runs the Librarian Bot test set against the API and scores it.

    python evaluation/run_eval.py                     # tune rows, non-streaming
    python evaluation/run_eval.py --label "chunk 300" # tag the run (shows up in history.csv)
    python evaluation/run_eval.py --stream            # also measures time to first token
    python evaluation/run_eval.py --split holdout     # final check only; do NOT tune on these
    python evaluation/run_eval.py --ids DL01,RC03     # a few rows while debugging

What it measures (the project rubric)
  hit@k           share of on-topic questions where a correct work is among the top-k sources
                  (the team target is 80%). A source is correct if its title is in expected_titles
                  OR its subjects/description contain one of accept_keywords.
  precision@k     average share of retrieved sources that are correct
  refusal acc.    share of should-refuse questions the API refused (target 100%)
  false refusals  on-topic questions the API refused
  owned books     recommendation rows where a source is a book the user said they already have
  latency         median / p95 of the API's latency_ms, wall-clock time, and time to first token
Rows with no hit rule (e.g. AU06) are listed as "manual" and left out of the percentages.

Outputs (in evaluation/results/): <time>_<label>.csv (every row), <time>_<label>_summary.md
(the table to share), and one new line in history.csv so progress can be tracked over time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PIPE = "|"
TARGET_HIT_RATE = 0.80
TARGET_REFUSAL_ACC = 1.0

# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------
def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

def split_user_book(entry: str) -> tuple[str, str]:
    """'Title by Author' -> (title, author). The author is '' when the entry has none."""
    if " by " in entry:
        title, author = entry.rsplit(" by ", 1)
        return title.strip(), author.strip()
    return entry.strip(), ""

def author_matches(user_author: str, source_authors: list[str]) -> bool:
    """Compare last names with a little fuzziness, so 'frank herbart' still matches Herbert."""
    if not norm(user_author):
        return True  # the user gave no author, so the title alone decides
    last = norm(user_author).split()[-1]
    return any(
        SequenceMatcher(None, last, norm(a).split()[-1]).ratio() >= 0.8
        for a in source_authors if norm(a)
    )

def is_owned(source: dict, owned: list[tuple[str, str]]) -> bool:
    """True if the source is (a collection containing) a book the user already has.
    Title and author must both match, so 'The Martian Chronicles' by Bradbury is NOT
    'The Martian' by Andy Weir."""
    title_words = norm(source.get("title", "")).split()
    for owned_title, owned_author in owned:
        want = norm(owned_title)
        if not want:
            continue
        prefix = " ".join(title_words[: len(want.split())])  # catches "Dune, Dune Messiah, ..."
        if SequenceMatcher(None, want, prefix).ratio() >= 0.85 and author_matches(
            owned_author, source.get("authors") or []
        ):
            return True
    return False

class Catalog:
    """The works file, used to look up a source's subjects and description for accept_keywords."""

    def __init__(self, path: Path):
        self.by_id: dict[str, str] = {}
        self.by_title: dict[str, str] = {}
        self.loaded = path.exists()
        if not self.loaded:
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                w = json.loads(line)
                text = (" | ".join(w.get("subjects", [])) + " || " + w.get("description", "")).lower()
                self.by_id[w["work_id"]] = text
                self.by_title[norm(w["title"])] = text

    def text_for(self, source: dict) -> str:
        m = re.search(r"/works/(OL\d+W)", source.get("source_url") or "")
        if m and m.group(1) in self.by_id:
            return self.by_id[m.group(1)]
        return self.by_title.get(norm(source.get("title", "")), "")

# ---------------------------------------------------------------------------
# Calling the API
# ---------------------------------------------------------------------------
def parse_sse(resp: requests.Response):
    resp.encoding = "utf-8"
    event, data_lines = "message", []
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        yield event, "\n".join(data_lines)

def token_of(data: str) -> str:
    if data.startswith("{"):
        try:
            payload = json.loads(data)
            if isinstance(payload, dict):
                return str(payload.get("token", payload.get("text", "")))
        except ValueError:
            pass
    return data

def call_api(api_url: str, payload: dict, stream: bool, timeout: float) -> dict:
    started = time.perf_counter()
    ttft_ms = None
    if stream:
        tokens, final = [], {}
        with requests.post(f"{api_url}/ask/stream", json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for event, data in parse_sse(r):
                if event == "sources":
                    final = json.loads(data)
                elif event == "error":
                    raise RuntimeError(f"stream error: {data}")
                elif event == "done":
                    break
                else:
                    if ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - started) * 1000)
                    tokens.append(token_of(data))
        body = {"answer": "".join(tokens), **final}
    else:
        r = requests.post(f"{api_url}/ask", json=payload, timeout=timeout)
        r.raise_for_status()
        body = r.json()
    body["wall_ms"] = int((time.perf_counter() - started) * 1000)
    body["ttft_ms"] = ttft_ms
    return body

# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def grade(row: dict, res: dict, catalog: Catalog, k: int) -> dict:
    out = {"passed": None, "hit": None, "precision": None, "owned_violation": None, "note": ""}
    if res.get("error"):
        out.update(passed=False, note="API error")
        return out
    sources = (res.get("sources") or [])[:k]
    refused = bool(res.get("refused"))

    if row["group"] == "bad":
        out["passed"] = refused
        if not refused:
            out["note"] = "should have refused"
        elif sources:
            out["note"] = "refused but still returned sources"
        return out

    expected = {norm(t) for t in row["expected_titles"].split(PIPE) if t}
    keywords = [kw for kw in row["accept_keywords"].split(PIPE) if kw] if catalog.loaded else []
    owned = [split_user_book(b) for b in row["user_books"].split(PIPE) if b]

    if owned:
        out["owned_violation"] = any(is_owned(s, owned) for s in sources)
    if not expected and not keywords:  # e.g. AU06, where "I can't count that" is a fine answer
        out["note"] = "manual: no hit rule"
        return out
    if refused:
        out.update(passed=False, hit=False, note="false refusal")
        return out

    relevant = []
    for s in sources:
        ok = norm(s.get("title", "")) in expected or any(kw in catalog.text_for(s) for kw in keywords)
        relevant.append(ok and not (owned and is_owned(s, owned)))
    out["hit"] = any(relevant)
    out["precision"] = sum(relevant) / len(sources) if sources else 0.0
    out["passed"] = out["hit"] and not out["owned_violation"]
    if out["owned_violation"]:
        out["note"] = "recommended a book the user already has"
    elif not out["hit"]:
        out["note"] = "no correct work in top-k"
    return out

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"

def p95(values: list[float]) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return v[max(0, math.ceil(0.95 * len(v)) - 1)]

def med(values: list[float]) -> float | None:
    return statistics.median(values) if values else None

def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None

def summarize(results: list[dict]) -> dict:
    good = [r for r in results if r["group"] == "good"]
    bad = [r for r in results if r["group"] == "bad"]
    graded = [r for r in good if r["hit"] is not None]
    lat = [r["latency_ms"] for r in results if isinstance(r["latency_ms"], (int, float))]
    wall = [r["wall_ms"] for r in results if isinstance(r["wall_ms"], (int, float))]
    ttft = [r["ttft_ms"] for r in results if isinstance(r["ttft_ms"], (int, float))]
    return {
        "n_good": len(good), "n_bad": len(bad),
        "n_graded": len(graded), "n_manual": len([r for r in good if r["passed"] is None]),
        "hit_rate": mean([1.0 if r["hit"] else 0.0 for r in graded]),
        "precision": mean([r["precision"] for r in graded if r["precision"] is not None]),
        "false_refusals": sum(1 for r in good if r["note"] == "false refusal"),
        "owned_violations": sum(1 for r in good if r["owned_violation"]),
        "n_owned_rows": sum(1 for r in good if r["owned_violation"] is not None),
        "refusal_acc": mean([1.0 if r["passed"] else 0.0 for r in bad]),
        "refusal_misses": [r["id"] for r in bad if not r["passed"]],
        "errors": sum(1 for r in results if r["error"]),
        "median_latency_ms": med(lat), "p95_latency_ms": p95(lat),
        "median_wall_ms": med(wall), "median_ttft_ms": med(ttft),
    }

def by_type_table(results: list[dict]) -> list[tuple]:
    rows = []
    for t in sorted({r["type"] for r in results}, key=lambda t: [r["type"] for r in results].index(t)):
        rs = [r for r in results if r["type"] == t and r["passed"] is not None]
        rows.append((t, len(rs), mean([1.0 if r["passed"] else 0.0 for r in rs])))
    return rows

def render_markdown(s: dict, results: list[dict], meta: dict) -> str:
    def ok(value, target):
        return "n/a" if value is None else ("PASS" if value >= target else "FAIL")

    ms = lambda v: "n/a" if v is None else f"{v:,.0f} ms"
    lines = [
        f"## Librarian Bot evaluation: {meta['label']}",
        "",
        f"Run {meta['timestamp']} | split: {meta['split']} | API model: {meta['model']} | "
        f"mode: {meta['mode']} | top_k: {meta['top_k']}",
        "",
        "| Metric | Result | Target |",
        "|---|---|---|",
        f"| hit@{meta['top_k']} (on-topic) | {pct(s['hit_rate'])} ({s['n_graded']} graded) | 80% {ok(s['hit_rate'], TARGET_HIT_RATE)} |",
        f"| precision@{meta['top_k']} | {pct(s['precision'])} | |",
        f"| Refusal accuracy | {pct(s['refusal_acc'])} ({s['n_bad']} rows) | 100% {ok(s['refusal_acc'], TARGET_REFUSAL_ACC)} |",
        f"| False refusals | {s['false_refusals']} of {s['n_good']} | 0 |",
        f"| Owned books recommended | {s['owned_violations']} of {s['n_owned_rows']} rows with a list | 0 |",
        f"| Median latency (API) | {ms(s['median_latency_ms'])} | |",
        f"| p95 latency (API) | {ms(s['p95_latency_ms'])} | |",
        f"| Median time to first token | {ms(s['median_ttft_ms'])} | |",
        "",
        "| Question type | Rows scored | Pass rate |",
        "|---|---|---|",
    ]
    lines += [f"| {t} | {n} | {pct(p)} |" for t, n, p in by_type_table(results)]
    failed = [f"{r['id']} ({r['note']})" for r in results if r["passed"] is False]
    manual = [r["id"] for r in results if r["passed"] is None and r["group"] == "good"]
    lines += ["", f"Failed rows: {', '.join(failed) if failed else 'none'}"]
    if manual:
        lines.append(f"Graded by hand (no hit rule): {', '.join(manual)}")
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Librarian Bot test set against the API.")
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--test-set", type=Path, default=HERE / "test_set.csv")
    ap.add_argument("--works", type=Path, default=HERE.parent / "data" / "raw" / "works.jsonl",
                    help="works file, used to check accept_keywords against retrieved sources")
    ap.add_argument("--split", choices=["tune", "holdout", "all"], default="tune")
    ap.add_argument("--ids", help="comma-separated row ids to run, e.g. DL01,RC03")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--stream", action="store_true", help="use /ask/stream and measure time to first token")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--label", default="run", help="short name for this run, e.g. 'chunk 300'")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    a = ap.parse_args()

    with open(a.test_set, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if a.ids:
        wanted = {i.strip() for i in a.ids.split(",")}
        rows = [r for r in rows if r["id"] in wanted]
    elif a.split != "all":
        rows = [r for r in rows if r["split"] == a.split]
    unresolved = [r["id"] for r in rows if "{{" in r["question"] + r["user_books"]]
    if unresolved:
        print(f"Skipping rows with unfilled {{{{slots}}}}: {', '.join(unresolved)}")
        rows = [r for r in rows if r["id"] not in unresolved]
    if not rows:
        print("No rows to run.")
        return 1

    catalog = Catalog(a.works)
    if not catalog.loaded:
        print(f"Note: {a.works} not found, so accept_keywords rules are skipped (expected_titles still used).")

    model = "unknown"
    try:
        h = requests.get(f"{a.api_url}/health", timeout=5)
        h.raise_for_status()
        model = str(h.json().get("model", "unknown"))
    except Exception as exc:  # noqa: BLE001
        print(f"Can't reach {a.api_url}/health ({exc}). Is the API running?")
        return 1
    if model == "mock":
        print("Warning: the API reports model 'mock', so these scores are not meaningful.")
    if a.split in ("holdout", "all") and not a.ids:
        print("Reminder: holdout rows are for the final check. Don't tune chunking, prompts or thresholds on them.")

    print(f"Running {len(rows)} rows against {a.api_url} ({'stream' if a.stream else 'ask'}, top_k={a.top_k})")
    results = []
    for i, row in enumerate(rows, 1):
        payload = {"question": row["question"], "top_k": a.top_k}
        books = [b for b in row["user_books"].split(PIPE) if b]
        if books:
            payload["user_books"] = books
        res: dict = {}
        error = ""
        try:
            res = call_api(a.api_url, payload, a.stream, a.timeout)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            res = {"error": error}
        g = grade(row, res, catalog, a.top_k)
        sources = (res.get("sources") or [])[: a.top_k]
        results.append({
            "id": row["id"], "group": row["group"], "type": row["type"], "split": row["split"],
            "question": row["question"], "user_books": row["user_books"],
            "refused": res.get("refused"), "passed": g["passed"], "hit": g["hit"],
            "precision": None if g["precision"] is None else round(g["precision"], 3),
            "owned_violation": g["owned_violation"], "note": g["note"],
            "sources": PIPE.join(s.get("title", "") for s in sources),
            "scores": PIPE.join(str(s.get("score", "")) for s in sources),
            "latency_ms": res.get("latency_ms"), "wall_ms": res.get("wall_ms"), "ttft_ms": res.get("ttft_ms"),
            "answer": (res.get("answer") or "").replace("\n", " "),
            "reference_answer": row["reference_answer"], "error": error,
        })
        mark = {True: "pass", False: "FAIL", None: "manual"}[g["passed"]]
        print(f"  [{i}/{len(rows)}] {row['id']:<6} {mark:<6} {g['note']}")

    s = summarize(results)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", a.label.lower()).strip("-") or "run"
    meta = {"label": a.label, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "split": a.split if not a.ids else "custom",
            "model": model, "mode": "stream" if a.stream else "ask", "top_k": a.top_k}

    a.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = a.out_dir / f"{stamp}_{slug}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(results)
    md = render_markdown(s, results, meta)
    (a.out_dir / f"{stamp}_{slug}_summary.md").write_text(md, encoding="utf-8")

    history = a.out_dir / "history.csv"
    new = not history.exists()
    with open(history, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "label", "split", "model", "mode", "top_k", "hit_rate", "precision",
                        "false_refusals", "owned_violations", "refusal_acc", "median_latency_ms",
                        "p95_latency_ms", "median_ttft_ms", "errors"])
        r2 = lambda v: "" if v is None else round(v, 3)
        w.writerow([meta["timestamp"], a.label, meta["split"], model, meta["mode"], a.top_k, r2(s["hit_rate"]),
                    r2(s["precision"]), s["false_refusals"], s["owned_violations"], r2(s["refusal_acc"]),
                    r2(s["median_latency_ms"]), r2(s["p95_latency_ms"]), r2(s["median_ttft_ms"]), s["errors"]])

    print("\n" + md)
    print(f"Saved {csv_path} and history.csv")
    return 0

if __name__ == "__main__":
    sys.exit(main())