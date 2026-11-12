"""digester.py
reads the Open Library dumps in data/dumps/
streams them line by line, picks top rated works for each category, and
writes raw records as json. Cleaning and chunking happen in a later step, so
descriptions and subjects are written exactly as they appear in the dump.

usage examples:
    python digester.py                  # run the whole pipeline
    python digester.py --inspect        # print the first lines of each dump
    python digester.py --min-ratings 3  # override a setting 


Expected input (data/dumps/), either "latest" or dated names such as
ol_dump_works_2026-10-31.txt.gz:
    ol_dump_works_*.txt.gz, ol_dump_authors_*.txt.gz, ol_dump_ratings_*.txt.gz
get them at https://openlibrary.org/developers/dumps
    
output (data/raw/):
    works.jsonl, sample_50.jsonl, manifest.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
DUMP_NAMES = ("works", "authors", "ratings")
DIVIDER = "----------"
PROGRESS_EVERY = 500_000
SAMPLE_SIZE = 50


@dataclass(frozen=True)
class CategoryRule:
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()

DEFAULT_CATEGORIES = {
    "science fiction": CategoryRule(("science fiction",)),
    "fantasy": CategoryRule(("fantasy", "fantasy fiction")),
    "historical fiction": CategoryRule(("historical fiction", "fiction, historical")),
    "romance": CategoryRule(
        ("romance", "romance fiction"),
        ("romance languages", "romance philology", "romance literature"),
    ),
}
# excluding these cause vinland saga and mha took over everything cause they had a lot of editions
DEFAULT_EXCLUDE_SUBJECTS = ("comics", "comic book", "graphic novel", "manga")


@dataclass
class Config:
    categories: dict[str, CategoryRule] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORIES)
    )
    per_category: int = 200
    min_description_chars: int = 200
    min_ratings: int = 15
    exclude_subjects: tuple[str, ...] = DEFAULT_EXCLUDE_SUBJECTS
    bayesian_prior_weight: float = 10
    title_exclude_regex: str = (
        r"\b(box(ed)? set|omnibus|complete collection|collected works)\b"
    )
    dumps_dir: Path = SCRIPT_DIR / "data" / "dumps"
    raw_dir: Path = SCRIPT_DIR / "data" / "raw"

    def dump_path(self, name: str) -> Path:
        latest = self.dumps_dir / f"ol_dump_{name}_latest.txt.gz"
        dated = sorted(self.dumps_dir.glob(f"ol_dump_{name}_????-??-??.txt.gz"))
        return latest if latest.exists() or not dated else dated[-1]


RECORD_FIELDS = (
    "work_id", "title", "authors", "author_bio", "author_bio_author", "description",
    "subjects", "first_publish_year", "source_url", "popularity", "cover_id",
)

# helpers


def extract_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("value"), str):
        return value["value"]
    return None


def description_check_length(text: str) -> int:
    return len(text.split(DIVIDER, 1)[0].strip())


def parse_year(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", value)
    return int(m.group(1)) if m else None

# this is optional, in case we want to have covers in chatbot responses
def pick_cover_id(covers: Any) -> int | None:
    for c in covers if isinstance(covers, list) else []:
        if isinstance(c, int) and not isinstance(c, bool) and c > 0:
            return c
    return None


def _parts(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,/;:]", text.lower()) if p.strip()]


def subject_matches(subject: str, rule: CategoryRule) -> bool:
    # True if some include phrase matches whole parts of the subject.
    # The phrase's last part may have a plain qualifier in front ("American science
    # fiction", "paranormal romance"), but blends like "Love & Romance" or
    # "Science Fiction & Fantasy" do not match.
    low = subject.lower()
    if any(x.lower() in low for x in rule.exclude):
        return False
    parts = _parts(subject)
    for phrase in rule.include:
        want = _parts(phrase)
        n = len(want)
        for i in range(len(parts) - n + 1):
            window = parts[i:i + n]
            last, qualifier = window[-1], window[-1][: -len(want[-1])]
            if window[:-1] == want[:-1] and (
                last == want[-1]
                or (last.endswith(" " + want[-1]) and "&" not in qualifier
                    and " and " not in qualifier)
            ):
                return True
    return False


def match_categories(subjects: list[str], categories: dict[str, CategoryRule]) -> tuple[str, ...]:
    return tuple(
        name for name, rule in categories.items()
        if any(subject_matches(s, rule) for s in subjects)
    )


def has_excluded_subject(subjects: list[str], exclude: tuple[str, ...]) -> bool:
    return any(x.lower() in s.lower() for s in subjects for x in exclude)


def title_stem(title: str) -> str:
    # Title without volume/book numbers, so "Saga, Volume 2" and "Saga #3" both give "saga".
    t = title.lower()
    t = re.sub(r"\(.*?\)", "", t)
    t = re.sub(r"[,:]?\s*\b(vol(ume)?s?|books?|parts?|no|number|issue)\b\.?\s*[\divxl]+\b.*$", "", t)
    t = re.sub(r"\s*#\s*\d+.*$", "", t)
    t = re.sub(r"[\s,:-]+\d+$", "", t)
    return t.strip()

# using this for "popularity"
def bayesian_score(v: int, r: float, c: float, m: float) -> float:
    return (v / (v + m)) * r + (m / (v + m)) * c


def author_keys_from_work(work: dict) -> list[str]:
    keys: list[str] = []
    for entry in work.get("authors") or []:
        author = entry.get("author") if isinstance(entry, dict) else None
        key = author.get("key") if isinstance(author, dict) else None
        if isinstance(key, str) and key not in keys:
            keys.append(key)
    return keys


@dataclass
class Candidate:
    work_id: str
    score: float
    ratings_count: int
    categories: tuple[str, ...]
    title: str = ""
    author_keys: list[str] = field(default_factory=list)
    description: str = ""
    subjects: list[str] = field(default_factory=list)
    first_publish_year: int | None = None
    cover_id: int | None = None
    series_key: str = ""  # first author + title stem; "" means never treated as a series


def assign_greedy(candidates: list[Candidate], category_order, quota: int) -> dict[str, list[Candidate]]:
    selected: dict[str, list[Candidate]] = {name: [] for name in category_order}
    ranked = sorted(candidates, key=lambda c: (-c.score, -c.ratings_count, c.work_id))
    used_series: set[str] = set()  # only the best-ranked book of a series is kept
    for cand in ranked:
        if cand.series_key and cand.series_key in used_series:
            continue
        for name in selected:
            if name in cand.categories and len(selected[name]) < quota:
                selected[name].append(cand)
                used_series.add(cand.series_key)
                break
        if all(len(v) >= quota for v in selected.values()):
            break
    return selected


def sample_balanced(selected: dict[str, list], size: int) -> list:
    base, extra = divmod(size, len(selected))
    out = []
    for i, items in enumerate(selected.values()):
        take = min(base + (i < extra), len(items))
        out += [items[(k * len(items)) // take] for k in range(take)]
    return out


def validate_record(rec: dict) -> list[str]:
    def is_int(v):
        return isinstance(v, int) and not isinstance(v, bool)

    def str_list(v):
        return isinstance(v, list) and all(isinstance(x, str) for x in v)

    checks = {
        "work_id": isinstance(rec.get("work_id"), str),
        "title": isinstance(rec.get("title"), str),
        "authors": str_list(rec.get("authors")),
        "author_bio": rec.get("author_bio") is None or isinstance(rec.get("author_bio"), str),
        "author_bio_author": rec.get("author_bio_author") is None
        or isinstance(rec.get("author_bio_author"), str),
        "description": isinstance(rec.get("description"), str),
        "subjects": str_list(rec.get("subjects")),
        "first_publish_year": rec.get("first_publish_year") is None or is_int(rec.get("first_publish_year")),
        "source_url": isinstance(rec.get("source_url"), str),
        "popularity": is_int(rec.get("popularity")),
        "cover_id": rec.get("cover_id") is None or is_int(rec.get("cover_id")),
    }
    errors = [f"bad {name}" for name, ok in checks.items() if not ok]
    if tuple(rec) != RECORD_FIELDS:
        errors.append(f"wrong fields: {list(rec)}")
    return errors


# dump reading

def read_lines(path: Path) -> Iterator[bytes]:
    t0 = time.monotonic()
    with gzip.open(path, "rb") as f:
        for n, line in enumerate(f, 1):
            if n % PROGRESS_EVERY == 0:
                print(f"  {path.name}: {n:,} lines ({time.monotonic() - t0:.0f}s)")
            yield line


def inspect_dumps(cfg: Config) -> None:
    for name in DUMP_NAMES:
        path = cfg.dump_path(name)
        print(f"\n=== {path.name} ===")
        if not path.exists():
            print("  missing")
            continue
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for i, line in zip(range(3), f):
                cols = line.rstrip("\n").split("\t")
                print(f"  [{i + 1}] {len(cols)} columns: {line.rstrip()[:300]}")


def ratings_pass(cfg: Config, stats: dict) -> dict[bytes, tuple[int, float]]:
    totals: dict[str, list[int]] = {}  # work_key -> [count, sum]
    for line in read_lines(cfg.dump_path("ratings")):
        cols = line.decode("utf-8", errors="replace").rstrip("\n").split("\t")
        try:
            key, rating = cols[0], int(cols[2])
            if not key.startswith("/works/") or not 1 <= rating <= 5:
                raise ValueError
        except (IndexError, ValueError):
            stats["malformed_lines"]["ratings"] += 1
            continue
        t = totals.setdefault(key, [0, 0])
        t[0] += 1
        t[1] += rating

    n_ratings = sum(t[0] for t in totals.values())
    if n_ratings == 0:
        sys.exit("No valid ratings found - run with --inspect to check the format.")
    global_mean = sum(t[1] for t in totals.values()) / n_ratings

    eligible = {}
    for key, (count, total) in totals.items():
        if count < cfg.min_ratings:
            stats["dropped"]["too_few_ratings"] += 1
            continue
        score = bayesian_score(count, total / count, global_mean, cfg.bayesian_prior_weight)
        eligible[key.encode()] = (count, score)

    stats["ratings_processed"] = n_ratings
    stats["global_mean_rating"] = round(global_mean, 6)
    print(f"ratings: {n_ratings:,} ratings, {len(eligible):,} works with >= {cfg.min_ratings}")
    return eligible


def works_pass(cfg: Config, eligible: dict, stats: dict) -> list[Candidate]:
    title_re = re.compile(cfg.title_exclude_regex, re.IGNORECASE)
    dropped = stats["dropped"]
    candidates = []

    for line in read_lines(cfg.dump_path("works")):
        # Columns: type, key, revision, last_modified, JSON
        cols = line.split(b"\t", 4)
        if len(cols) != 5:
            stats["malformed_lines"]["works"] += 1
            continue
        if cols[1] not in eligible:  # cheap check before json.loads
            continue
        try:
            work = json.loads(cols[4])
            if not isinstance(work, dict):
                raise ValueError
        except ValueError:
            stats["malformed_lines"]["works"] += 1
            continue
        count, score = eligible[cols[1]]

        title = work.get("title")
        if not isinstance(title, str) or not title.strip() or title_re.search(title):
            dropped["title_excluded"] += 1  # missing or matches the exclude regex
            continue
        author_keys = author_keys_from_work(work)
        if not author_keys:
            dropped["no_author"] += 1
            continue
        description = extract_text(work.get("description"))
        if description is None or description_check_length(description) < cfg.min_description_chars:
            dropped["short_description"] += 1  # missing or too short
            continue
        subjects = [s for s in work.get("subjects") or [] if isinstance(s, str)]
        if has_excluded_subject(subjects, cfg.exclude_subjects):
            dropped["comics_or_manga"] += 1
            continue
        categories = match_categories(subjects, cfg.categories)
        if not categories:
            dropped["no_subject_match"] += 1
            continue

        candidates.append(Candidate(
            work_id=cols[1].decode().removeprefix("/works/"),
            score=score,
            ratings_count=count,
            categories=categories,
            title=title,
            author_keys=author_keys,
            description=description,
            subjects=subjects,
            first_publish_year=parse_year(work.get("first_publish_date")),
            cover_id=pick_cover_id(work.get("covers")),
            series_key=f"{author_keys[0]}|{stem}" if (stem := title_stem(title)) else "",
        ))

    print(f"works: {len(candidates):,} candidates")
    return candidates


def authors_pass(cfg: Config, needed: set[str], stats: dict) -> dict[str, tuple[str | None, str | None]]:
    wanted = {k.encode() for k in needed}
    authors = {}
    for line in read_lines(cfg.dump_path("authors")):
        cols = line.split(b"\t", 4)
        if len(cols) != 5:
            stats["malformed_lines"]["authors"] += 1
            continue
        if cols[1] not in wanted:
            continue
        try:
            author = json.loads(cols[4])
            if not isinstance(author, dict):
                raise ValueError
        except ValueError:
            stats["malformed_lines"]["authors"] += 1
            continue
        name = author.get("name") if isinstance(author.get("name"), str) else None
        bio = extract_text(author.get("bio"))
        authors[cols[1].decode()] = (name, bio if bio and bio.strip() else None)
        if len(authors) == len(wanted):
            break  # found everyone, no need to read the rest
    print(f"authors: found {len(authors):,} of {len(needed):,}")
    return authors


def build_record(cand: Candidate, authors: dict) -> dict:
    names, bio, bio_author = [], None, None
    for key in cand.author_keys:
        name, author_bio = authors.get(key, (None, None))
        if name:
            names.append(name)
            if bio is None and author_bio:  # the bio must belong to a named author
                bio, bio_author = author_bio, name
    return {
        "work_id": cand.work_id,
        "title": cand.title,
        "authors": names,
        "author_bio": bio,
        "author_bio_author": bio_author,  # name of the author the bio belongs to
        "description": cand.description,
        "subjects": cand.subjects,
        "first_publish_year": cand.first_publish_year,
        "source_url": f"https://openlibrary.org/works/{cand.work_id}",
        "popularity": cand.ratings_count,
        "cover_id": cand.cover_id,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# running
def run(cfg: Config) -> int:
    missing = [cfg.dump_path(n).name for n in DUMP_NAMES if not cfg.dump_path(n).exists()]
    if missing:
        sys.exit(f"Missing in {cfg.dumps_dir}: {', '.join(missing)}")

    dumps = {}
    for name in DUMP_NAMES:
        st = cfg.dump_path(name).stat()
        modified = datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        dumps[name] = {"file": cfg.dump_path(name).name, "size_bytes": st.st_size, "modified": modified}
        print(f"using {dumps[name]['file']} ({st.st_size:,} bytes, modified {modified})")

    stats = {
        "ratings_processed": 0,
        "global_mean_rating": None,
        "malformed_lines": {"ratings": 0, "works": 0, "authors": 0},
        "dropped": {"too_few_ratings": 0, "title_excluded": 0, "no_author": 0,
                    "short_description": 0, "no_subject_match": 0, "comics_or_manga": 0},
    }

    eligible = ratings_pass(cfg, stats)
    candidates = works_pass(cfg, eligible, stats)
    selected = assign_greedy(candidates, cfg.categories, cfg.per_category)

    candidates_per_category = {
        name: sum(name in c.categories for c in candidates) for name in cfg.categories
    }
    selected_per_category = {name: len(lst) for name, lst in selected.items()}
    shortfalls = {name: cfg.per_category - n for name, n in selected_per_category.items()
                  if n < cfg.per_category}

    needed = {k for lst in selected.values() for c in lst for k in c.author_keys}
    authors = authors_pass(cfg, needed, stats)
    records_by_cat = {name: [build_record(c, authors) for c in lst] for name, lst in selected.items()}
    records = [r for lst in records_by_cat.values() for r in lst]

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cfg.raw_dir / "works.jsonl", records)
    write_jsonl(cfg.raw_dir / f"sample_{SAMPLE_SIZE}.jsonl", sample_balanced(records_by_cat, SAMPLE_SIZE))

    manifest = {
        "dumps": dumps,
        "config": {
            "categories": {n: {"include": list(r.include), "exclude": list(r.exclude)}
                           for n, r in cfg.categories.items()},
            "per_category": cfg.per_category,
            "min_description_chars": cfg.min_description_chars,
            "min_ratings": cfg.min_ratings,
            "exclude_subjects": list(cfg.exclude_subjects),
            "bayesian_prior_weight": cfg.bayesian_prior_weight,
            "title_exclude_regex": cfg.title_exclude_regex,
        },
        **stats,
        "candidates_per_category": candidates_per_category,
        "selected_per_category": selected_per_category,
        "shortfalls": shortfalls,
        "category": {r["work_id"]: name for name, lst in records_by_cat.items() for r in lst},
    }
    with open(cfg.raw_dir / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Validation
    errors = [f"{r.get('work_id')}: {e}" for r in records for e in validate_record(r)]
    ids = [r["work_id"] for r in records]
    if len(ids) != len(set(ids)):
        errors.append("duplicate work_ids found")

    print(f"\n{'Category':<20}{'Candidates':>12}{'Selected':>10}{'Short':>8}")
    for name in cfg.categories:
        print(f"{name:<20}{candidates_per_category[name]:>12}"
              f"{selected_per_category[name]:>10}{shortfalls.get(name, 0):>8}")
    print(f"{'TOTAL':<20}{len(candidates):>12}{len(records):>10}{sum(shortfalls.values()):>8}")

    if shortfalls:
        print("\n" + "!" * 70)
        print("WARNING: not every category was filled: "
              + ", ".join(f"{n} short by {k}" for n, k in shortfalls.items()))
        print(f"Try lowering --min-ratings (currently {cfg.min_ratings}).")
        print("!" * 70)

    print_readme_summary(cfg, manifest)

    if errors:
        print(f"\nVALIDATION FAILED ({len(errors)} problems):")
        for e in errors[:20]:
            print("  " + e)
        return 1
    print(f"\nValidation passed: {len(records)} records, no duplicate work_ids.")
    return 0


def print_readme_summary(cfg: Config, m: dict) -> None:
    d = m["dropped"]
    lines = [
        "\n----- README summary (copy below) -----",
        "### Data acquisition (digester.py)",
        "",
        "Source: Open Library data dumps "
        + ", ".join(f"`{i['file']}` (modified {i['modified']})" for i in m["dumps"].values()) + ".",
        "",
        f"Top {cfg.per_category} works per category ({', '.join(cfg.categories)}) by Bayesian "
        f"average rating (m = {cfg.bayesian_prior_weight}, global mean = {m['global_mean_rating']:.3f}). "
        "Each work is assigned to the first matching category with room, so there are no duplicates.",
        "",
        "Filters: at least " f"{cfg.min_ratings} ratings; a title that isn't a box set/omnibus; "
        "no comics/manga subjects; only the best-rated book per author and title stem (one entry per series); "
        f"at least one author; a description of at least {cfg.min_description_chars} characters "
        f"(before any `{DIVIDER}` divider); at least one subject matching a category.",
        "",
        "| Step | Count |",
        "|---|---|",
        f"| Ratings processed | {m['ratings_processed']:,} |",
        f"| Dropped: too few ratings | {d['too_few_ratings']:,} |",
        f"| Dropped: title missing/excluded | {d['title_excluded']:,} |",
        f"| Dropped: no author | {d['no_author']:,} |",
        f"| Dropped: description missing/short | {d['short_description']:,} |",
        f"| Dropped: no subject match | {d['no_subject_match']:,} |",
        f"| Dropped: comics/manga subject | {d['comics_or_manga']:,} |",
    ]
    lines += [f"| Selected: {n} | {k} |" for n, k in m["selected_per_category"].items()]
    lines += [f"| **Selected total** | **{sum(m['selected_per_category'].values())}** |",
              "----- end README summary -----"]
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    d = Config()
    p = argparse.ArgumentParser(description="Select top-rated Open Library works per category.")
    p.add_argument("--per-category", type=int, default=d.per_category)
    p.add_argument("--min-description-chars", type=int, default=d.min_description_chars)
    p.add_argument("--min-ratings", type=int, default=d.min_ratings)
    p.add_argument("--prior-weight", type=float, default=d.bayesian_prior_weight)
    p.add_argument("--dumps-dir", type=Path, default=d.dumps_dir)
    p.add_argument("--out-dir", type=Path, default=d.raw_dir)
    p.add_argument("--inspect", action="store_true", help="print the first lines of each dump and exit")
    a = p.parse_args(argv)

    cfg = Config(
        per_category=a.per_category,
        min_description_chars=a.min_description_chars,
        min_ratings=a.min_ratings,
        bayesian_prior_weight=a.prior_weight,
        dumps_dir=a.dumps_dir,
        raw_dir=a.out_dir,
    )
    if a.inspect:
        inspect_dumps(cfg)
        return 0
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
