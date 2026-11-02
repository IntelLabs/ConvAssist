"""Compare two profile_convassist.py runs and flag performance regressions/improvements.

Usage:
    python compare_profiles.py                    # two most recent runs in profile_runs/
    python compare_profiles.py baseline           # run matching 'baseline' vs most recent
    python compare_profiles.py a.json b.json      # explicit files
    python compare_profiles.py --list
"""

import argparse
import glob
import json
import os
import re
import sys

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_runs")

# Startup is a single sample per run, so it needs a wide margin and an absolute floor.
STARTUP_MIN_DELTA_S = 0.05


def list_runs(runs_dir):
    # Timestamp is in the filename, so a lexical sort is chronological.
    return sorted(glob.glob(os.path.join(runs_dir, "profile_results_*.json")))


def resolve(ref, runs_dir):
    """Accept a file path, or a substring (e.g. a label) matching a run; newest match wins."""
    if os.path.isfile(ref):
        return ref
    matches = [r for r in list_runs(runs_dir) if ref in os.path.basename(r)]
    if not matches:
        sys.exit(f"No run found for '{ref}' (looked in {runs_dir})")
    return matches[-1]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def status(base, cand, threshold, min_delta):
    delta = cand - base
    pct = (delta / base * 100) if base else 0.0
    if abs(delta) < min_delta or abs(pct) < threshold:
        return pct, ""
    return pct, "REGRESSION" if delta > 0 else "improved"


def fmt_row(label, base, cand, threshold, min_delta, counts, width=44):
    # threshold=None marks an informational row that is never flagged.
    pct, flag = status(base, cand, threshold, min_delta) if threshold is not None else ((cand - base) / base * 100 if base else 0.0, "")
    if flag:
        counts[flag] += 1
    print(f"  {label:{width}s} {base:10.2f} {cand:10.2f} {cand - base:+9.2f} {pct:+7.1f}%  {flag}")


def func_key(entry):
    # Line numbers shift as code is edited, so key on the last two path parts + function name.
    m = re.match(r"^.*?:\d+\((.*)\)$", entry["function"])
    name = m.group(1) if m else entry["function"]
    parts = entry["file"].replace("\\", "/").split("/")
    return f"{'/'.join(parts[-2:])}:{name}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", nargs="?", help="Baseline run: file path or label/timestamp substring")
    p.add_argument("candidate", nargs="?", help="Candidate run (default: most recent)")
    p.add_argument("--dir", default=RUNS_DIR, help="Runs directory (default: profile_runs/ next to this script)")
    p.add_argument("--threshold", type=float, default=10.0, help="Percent change to flag for per-row p50 (default: 10)")
    p.add_argument("--overall-threshold", type=float, default=5.0, help="Percent change to flag for overall p50 (default: 5)")
    p.add_argument("--startup-threshold", type=float, default=20.0, help="Percent change to flag for startup time (default: 20)")
    p.add_argument("--min-delta-ms", type=float, default=1.0, help="Ignore per-row p50 changes below this (default: 1.0)")
    p.add_argument("--top", type=int, default=10, help="cProfile functions to show per direction (default: 10)")
    p.add_argument("--fail-on-regression", action="store_true", help="Exit 1 if any regression is flagged")
    p.add_argument("--list", action="store_true", help="List available runs and exit")
    args = p.parse_args()

    runs = list_runs(args.dir)
    if args.list:
        for r in runs:
            d = load(r)
            print(f"{os.path.basename(r)}  label={d.get('label', '')!r}  p50={d['overall_latency_ms']['p50']:.2f}ms")
        return 0

    if args.baseline and args.candidate:
        base_path, cand_path = resolve(args.baseline, args.dir), resolve(args.candidate, args.dir)
    elif args.baseline:
        if not runs:
            sys.exit(f"No runs in {args.dir}")
        base_path, cand_path = resolve(args.baseline, args.dir), runs[-1]
    else:
        if len(runs) < 2:
            sys.exit(f"Need at least two runs in {args.dir} (found {len(runs)})")
        base_path, cand_path = runs[-2], runs[-1]

    base, cand = load(base_path), load(cand_path)
    counts = {"REGRESSION": 0, "improved": 0}

    print(f"baseline : {os.path.basename(base_path)}  ({base['timestamp']}, label={base.get('label', '')!r})")
    print(f"candidate: {os.path.basename(cand_path)}  ({cand['timestamp']}, label={cand.get('label', '')!r})")
    print(
        f"flags: per-row p50 >= {args.threshold}% and {args.min_delta_ms} ms, "
        f"overall p50 >= {args.overall_threshold}%, startup >= {args.startup_threshold}%; "
        "p95 and max are informational (too noisy to gate on)"
    )
    if base.get("iters_per_context") != cand.get("iters_per_context"):
        print("WARNING: --iters differs between runs; percentiles are less comparable.")

    header = f"  {'':44s} {'baseline':>10s} {'candidate':>10s} {'delta':>9s} {'%':>8s}"

    print("\n== Startup (seconds) ==")
    print(header)
    for mode, b in base["startup_seconds"].items():
        if mode in cand["startup_seconds"]:
            fmt_row(mode, b, cand["startup_seconds"][mode], args.startup_threshold, STARTUP_MIN_DELTA_S, counts)

    print("\n== Steady-state predict() latency (ms) ==")
    print(header)
    cand_lat = {(r["mode"], r["context"]): r for r in cand["latency_ms"]}
    for r in base["latency_ms"]:
        c = cand_lat.get((r["mode"], r["context"]))
        if not c:
            continue
        name = f"{r['mode']}/{r['context'][:30]}"
        fmt_row(f"{name} p50", r["p50"], c["p50"], args.threshold, args.min_delta_ms, counts)
        fmt_row(f"{name} p95 (info)", r["p95"], c["p95"], None, 0, counts)

    print("\n== Overall (ms) ==")
    print(header)
    for k in ("p50", "p95", "max"):
        # Overall p50 aggregates hundreds of samples, so only the percentage matters.
        thr, floor = (args.overall_threshold, 0) if k == "p50" else (None, 0)
        fmt_row(k if k == "p50" else f"{k} (info)", base["overall_latency_ms"][k], cand["overall_latency_ms"][k], thr, floor, counts)

    if base.get("cprofile_top_cumulative") and cand.get("cprofile_top_cumulative"):
        b = {func_key(e): e for e in base["cprofile_top_cumulative"]}
        c = {func_key(e): e for e in cand["cprofile_top_cumulative"]}
        # Functions missing from one top-N list are treated as 0 s, which can overstate their change.
        rows = [
            (k, b.get(k, {}).get("cumtime_s", 0.0), c.get(k, {}).get("cumtime_s", 0.0))
            for k in set(b) | set(c)
        ]
        rows.sort(key=lambda r: r[2] - r[1])
        better = [r for r in rows if r[2] < r[1]][: args.top]
        worse = [r for r in reversed(rows) if r[2] > r[1]][: args.top]
        print("\n== cProfile cumulative time (s): biggest improvements ==")
        for k, bt, ct in better:
            print(f"  {k[:70]:70s} {bt:8.2f} -> {ct:8.2f}  ({ct - bt:+.2f})")
        print("\n== cProfile cumulative time (s): biggest regressions ==")
        for k, bt, ct in worse:
            print(f"  {k[:70]:70s} {bt:8.2f} -> {ct:8.2f}  ({ct - bt:+.2f})")

    print(f"\nSummary: {counts['REGRESSION']} regression(s), {counts['improved']} improvement(s)")
    return 1 if args.fail_on_regression and counts["REGRESSION"] else 0


if __name__ == "__main__":
    sys.exit(main())
