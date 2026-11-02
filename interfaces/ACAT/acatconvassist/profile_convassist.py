"""Before-picture profiler for the ACAT ConvAssist prediction path."""

import argparse
import cProfile
import io
import json
import logging
import os
import platform
import pstats
import statistics
import sys
import time
from configparser import ConfigParser
from datetime import datetime

from convassist.ConvAssist import ConvAssist
from convassist.utilities.logging_utility import LoggingUtility

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_runs")

MODES = {
    "normal": "wordPredMode.ini",
    "shorthand": "shortHandMode.ini",
    "sentences": "sentenceMode.ini",
    "cannedphrases": "cannedPhrasesMode.ini",
}

# Typical per-keystroke contexts: word-level, partial word, and sentence-level.
CONTEXTS = [
    "i",
    "i want to",
    "i want to go to the",
    "could you please pass me the",
    "how are you doing this morning my friend",
]


def load_config(ini_dir, ini_name, static_path, personal_path, log_location):
    cfg = ConfigParser()
    cfg.read(os.path.join(ini_dir, ini_name))
    for section in cfg.sections():
        for key in list(cfg[section]):
            if key == "static_resources_path":
                cfg[section][key] = static_path
            elif key == "personalized_resources_path":
                cfg[section][key] = personal_path
            elif key == "log_location":
                cfg[section][key] = log_location
            elif key == "log_level":
                cfg[section][key] = "WARNING"
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefs", help="ACATConvAssist_preferences.json; supplies defaults for the path options below")
    p.add_argument("--ini-dir", help="Directory containing the four ini files (prefs: path)")
    p.add_argument("--static", help="static_resources_path value (prefs: pathstatic)")
    p.add_argument("--personal", help="personalized_resources_path value (prefs: pathpersonalized)")
    p.add_argument("--log-dir", help="Log directory (prefs: pathlog)")
    p.add_argument("--iters", type=int, default=20, help="Predict calls per context per mode")
    p.add_argument("--profile", action="store_true", help="Run cProfile over steady state")
    p.add_argument(
        "--output",
        default=None,
        help="JSON report path (default: profile_runs/profile_results_<timestamp>[_<label>].json); pass '' to skip",
    )
    p.add_argument("--label", default="", help="Tag for this run, e.g. 'baseline' or 'sqlite-cache'")
    args = p.parse_args()

    # Explicit command-line values take precedence over the preferences file.
    if args.prefs:
        with open(args.prefs, encoding="utf-8") as f:
            prefs = json.load(f)
        args.ini_dir = args.ini_dir or prefs.get("path")
        args.static = args.static or prefs.get("pathstatic")
        args.personal = args.personal or prefs.get("pathpersonalized")
        args.log_dir = args.log_dir or prefs.get("pathlog")
    args.log_dir = args.log_dir or "/tmp/convassist_profile_logs"

    missing = [n for n, v in (("--ini-dir", args.ini_dir), ("--static", args.static), ("--personal", args.personal)) if not v]
    if missing:
        p.error(f"missing {', '.join(missing)} (provide them or use --prefs)")

    os.makedirs(args.log_dir, exist_ok=True)

    # 1. Startup cost, per mode, mirroring ACATConvAssistInterface.initialize_or_configure_convassists
    instances = {}
    startup = {}
    for mode, ini in MODES.items():
        cfg = load_config(args.ini_dir, ini, args.static, args.personal, args.log_dir)
        ca = ConvAssist(mode, ini_file=ini, log_file=False, log_level=logging.WARNING)
        t0 = time.perf_counter()
        ca.initialize(cfg, args.log_dir, logging.WARNING)
        startup[mode] = time.perf_counter() - t0
        instances[mode] = ca

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "iters_per_context": args.iters,
        "paths": {"ini_dir": args.ini_dir, "static": args.static, "personal": args.personal},
        "startup_seconds": {**startup, "total": sum(startup.values())},
        "latency_ms": [],
    }

    print("\n== Startup (initialize) seconds ==")
    for mode, t in startup.items():
        print(f"  {mode:14s} {t:8.3f}")
    print(f"  {'total':14s} {sum(startup.values()):8.3f}")

    # 2. Steady-state per-keystroke latency, mirroring make_prediction()
    # Warm up once so one-time lazy costs (model first-forward, caches) don't skew results.
    for mode, ca in instances.items():
        for ctx in CONTEXTS:
            ca.context_tracker.context = ctx
            ca.predict()

    print("\n== Steady-state predict() latency (ms) ==")
    print(f"  {'mode':14s} {'context':40s} {'p50':>8s} {'p95':>8s} {'max':>8s}")
    all_samples = []
    for mode, ca in instances.items():
        for ctx in CONTEXTS:
            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                ca.context_tracker.context = ctx
                ca.predict()
                samples.append((time.perf_counter() - t0) * 1000)
            samples.sort()
            p50 = statistics.median(samples)
            p95 = samples[int(0.95 * (len(samples) - 1))]
            print(f"  {mode:14s} {ctx[:40]:40s} {p50:8.2f} {p95:8.2f} {samples[-1]:8.2f}")
            report["latency_ms"].append({
                "mode": mode,
                "context": ctx,
                "p50": p50,
                "p95": p95,
                "min": samples[0],
                "max": samples[-1],
                "mean": statistics.fmean(samples),
                "n": len(samples),
            })
            all_samples.extend(samples)

    all_samples.sort()
    report["overall_latency_ms"] = {
        "p50": statistics.median(all_samples),
        "p95": all_samples[int(0.95 * (len(all_samples) - 1))],
        "max": all_samples[-1],
        "n": len(all_samples),
    }
    print(f"\n  overall p50={report['overall_latency_ms']['p50']:.2f} ms  "
          f"p95={report['overall_latency_ms']['p95']:.2f} ms  "
          f"n={len(all_samples)}")

    # 3. Optional cProfile over steady state to find hot functions
    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
        for ca in instances.values():
            for ctx in CONTEXTS:
                for _ in range(args.iters):
                    ca.context_tracker.context = ctx
                    ca.predict()
        prof.disable()
        buf = io.StringIO()
        stats = pstats.Stats(prof, stream=buf).sort_stats("cumulative")
        stats.print_stats(40)
        print("\n== cProfile (top 40 by cumulative) ==")
        print(buf.getvalue())

        top = sorted(stats.stats.items(), key=lambda kv: kv[1][3], reverse=True)[:40]
        report["cprofile_top_cumulative"] = [
            {
                "function": f"{os.path.basename(fn)}:{line}({name})",
                "file": fn,
                "ncalls": nc,
                "primitive_calls": cc,
                "tottime_s": tt,
                "cumtime_s": ct,
            }
            for (fn, line, name), (cc, nc, tt, ct, _callers) in top
        ]

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"profile_results_{stamp}{'_' + args.label if args.label else ''}.json"
        args.output = os.path.join(RUNS_DIR, name)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults written to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
