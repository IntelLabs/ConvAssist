"""Before-picture profiler for the ACAT ConvAssist prediction path."""

import argparse
import cProfile
import hashlib
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

# Synthetic fixtures, not a representative conversational corpus. Identifiers ensure
# exact-text uniqueness, but can change tokenization and semantic match quality.
FIXTURES = [
    "could you bring some water",
    "please help me open the window",
    "i would like to visit the garden",
    "can we listen to music together",
    "thank you for staying with me",
]
SEPARATE_WARMUP = [f"unrelated warmup query number {i}" for i in range(5)]


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def workload_groups(workload, iters, phase="timed"):
    """Return (row label, sessions) pairs; each session is an ordered text stream.

    repeated: iters calls per original fixed context (legacy order).
    unseen: iters unique queries per fixture group, disjoint from warmup/profile.
    typing: iters independent phrases per group, every character then five
    backspaces and five retyped characters. A unique FIRST token isolates each
    session's exact keys without clearing caches or timing cache-reset work.
    """
    if workload == "repeated":
        return [(ctx, [[ctx] * iters]) for ctx in CONTEXTS]
    groups = []
    for group, phrase in enumerate(FIXTURES):
        if workload == "unseen":
            sessions = [[f"{phase} unseen {group} {i} {phrase}" for i in range(iters)]]
        else:
            sessions = []
            for i in range(iters):
                # Start with a complete identifier so even short prefixes cannot
                # reuse keys from a previous session. No blank/identifier typing.
                prefix = f"{phase}-session-{group}-{i} "
                text = prefix + phrase
                forward = [text[:n] for n in range(len(prefix) + 1, len(text) + 1)]
                back = [text[:-n] for n in range(1, 6)]
                retype = [text[:n] for n in range(len(text) - 4, len(text) + 1)]
                sessions.append(forward + back + retype)
        groups.append((f"{workload} fixture {group + 1}", sessions))
    return groups


def context_details(sessions):
    texts = [text for session in sessions for text in session]
    return {
        "sample_count": len(texts),
        "unique_context_count": len(set(texts)),
        "session_count": len(sessions),
        "first_context": texts[0],
        "last_context": texts[-1],
        "sequence_sha256": hashlib.sha256(json.dumps(texts).encode("utf-8")).hexdigest(),
    }


def workload_metadata(workload, groups, warmup):
    details = context_details([session for _, sessions in groups for session in sessions])
    return {
        "name": workload,
        "protocol_version": 1,
        "iters_unit": {
            "repeated": "calls per fixed context per mode",
            "unseen": "unique calls per fixture group per mode",
            "typing": "sessions per fixture group per mode",
        }[workload],
        "scope": "per mode; each mode receives the same deterministic stream",
        "cache_warming": {
            "repeated": "timed contexts pre-warmed; cache-hit oriented",
            "unseen": "warmup disjoint; no timed exact-text reuse",
            "typing": "warmup disjoint; unique first-token session identifiers; cold exact keys at session start, revisits within session",
        }[workload],
        "warmup_contexts": warmup,
        "warmup_disjoint": not set(warmup).intersection(
            text for _, sessions in groups for session in sessions for text in session),
        "row_context_kind": "fixed text" if workload == "repeated" else "varying-context fixture group",
        "timed_per_mode": details,
        "limitation": (
            "Five pre-warmed fixed contexts; not unseen-text latency."
            if workload == "repeated" else
            "Synthetic phrases and sequence identifiers affect tokenization/semantics; not a natural conversation distribution. "
            "Typing begins after an inserted identifier, includes backspaces/revisits; unrelated model/runtime state remains warm, no cache resets."
            if workload == "typing" else
            "Synthetic multi-token phrases plus sequence identifiers ensure unique text, not semantic diversity or a natural conversation distribution. Model/runtime state remains warm."
        ),
        "cprofile_policy": "separate pass; new phase identifiers for unseen/typing, repeated remains warm",
    }


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
    p.add_argument("--iters", type=positive_int, default=20,
                   help="Positive count per mode: repeated calls per fixed context; unseen queries per fixture group; typing sessions per fixture group (many calls/session). Report gives actual sample counts.")
    p.add_argument("--workload", choices=("repeated", "unseen", "typing"), default="repeated",
                   help="repeated: legacy pre-warmed cache-hit oriented; unseen: unique synthetic multi-token queries; typing: isolated unique-phrase character streams with backspaces/revisits")
    p.add_argument("--profile", action="store_true", help="Run cProfile over steady state")
    p.add_argument(
        "--output",
        default=None,
        help="JSON report path (default: profile_runs/profile_results_<timestamp>[_<label>].json); pass '' to skip",
    )
    p.add_argument("--label", default="", help="Tag for this run, e.g. 'baseline' or 'sqlite-cache'")
    args = p.parse_args()
    groups = workload_groups(args.workload, args.iters)
    warmup = CONTEXTS if args.workload == "repeated" else SEPARATE_WARMUP

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
        "workload": workload_metadata(args.workload, groups, warmup),
    }

    print("\n== Startup (initialize) seconds ==")
    for mode, t in startup.items():
        print(f"  {mode:14s} {t:8.3f}")
    print(f"  {'total':14s} {sum(startup.values()):8.3f}")

    # 2. Steady-state per-keystroke latency, mirroring make_prediction()
    # Warm up once so one-time lazy costs (model first-forward, caches) don't skew results.
    for mode, ca in instances.items():
        for ctx in warmup:
            ca.context_tracker.context = ctx
            ca.predict()

    print(f"\nWorkload: {args.workload} — {report['workload']['cache_warming']}")
    print(f"  {report['workload']['limitation']}")
    print(f"  per mode: {report['workload']['timed_per_mode']['sample_count']} calls, "
          f"{report['workload']['timed_per_mode']['unique_context_count']} unique texts; "
          f"rows: {report['workload']['row_context_kind']}")
    print("\n== Steady-state predict() latency (ms) ==")
    print(f"  {'mode':14s} {'context':40s} {'p50':>8s} {'p95':>8s} {'max':>8s}")
    all_samples = []
    for mode, ca in instances.items():
        for label, sessions in groups:
            samples = []
            for session in sessions:
                for ctx in session:
                    t0 = time.perf_counter()
                    ca.context_tracker.context = ctx
                    ca.predict()
                    samples.append((time.perf_counter() - t0) * 1000)
            samples.sort()
            p50 = statistics.median(samples)
            p95 = samples[int(0.95 * (len(samples) - 1))]
            print(f"  {mode:14s} {label[:40]:40s} {p50:8.2f} {p95:8.2f} {samples[-1]:8.2f}")
            report["latency_ms"].append({
                "mode": mode,
                "context": label,
                "p50": p50,
                "p95": p95,
                "min": samples[0],
                "max": samples[-1],
                "mean": statistics.fmean(samples),
                "n": len(samples),
                "context_details": context_details(sessions),
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
        profile_groups = workload_groups(args.workload, args.iters, phase="profile")
        report["workload"]["cprofile_per_mode"] = context_details(
            [session for _, sessions in profile_groups for session in sessions])
        prof = cProfile.Profile()
        prof.enable()
        for ca in instances.values():
            for _, sessions in profile_groups:
                for session in sessions:
                    for ctx in session:
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
