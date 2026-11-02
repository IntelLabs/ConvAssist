"""Before-picture profiler for the ACAT ConvAssist prediction path."""

import argparse
import cProfile
import io
import logging
import os
import pstats
import statistics
import time
from configparser import ConfigParser

from convassist.ConvAssist import ConvAssist
from convassist.utilities.logging_utility import LoggingUtility

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
    p.add_argument("--ini-dir", required=True, help="Directory containing the four ini files")
    p.add_argument("--static", required=True, help="static_resources_path value")
    p.add_argument("--personal", required=True, help="personalized_resources_path value")
    p.add_argument("--log-dir", default="/tmp/convassist_profile_logs")
    p.add_argument("--iters", type=int, default=20, help="Predict calls per context per mode")
    p.add_argument("--profile", action="store_true", help="Run cProfile over steady state")
    args = p.parse_args()

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
            all_samples.extend(samples)

    all_samples.sort()
    print(f"\n  overall p50={statistics.median(all_samples):.2f} ms  "
          f"p95={all_samples[int(0.95*(len(all_samples)-1))]:.2f} ms  "
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
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
        print("\n== cProfile (top 40 by cumulative) ==")
        print(buf.getvalue())


if __name__ == "__main__":
    main()
