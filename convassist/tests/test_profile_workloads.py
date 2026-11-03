"""Benchmark protocol tests using stub applications, never production profiling."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "interfaces" / "ACAT" / "acatconvassist"
LEGACY_CONTEXTS = ["i", "i want to", "i want to go to the",
                   "could you please pass me the", "how are you doing this morning my friend"]


@pytest.fixture
def scripts(monkeypatch):
    class Application:
        instances = []

        def __init__(self, mode, **kwargs):
            self.mode = mode
            self.context_tracker = SimpleNamespace(context="")
            self.calls = []
            self.instances.append(self)

        def initialize(self, *args):
            pass

        def predict(self):
            self.calls.append(self.context_tracker.context)

    def load(name):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    # Import the actual scripts without importing ConvAssist or any model packages.
    with monkeypatch.context() as imports:
        app = ModuleType("convassist.ConvAssist")
        app.ConvAssist = Application
        logs = ModuleType("convassist.utilities.logging_utility")
        logs.LoggingUtility = object
        imports.setitem(sys.modules, app.__name__, app)
        imports.setitem(sys.modules, logs.__name__, logs)
        profiler = load("profile_convassist")
    return profiler, load("compare_profiles"), Application


def flatten(groups):
    return [text for _, sessions in groups for session in sessions for text in session]


def test_repeated_preserves_legacy_contexts_counts_order_and_label(scripts):
    profiler, _, _ = scripts
    assert profiler.CONTEXTS == LEGACY_CONTEXTS
    groups = profiler.workload_groups("repeated", 7)
    assert [label for label, _ in groups] == LEGACY_CONTEXTS
    assert flatten(groups) == [ctx for ctx in LEGACY_CONTEXTS for _ in range(7)]
    metadata = profiler.workload_metadata("repeated", groups, profiler.CONTEXTS)
    assert not metadata["warmup_disjoint"]
    assert "cache-hit oriented" in metadata["cache_warming"]


def test_unseen_unique_across_entire_timing_warmup_and_profile(scripts):
    profiler, _, _ = scripts
    groups = profiler.workload_groups("unseen", 400)
    texts = flatten(groups)
    assert len(texts) == len(set(texts)) == 2000
    assert all(len(text.split()) > 2 for text in texts)
    assert not set(texts).intersection(profiler.SEPARATE_WARMUP)
    assert not set(texts).intersection(flatten(profiler.workload_groups("unseen", 400, "profile")))
    assert groups == profiler.workload_groups("unseen", 400)
    assert [label for label, _ in groups] == [f"unseen fixture {i}" for i in range(1, 6)]


def test_typing_backspaces_revisits_semantic_queries_and_isolated_sessions(scripts):
    profiler, _, _ = scripts
    groups = profiler.workload_groups("typing", 3)
    previous = set(profiler.SEPARATE_WARMUP)
    total = 0
    for _, sessions in groups:
        assert len(sessions) == 3
        for session in sessions:
            assert not previous.intersection(session)
            assert len(session) > len(set(session))  # deliberate revisits inside session
            assert any(len(text.split()) > 2 for text in session)
            assert any(b == a[:-1] for a, b in zip(session, session[1:]))
            assert all(abs(len(b) - len(a)) == 1 for a, b in zip(session, session[1:]))
            previous.update(session)
            total += len(session)
    details = profiler.workload_metadata("typing", groups, profiler.SEPARATE_WARMUP)
    assert details["timed_per_mode"]["sample_count"] == total
    assert details["timed_per_mode"]["session_count"] == 15
    assert details["warmup_disjoint"]
    assert not previous.intersection(flatten(profiler.workload_groups("typing", 3, "profile")))


@pytest.mark.parametrize("workload", ["repeated", "unseen", "typing"])
def test_cli_reports_counts_schema_and_actual_stream(scripts, tmp_path, monkeypatch, workload):
    profiler, _, application = scripts
    prefs = tmp_path / "prefs.json"
    prefs.write_text(json.dumps({"path": str(tmp_path), "pathstatic": str(tmp_path),
                                "pathpersonalized": str(tmp_path), "pathlog": str(tmp_path)}))
    output = tmp_path / "result.json"
    argv = ["profile_convassist", "--prefs", str(prefs), "--iters", "2",
            "--output", str(output), "--label", "d1-check", "--profile"]
    # Also exercise legacy default rather than spelling repeated explicitly.
    if workload != "repeated":
        argv.extend(["--workload", workload])
    monkeypatch.setattr(sys, "argv", argv)
    profiler.main()
    report = json.loads(output.read_text())
    assert report["label"] == "d1-check"
    assert report["iters_per_context"] == 2
    groups = profiler.workload_groups(workload, 2)
    timed = flatten(groups)
    warmup = LEGACY_CONTEXTS if workload == "repeated" else profiler.SEPARATE_WARMUP
    assert all(ca.calls == warmup + timed + flatten(profiler.workload_groups(workload, 2, "profile"))
               for ca in application.instances)
    assert report["overall_latency_ms"]["n"] == len(timed) * 4
    assert len(report["latency_ms"]) == 20
    assert [row["context"] for row in report["latency_ms"][:5]] == [label for label, _ in groups]
    assert sum(row["n"] for row in report["latency_ms"]) == len(timed) * 4
    for row in report["latency_ms"]:
        assert {"mode", "context", "p50", "p95", "min", "max", "mean", "n"} <= row.keys()
        assert row["context_details"]["sample_count"] == row["n"]
    assert report["workload"]["timed_per_mode"]["unique_context_count"] == len(set(timed))
    assert "cprofile_top_cumulative" in report


@pytest.mark.parametrize("value", ["0", "-1"])
def test_iters_must_be_positive_before_app_initialization(scripts, monkeypatch, value):
    profiler, _, application = scripts
    monkeypatch.setattr(sys, "argv", ["profiler", "--iters", value])
    with pytest.raises(SystemExit) as exc:
        profiler.main()
    assert exc.value.code == 2
    assert application.instances == []


def report_for(profiler, workload="repeated", iters=2):
    groups = profiler.workload_groups(workload, iters)
    return {
        "timestamp": "fixture", "label": "stable", "iters_per_context": iters,
        "startup_seconds": {"normal": 1},
        "latency_ms": [{"mode": "normal", "context": label, "p50": 1, "p95": 1}
                       for label, _ in groups],
        "overall_latency_ms": {"p50": 1, "p95": 1, "max": 1},
        "workload": profiler.workload_metadata(
            workload, groups,
            profiler.CONTEXTS if workload == "repeated" else profiler.SEPARATE_WARMUP),
    }


def compare(comparator, base, candidate, tmp_path, monkeypatch):
    paths = [tmp_path / "base.json", tmp_path / "candidate.json"]
    for path, report in zip(paths, [base, candidate]):
        path.write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["comparator", *map(str, paths)])
    return comparator.main()


@pytest.mark.parametrize("legacy_side", ["base", "candidate", "both", "neither"])
def test_comparison_accepts_legacy_repeated(scripts, tmp_path, monkeypatch, legacy_side):
    profiler, comparator, _ = scripts
    base, candidate = report_for(profiler), report_for(profiler)
    if legacy_side in ("base", "both"):
        del base["workload"]
    if legacy_side in ("candidate", "both"):
        del candidate["workload"]
    assert compare(comparator, base, candidate, tmp_path, monkeypatch) == 0


@pytest.mark.parametrize("workload", ["unseen", "typing"])
def test_comparison_accepts_matching_new_workloads(scripts, tmp_path, monkeypatch, workload):
    profiler, comparator, _ = scripts
    base = report_for(profiler, workload)
    assert compare(comparator, base, copy.deepcopy(base), tmp_path, monkeypatch) == 0


@pytest.mark.parametrize("left,right", [("repeated", "unseen"), ("unseen", "typing"),
                                        ("typing", "repeated")])
def test_comparison_blocks_different_workloads(scripts, tmp_path, monkeypatch, capsys, left, right):
    profiler, comparator, _ = scripts
    with pytest.raises(SystemExit) as exc:
        compare(comparator, report_for(profiler, left), report_for(profiler, right), tmp_path, monkeypatch)
    assert exc.value.code == 2
    assert "incompatible workload" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["cache_warming", "protocol_version", "timed_per_mode", "warmup_contexts", "cprofile_policy"])
def test_comparison_blocks_changed_protocol_or_stream(scripts, tmp_path, monkeypatch, field):
    profiler, comparator, _ = scripts
    base = report_for(profiler, "typing")
    candidate = copy.deepcopy(base)
    candidate["workload"][field] = "changed"
    with pytest.raises(SystemExit) as exc:
        compare(comparator, base, candidate, tmp_path, monkeypatch)
    assert exc.value.code == 2


@pytest.mark.parametrize("workload", ["repeated", "unseen", "typing"])
def test_optional_cprofile_does_not_block_latency_comparison(scripts, tmp_path, monkeypatch, workload):
    profiler, comparator, _ = scripts
    base = report_for(profiler, workload)
    candidate = copy.deepcopy(base)
    candidate["workload"]["cprofile_per_mode"] = profiler.context_details(
        [session for _, sessions in profiler.workload_groups(workload, 2, "profile")
         for session in sessions])
    assert compare(comparator, base, candidate, tmp_path, monkeypatch) == 0


@pytest.mark.parametrize("field", ["warmup_contexts", "scope", "row_context_kind", "cache_warming"])
def test_repeated_changed_metadata_is_not_treated_as_legacy(scripts, tmp_path, monkeypatch, field):
    profiler, comparator, _ = scripts
    base = report_for(profiler)
    candidate = copy.deepcopy(base)
    del base["workload"]
    candidate["workload"][field] = "changed"
    with pytest.raises(SystemExit) as exc:
        compare(comparator, base, candidate, tmp_path, monkeypatch)
    assert exc.value.code == 2


@pytest.mark.parametrize("workload", ["unseen", "typing"])
def test_legacy_repeated_cannot_compare_with_new_workload(scripts, tmp_path, monkeypatch, workload):
    profiler, comparator, _ = scripts
    base = report_for(profiler)
    del base["workload"]
    with pytest.raises(SystemExit) as exc:
        compare(comparator, base, report_for(profiler, workload), tmp_path, monkeypatch)
    assert exc.value.code == 2


def test_repeated_different_iters_keeps_legacy_warning(scripts, tmp_path, monkeypatch, capsys):
    profiler, comparator, _ = scripts
    assert compare(comparator, report_for(profiler, iters=2), report_for(profiler, iters=3),
                   tmp_path, monkeypatch) == 0
    assert "WARNING: --iters differs" in capsys.readouterr().out


@pytest.fixture
def run_report(tmp_path):
    def write(stamp, suffix, **report):
        path = tmp_path / f"profile_results_{stamp}_{suffix}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return str(path)
    return write


def test_resolve_exact_label_beats_newer_filename_and_label_substrings(scripts, tmp_path, run_report):
    _, comparator, _ = scripts
    baseline = run_report("20260101", "original", label="baseline")
    run_report("20260201", "milestone-c2-baseline1", label="milestone-c2-baseline1")
    run_report("20260301", "milestone-c2-baseline2", label="milestone-c2-baseline2")
    assert comparator.resolve("baseline", str(tmp_path)) == baseline


def test_resolve_existing_path_precedes_runs_without_reading_json(scripts, tmp_path, run_report, monkeypatch):
    _, comparator, _ = scripts
    run_report("20260101", "other", label="baseline.json")
    explicit = tmp_path / "baseline.json"
    explicit.write_text("not JSON", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(comparator, "load", lambda _: pytest.fail("must not read JSON"))
    assert comparator.resolve("baseline.json", str(tmp_path)) == "baseline.json"


def test_resolve_exact_filename_precedes_label_without_reading_json(scripts, tmp_path, run_report, monkeypatch):
    _, comparator, _ = scripts
    exact = run_report("20260101", "baseline", label="other")
    run_report("20260201", "other", label=Path(exact).name)
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setattr(comparator, "load", lambda _: pytest.fail("must not read JSON"))
    assert comparator.resolve(Path(exact).name, str(tmp_path)) == exact


def test_resolve_duplicate_exact_labels_uses_newest_filename_not_report_timestamp(scripts, tmp_path, run_report):
    _, comparator, _ = scripts
    newest = run_report("20260301", "original", label="baseline", timestamp="1900")
    run_report("20260101", "original", label="baseline", timestamp="2100")
    run_report("20260401", "baseline2", label="baseline2")
    assert comparator.resolve("baseline", str(tmp_path)) == newest


@pytest.mark.parametrize("ref", ["unique-file", "unique-label"])
def test_resolve_unique_partial_filename_or_label(scripts, tmp_path, run_report, ref):
    _, comparator, _ = scripts
    expected = run_report("20260101", "unique-file-suffix", label="unique-label-suffix")
    run_report("20260201", "unrelated", label="other")
    assert comparator.resolve(ref, str(tmp_path)) == expected


def test_resolve_partial_matching_both_filename_and_label_counts_run_once(scripts, tmp_path, run_report):
    _, comparator, _ = scripts
    expected = run_report("20260101", "milestone-d", label="milestone-d")
    assert comparator.resolve("milestone", str(tmp_path)) == expected


def test_resolve_ambiguous_partial_lists_filename_and_label_matches(scripts, tmp_path, run_report):
    _, comparator, _ = scripts
    first = run_report("20260101", "baseline1", label="other")
    second = run_report("20260201", "unrelated", label="baseline2")
    with pytest.raises(SystemExit) as exc:
        comparator.resolve("baseline", str(tmp_path))
    message = str(exc.value)
    assert "Ambiguous" in message
    assert Path(first).name in message and Path(second).name in message
    assert "exact label" in message and "file" in message


def test_resolve_no_matches_keeps_friendly_error(scripts, tmp_path, run_report):
    _, comparator, _ = scripts
    run_report("20260101", "other", label="other")
    with pytest.raises(SystemExit, match="No run found for 'missing'"):
        comparator.resolve("missing", str(tmp_path))


@pytest.mark.parametrize("metadata", [{}, {"label": ""}])
def test_resolve_missing_or_empty_label_is_not_baseline(scripts, tmp_path, run_report, metadata):
    _, comparator, _ = scripts
    legacy = run_report("20260201", "baseline2", **metadata)
    assert comparator.resolve("baseline", str(tmp_path)) == legacy
    baseline = run_report("20260101", "original", label="baseline")
    assert comparator.resolve("baseline", str(tmp_path)) == baseline
