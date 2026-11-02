# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prediction-local cache contracts using only disposable SQLite databases."""

import json
import sqlite3
import string
from collections import Counter
from unittest.mock import Mock

import pytest

from convassist.context_tracker import ContextTracker
from convassist.predictor.smoothed_ngram_predictor import smoothed_ngram_predictor as module
from convassist.utilities.ngram.ngramutil import NGramUtil


@pytest.fixture
def predictor(tmp_path):
    # Bypass Predictor.__init__: no models, resource setup, or file loggers.
    result = object.__new__(module.SmoothedNgramPredictor)
    result._personalized_resources_path = str(tmp_path)
    result._database = "ngrams.db"
    result._predictor_name = "cache-test"
    result._deltas = "0.01 0.1 0.89"
    result._learn = True
    result._startwords = "starts.json"
    result.logger = Mock()
    result.context_tracker = ContextTracker()
    (tmp_path / "starts.json").write_text(json.dumps({"tea": 0.6, "toast": 0.4}))
    with sqlite3.connect(result.database) as db:
        for size in range(1, 4):
            columns = [f"word_{i}" for i in range(size - 1, 0, -1)] + ["word"]
            db.execute(
                f"CREATE TABLE _{size}_gram "
                f"({', '.join(c + ' TEXT' for c in columns)}, count INTEGER, "
                f"UNIQUE({', '.join(columns)}))"
            )
        db.executemany("INSERT INTO _1_gram VALUES (?, ?)", [
            ("i", 10), ("like", 8), ("tea", 6), ("toast", 4), ("tart", 0), ("!!!", 3),
        ])
        db.executemany("INSERT INTO _2_gram VALUES (?, ?, ?)", [
            ("i", "like", 8), ("like", "tea", 5), ("like", "toast", 2),
            ("like", "ghost", 1), ("like", "tart", 1), ("like", "!!!", 1),
        ])
        db.executemany("INSERT INTO _3_gram VALUES (?, ?, ?, ?)", [
            ("i", "like", "tea", 4), ("i", "like", "toast", 1),
            ("i", "like", "ghost", 1), ("i", "like", "tart", 1),
            ("i", "like", "!!!", 1),
        ])
    return result


def signature(predictions):
    return [[(s.word, s.probability, s.predictor_name) for s in group]
            for group in predictions]


def uncached_reference(predictor, limit):
    """Original accumulated-candidate/rescoring algorithm, independent of predict()."""
    actual, tokens = predictor.context_tracker.get_tokens(predictor.cardinality)
    if not actual:
        return signature(([], predictor.get_frequent_start_words(limit)))
    candidates = []
    words = []
    for size in range(actual, 0, -1):
        if len(candidates) >= limit:
            break
        with NGramUtil(predictor.database, size) as util:
            prefix = tokens[-size:]
            candidates.extend(row[-2] for row in util.fetch_like(prefix, limit - len(candidates)))
            total = util.unigram_counts_sum()
            candidate_tokens = prefix.copy()
            for candidate in candidates:
                candidate_tokens[-1] = candidate
                probability = 0.0
                for k in range(len(candidate_tokens)):
                    numerator = util.count(candidate_tokens, 0, k + 1)
                    denominator = total
                    if numerator > 0:
                        denominator = util.count(candidate_tokens, -1, k)
                    frequency = float(numerator) / denominator if denominator > 0 else 0
                    probability += float(predictor.deltas[k]) * frequency
                if probability > 0 and not all(c in string.punctuation for c in candidate):
                    words.append((candidate, probability, predictor.predictor_name))
                    words.sort(key=lambda s: s[1], reverse=True)
    return [[], words]


@pytest.fixture
def spy(monkeypatch):
    events = []
    queries = []
    sums = []
    instances = []

    class TrackedNGramUtil(NGramUtil):
        def __enter__(self):
            result = super().__enter__()
            instances.append(self)
            events.append(("enter", self._cardinality))
            return result

        def __exit__(self, exc_type, exc_val, exc_tb):
            super().__exit__(exc_type, exc_val, exc_tb)
            events.append(("exit", self._cardinality, exc_type))
            assert self.connection.conn is None

        def count(self, tokens, offset, ngram_size):
            key = tuple(tokens[len(tokens) - ngram_size + offset:len(tokens) + offset])
            value = super().count(tokens, offset, ngram_size)
            queries.append((key, self._cardinality, offset, ngram_size, value))
            return value

        def unigram_counts_sum(self):
            sums.append(self._cardinality)
            return super().unigram_counts_sum()

    monkeypatch.setattr(module, "NGramUtil", TrackedNGramUtil)
    return TrackedNGramUtil, events, queries, sums, instances


@pytest.mark.parametrize("context", [
    "", " ", "i ", "i like ", "i like t", "like t", "t", "zz", "x y z ", "i like tea",
])
@pytest.mark.parametrize("limit", [1, 4, 20])
def test_characterization_matches_uncached_order_scores_and_duplicates(predictor, context, limit):
    predictor.context_tracker.context = context
    expected = uncached_reference(predictor, limit)
    assert signature(predictor.predict(limit, None)) == expected


def test_accumulation_rescores_duplicates_and_filters_punctuation(predictor):
    predictor.context_tracker.context = "i like "
    words = predictor.predict(20, None)[1]
    assert Counter(s.word for s in words)["tea"] == 6
    assert "!!!" not in [s.word for s in words]
    # Missing unigram counts do not prevent higher-order scores.
    assert any(s.word == "ghost" and s.probability > 0 for s in words)
    assert any(s.word == "tart" and s.probability > 0 for s in words)


def test_actual_query_keys_reused_across_candidates_offsets_and_cardinalities(predictor, spy):
    _, events, queries, sums, instances = spy
    predictor.context_tracker.context = "i like "
    assert signature(predictor.predict(20, None)) == uncached_reference(predictor, 20)
    positive_size_queries = [q for q in queries if q[3] > 0]
    counts = Counter(q[0] for q in positive_size_queries)
    assert counts[("like",)] == 1  # shared denominator, also a later unigram numerator
    assert counts[("tea",)] == 1  # repeated through all three candidate lengths
    assert counts[("like", "tea")] == 1
    assert all(count == 1 for count in counts.values())
    assert not any(q[3] == 0 for q in queries)  # shares the cached SUM, not count(..., 0)
    assert sums == [3]
    assert events == [item for size in (3, 2, 1)
                      for item in (("enter", size), ("exit", size, None))]
    assert len(instances) == 3


def test_zero_counts_are_memoized_without_skipping_higher_order_scores(predictor, spy):
    _, _, queries, _, _ = spy
    predictor.context_tracker.context = "i like "
    words = predictor.predict(20, None)[1]
    for token in ("tart", "ghost"):
        matching = [q for q in queries if q[0] == (token,) and q[3] > 0]
        assert len(matching) == 1
        assert matching[0][4] == 0
        assert any(s.word == token for s in words)


def test_zero_sum_is_cached_even_with_no_candidates(predictor, spy):
    _, _, _, sums, _ = spy
    with sqlite3.connect(predictor.database) as db:
        for size in (1, 2, 3):
            db.execute(f"DELETE FROM _{size}_gram")
    predictor.context_tracker.context = "i like "
    assert signature(predictor.predict(20, None)) == [[], []]
    assert sums == [3]


@pytest.mark.parametrize("zero_all_unigrams", [False, True])
def test_positive_numerators_with_zero_denominators_match_reference(predictor, spy, zero_all_unigrams):
    _, _, queries, sums, _ = spy
    with sqlite3.connect(predictor.database) as db:
        if zero_all_unigrams:
            db.execute("UPDATE _1_gram SET count = 0")
        else:
            db.execute("UPDATE _1_gram SET count = 0 WHERE word = 'like'")
    predictor.context_tracker.context = "i like "
    expected = uncached_reference(predictor, 20)
    assert signature(predictor.predict(20, None)) == expected
    assert expected[1]  # higher-order terms still contribute
    assert any(q[2:] == (-1, 1, 0) for q in queries)
    assert sums == [3]


def test_candidate_limit_preserves_early_connection_exit(predictor, spy):
    _, events, _, sums, _ = spy
    predictor.context_tracker.context = "i like "
    assert signature(predictor.predict(1, None)) == uncached_reference(predictor, 1)
    assert events == [("enter", 3), ("exit", 3, None)]
    assert sums == [3]


@pytest.mark.parametrize("context,limit", [("", 20), ("i like ", 0)])
def test_no_sum_or_connection_when_not_queried(predictor, spy, context, limit):
    _, events, queries, sums, _ = spy
    predictor.context_tracker.context = context
    assert signature(predictor.predict(limit, None)) == uncached_reference(predictor, limit)
    assert events == queries == sums == []


def test_request_and_instance_caches_are_independent(predictor, spy):
    _, _, queries, sums, _ = spy
    predictor.context_tracker.context = "i like "
    first = signature(predictor.predict(20, None))
    first_queries = list(queries)
    queries.clear()
    assert signature(predictor.predict(20, None)) == first
    assert queries == first_queries
    # A second object sharing the same database must still make fresh queries.
    other = object.__new__(module.SmoothedNgramPredictor)
    other.__dict__.update(predictor.__dict__)
    queries.clear()
    assert signature(other.predict(20, None)) == first
    assert queries == first_queries
    assert sums == [3, 3, 3]


@pytest.mark.parametrize("update", ["learn", "external"])
def test_updates_between_requests_are_visible(predictor, update):
    predictor.context_tracker.context = "i like "
    before = signature(predictor.predict(20, None))
    if update == "learn":
        predictor.learn("i like tea")
    else:
        with sqlite3.connect(predictor.database) as db:
            db.execute("UPDATE _1_gram SET count = count + 50 WHERE word = 'tea'")
            db.execute("UPDATE _3_gram SET count = count + 7 WHERE word = 'tea'")
    after = signature(predictor.predict(20, None))
    assert after != before
    assert after == uncached_reference(predictor, 20)
    predictor.logger.error.assert_not_called()


def test_fetch_failure_continues_and_preserves_context_lifecycle(predictor, spy, monkeypatch):
    tracked, events, _, sums, _ = spy
    original = tracked.fetch_like

    def fail_trigram(self, *args):
        if self._cardinality == 3:
            raise RuntimeError("fetch failure")
        return original(self, *args)

    monkeypatch.setattr(tracked, "fetch_like", fail_trigram)
    predictor.context_tracker.context = "i like "
    words = predictor.predict(20, None)[1]
    assert words
    assert events == [item for size in (3, 2, 1)
                      for item in (("enter", size), ("exit", size, None))]
    assert sums == [2]
    assert "Error fetching ngrams" in predictor.logger.error.call_args.args[0]


def test_all_fetch_failures_do_not_query_sum(predictor, spy, monkeypatch):
    tracked, events, queries, sums, _ = spy
    monkeypatch.setattr(tracked, "fetch_like", Mock(side_effect=RuntimeError("fetch failure")))
    predictor.context_tracker.context = "i like "
    assert signature(predictor.predict(20, None)) == [[], []]
    assert len(events) == 6
    assert queries == sums == []
    assert predictor.logger.error.call_count == 3


@pytest.mark.parametrize("operation", ["count", "unigram_counts_sum"])
def test_query_failure_aborts_closes_and_is_retried_next_request(predictor, spy, monkeypatch, operation):
    tracked, events, _, _, _ = spy
    original = getattr(tracked, operation)
    attempts = []

    def fail_once(self, *args):
        attempts.append(args)
        if len(attempts) == 1:
            raise RuntimeError("query failure")
        return original(self, *args)

    monkeypatch.setattr(tracked, operation, fail_once)
    predictor.context_tracker.context = "i like "
    assert signature(predictor.predict(20, None)) == [[], []]
    assert events == [("enter", 3), ("exit", 3, RuntimeError)]
    assert len(attempts) == 1  # count/SUM failures abort the whole prediction, not continue
    assert "Exception in cache-test predict function: query failure" == predictor.logger.error.call_args.args[0]
    assert signature(predictor.predict(20, None)) == uncached_reference(predictor, 20)
    assert len(attempts) > 1


def test_count_failure_preserves_already_accumulated_suggestions(predictor, spy, monkeypatch):
    tracked, events, _, _, _ = spy
    original = tracked.count

    def fail_unigram(self, *args):
        if self._cardinality == 1:
            raise RuntimeError("later query failure")
        return original(self, *args)

    monkeypatch.setattr(tracked, "count", fail_unigram)
    predictor.context_tracker.context = "i like "
    words = predictor.predict(20, None)[1]
    assert words
    assert events == [("enter", 3), ("exit", 3, None),
                      ("enter", 2), ("exit", 2, None),
                      ("enter", 1), ("exit", 1, RuntimeError)]
    assert "later query failure" in predictor.logger.error.call_args.args[0]
