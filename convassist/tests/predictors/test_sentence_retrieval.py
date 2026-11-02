# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

"""Retrieval characterization with temporary data and no model initialization."""

import random
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy
import pytest

from convassist.predictor import sentence_completion_predictor as module


@pytest.fixture
def make_predictor(tmp_path, monkeypatch):
    monkeypatch.setattr(module.SentenceCompletionPredictor, "load_model", Mock())
    monkeypatch.setattr(module, "SentenceTransformer", Mock())
    monkeypatch.setattr(module.hnswlib, "Index", Mock())
    monkeypatch.setattr(module, "SVOUtil", Mock())

    def make(lines=(), personal=(), semantic=("Semantic only",)):
        (tmp_path / "corpus.txt").write_text("".join(lines), encoding="utf-8")
        (tmp_path / "blacklist.txt").write_text("", encoding="utf-8")
        (tmp_path / "stopwords.txt").write_text("", encoding="utf-8")
        (tmp_path / "semantic.index").touch()
        module.joblib.dump(
            {"sentences": list(semantic), "embeddings": numpy.zeros((len(semantic), 384))},
            tmp_path / "semantic.pkl",
        )
        with sqlite3.connect(tmp_path / "personal.db") as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS sentences (sentence TEXT UNIQUE, count INTEGER)")
            conn.execute("DELETE FROM sentences")
            conn.executemany("INSERT INTO sentences VALUES (?, ?)", personal)

        predictor = object.__new__(module.SentenceCompletionPredictor)
        predictor.device = "cpu"
        predictor.logger = Mock()
        predictor._predictor_name = "retrieval-test"
        predictor._static_resources_path = str(tmp_path)
        predictor._personalized_resources_path = str(tmp_path)
        predictor._retrieve_database = "corpus.txt"
        predictor._sent_database = "personal.db"
        predictor._blacklist_file = "blacklist.txt"
        predictor._stopwords = "stopwords.txt"
        predictor._personalized_allowed_toxicwords_file = "toxic.txt"
        predictor._embedding_cache_path = "semantic.pkl"
        predictor._index_path = "semantic.index"
        predictor._sentence_transformer_model = "unused-local-mock"
        predictor._learn = True
        predictor._retrieveaac = True
        predictor._model_loaded = False
        predictor._test_generalsentenceprediction = False
        predictor.context_tracker = SimpleNamespace(context="")
        predictor.configure()
        return predictor

    return make


def suggestions(predictor, context):
    return [(s.word, s.probability) for s in predictor._retrieve_fromDataset(context)]


def legacy_result(lines, personal, context):
    """Independent oracle preserving the old raw-prefix/count/set/suffix algorithm."""
    retrieved = [re.split("[.\n?!]", line)[0] for line in lines
                 if line.lower().startswith(context.lower())]
    probs = {s: retrieved.count(s) / len(lines) for s in set(retrieved)}
    counts = {}
    for sentence, count in personal:
        if sentence.lower().startswith(context.lower()):
            display = re.split("[.\n?!]", sentence)[0]
            counts[display] = counts.get(display, 0) + count
    total = sum(count for _, count in personal)
    probs.update({s: count / total for s, count in counts.items()})
    result = []
    seen = set()
    for sentence, probability in sorted(probs.items(), key=lambda item: item[1], reverse=True):
        suffix = sentence[len(context):]
        if suffix not in seen:
            result.append((suffix, probability))
            seen.add(suffix)
        if len(result) == 6:
            break
    return result


@pytest.mark.parametrize("context", ["", "i", "I w", "i want. m", "i want? y", "absent", "İ", "\U0010ffff"])
def test_raw_prefix_case_duplicates_blanks_and_punctuation(make_predictor, context):
    lines = ["I want. More\n", "I want. More\n", "I want? Yes\n", "I want! No\n",
             "i want tea\n", "Other\n", "\n", "İstanbul\n", "\U0010ffff end\n"]
    predictor = make_predictor(lines)
    assert suggestions(predictor, context) == legacy_result(lines, (), context)


@pytest.mark.parametrize("lines", [[], ["Other\n", "\n"]])
def test_empty_or_no_matching_corpus(make_predictor, lines):
    assert suggestions(make_predictor(lines), "missing") == []


def test_probabilities_use_all_raw_lines_and_personal_weighted_total(make_predictor):
    lines = ["I want. One\n", "I want? Two\n", "I want tea\n", "Other\n", "\n"]
    personal = [("I want. Personal", 2), ("I want? Again", 1), ("Other", 7)]
    predictor = make_predictor(lines, personal)
    result = dict(suggestions(predictor, "I"))
    assert result == {" want": pytest.approx(0.3), " want tea": pytest.approx(0.2)}


def test_suffix_deduplication_and_existing_six_cap(make_predictor):
    lines = ["I same\n", "i same\n"] + [f"I option {i}\n" for i in range(10)]
    predictor = make_predictor(lines)
    result = suggestions(predictor, "i")
    assert result == legacy_result(lines, (), "i")
    assert len(result) == 6
    assert len({word for word, _ in result}) == 6


@pytest.mark.parametrize("seed", range(3))
def test_index_matches_legacy_in_process_order_for_varied_prefixes(make_predictor, seed):
    rng = random.Random(seed)
    starts = ["I", "i", "İ", "ß", "Σ", "\U0010ffff", "", "Other"]
    endings = [" tea\n", ". Again\n", "? Yes\n", "! No\n", " same\n", "\n"]
    lines = [rng.choice(starts) + rng.choice(endings) for _ in range(120)]
    personal = [("I personal. Again", 3), ("I personal? Yes", 2), ("Other", 11)]
    predictor = make_predictor(lines, personal)
    # Match the existing charset detector, which may reinterpret short UTF-8 data.
    lines = module.smart_readlines(predictor.retrieve_database)
    contexts = {"", "absent"}
    for line in lines:
        contexts.update(line[:length] for length in range(len(line) + 1))
    for context in contexts:
        assert suggestions(predictor, context) == legacy_result(lines, personal, context)


def test_personal_external_changes_are_live(make_predictor):
    predictor = make_predictor(["I static\n"], [("I personal", 1), ("Other", 3)])
    assert dict(suggestions(predictor, "I"))[" personal"] == 0.25
    with sqlite3.connect(predictor.sent_database) as conn:
        conn.execute("UPDATE sentences SET count = 3 WHERE sentence = 'I personal'")
        conn.execute("INSERT INTO sentences VALUES ('I new', 4)")
    result = dict(suggestions(predictor, "I"))
    assert result[" personal"] == 0.3
    assert result[" new"] == 0.4


def test_learn_new_and_existing_remains_live(make_predictor):
    predictor = make_predictor(["I static\n"], [("Other", 3)])
    predictor.embedder.encode.return_value = numpy.ones(384)
    predictor.learn("I learned")
    assert dict(suggestions(predictor, "I"))[" learned"] == 0.25
    predictor.learn("I learned")
    assert dict(suggestions(predictor, "I"))[" learned"] == 0.4
    assert predictor.corpus_sentences.count("I learned") == 1
    assert suggestions(predictor, "Semantic") == []


@pytest.mark.parametrize("retrieve,loaded,expected_calls", [(True, True, 0), (False, True, 1), (False, False, 0)])
def test_predict_aac_and_gpt_fallback_without_models(make_predictor, retrieve, loaded, expected_calls):
    predictor = make_predictor(["I static\n"])
    predictor.context_tracker.context = "I"
    predictor._retrieveaac = retrieve
    predictor._model_loaded = loaded
    predictor._generate = Mock(side_effect=lambda context, count, predictions: predictions)
    sentences, words = predictor.predict(5)
    assert [(s.word, s.probability) for s in sentences] == [(" static", 1.0)]
    assert words == []
    assert predictor._generate.call_count == expected_calls
    if expected_calls:
        predictor._generate.assert_called_once_with("I", 4, sentences)


def test_static_cache_refreshes_only_on_configure(make_predictor):
    predictor = make_predictor(["I original\n"], semantic=("I semantic only",))
    assert predictor.corpus_sentences == ["I semantic only"]
    assert suggestions(predictor, "I") == [(" original", 1.0)]

    Path(predictor.retrieve_database).write_text("I replacement\nOther\n", encoding="utf-8")
    assert suggestions(predictor, "I") == [(" original", 1.0)]
    predictor.configure()
    assert suggestions(predictor, "I") == [(" replacement", 0.5)]
    assert predictor.corpus_sentences == ["I semantic only"]


def test_no_hot_path_corpus_reads_or_semantic_cache_reuse(make_predictor, monkeypatch):
    predictor = make_predictor(["I original\n"])
    monkeypatch.setattr(module, "smart_readlines", Mock(side_effect=AssertionError("Hot-path file read")))
    predictor.corpus_sentences.append("I semantic only")
    assert suggestions(predictor, "I") == [(" original", 1.0)]


def test_learning_does_not_rebuild_static_cache(make_predictor):
    predictor = make_predictor(["I original\n"])
    cache = predictor._retrieval_lines
    predictor.embedder.encode.return_value = numpy.ones(384)
    predictor.learn("I learned")
    predictor.learn("I learned")
    assert predictor._retrieval_lines is cache
    assert dict(suggestions(predictor, "I")) == {" original": 1.0, " learned": 1.0}


@pytest.mark.parametrize("failure", [None, "connect", "fetch_all", "constructor", "suggestion"])
def test_personal_connection_closed_on_success_and_failure(make_predictor, monkeypatch, failure):
    predictor = make_predictor(["I original\n"])
    connector = Mock()
    connector.fetch_all.return_value = []
    factory = Mock(return_value=connector)
    monkeypatch.setattr(module, "SQLiteDatabaseConnector", factory)
    if failure == "constructor":
        factory.side_effect = RuntimeError("Cannot construct connector")
    elif failure == "suggestion":
        monkeypatch.setattr(module, "Suggestion", Mock(side_effect=RuntimeError("Cannot build suggestion")))
    elif failure:
        getattr(connector, failure).side_effect = RuntimeError("Cannot read personal database")
    predictor._retrieve_fromDataset("I")
    if failure != "constructor":
        connector.close.assert_called_once_with()
    if failure:
        predictor.logger.error.assert_called_once()
