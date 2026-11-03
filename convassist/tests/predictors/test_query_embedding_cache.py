"""Semantic-query characterization and D1 contracts; no models or resource files."""

from unittest.mock import Mock

import hnswlib
import numpy as np
import pytest

from convassist.predictor import canned_phrases_predictor as module
from convassist.predictor.utilities.prediction import Prediction
from convassist.predictor.utilities.suggestion import Suggestion


@pytest.fixture
def make_predictor(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "SentenceTransformer", Mock())
    monkeypatch.setattr(module, "cannedData", Mock())
    monkeypatch.setattr(module.joblib, "dump", Mock())
    module.cannedData.return_value.length = 0
    module.cannedData.return_value.all_phrases_as_dict.return_value = {}

    def make():
        predictor = object.__new__(module.CannedPhrasesPredictor)
        predictor.device = "cpu"
        predictor.logger = Mock()
        predictor._predictor_name = "semantic-test"
        predictor._static_resources_path = str(tmp_path)
        predictor._personalized_resources_path = str(tmp_path)
        predictor._personalized_cannedphrases = "unused.txt"
        predictor._sentences_db = "unused.db"
        predictor._embedding_cache_path = "unused.pkl"
        predictor._index_path = "unused.index"
        predictor._sbertmodel = "mock-model"
        predictor._learn = True
        predictor.configure()
        predictor.embedder = Mock()
        predictor.embedder.encode.side_effect = lambda text: np.array([3, 4, 2, 1], dtype=np.float32)
        predictor.corpus_phrases = [" first phrase ", "second phrase"]
        predictor.corpus_embeddings = np.ones((2, 4), dtype=np.float32)
        predictor.index = Mock()
        predictor.index.knn_query.return_value = (np.array([[0, 1]]), np.array([[0.3, 0.1]]))
        return predictor

    return make


def signature(prediction):
    return [(s.word, s.probability, s.predictor_name) for s in prediction]


def test_characterization_current_index_and_direct_dedup_each_call(make_predictor):
    predictor = make_predictor()
    direct = Prediction()
    direct.add_suggestion(Suggestion("second phrase", 2.5, predictor.predictor_name))
    result = predictor._find_semantic_matches("raw query text", direct)
    assert signature(result) == [
        ("second phrase", 2.5, "semantic-test"), ("first phrase", 0.7, "semantic-test")]
    # Neither the neighbor result nor direct-match exclusions may be cached.
    predictor.index = Mock()
    predictor.index.knn_query.return_value = (np.array([[1]]), np.array([[0.05]]))
    assert signature(predictor._find_semantic_matches("raw query text", Prediction())) == [
        ("second phrase", 0.95, "semantic-test")]
    predictor.index.knn_query.assert_called_once()
    assert predictor.index.knn_query.call_args.kwargs == {"k": 10}


def test_characterization_encode_exception_is_retried(make_predictor):
    predictor = make_predictor()
    vector = np.array([3, 4, 2, 1], dtype=np.float32)
    predictor.embedder.encode.side_effect = [RuntimeError("encode failed"), vector]
    with pytest.raises(RuntimeError, match="encode failed"):
        predictor._find_semantic_matches("retry this query", Prediction())
    predictor.index.knn_query.assert_not_called()
    assert len(predictor._find_semantic_matches("retry this query", Prediction())) == 2
    assert predictor.embedder.encode.call_count == 2
    np.testing.assert_array_equal(predictor.index.knn_query.call_args.args[0], vector)
    assert all(call.args == ("retry this query",) and not call.kwargs
               for call in predictor.embedder.encode.call_args_list)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_characterization_real_cosine_knn_reads_without_mutating(dtype):
    # Exercise the installed native API, not a mock's assumed read behavior.
    corpus = np.arange(1, 49, dtype=np.float32).reshape(12, 4)
    index = hnswlib.Index(space="cosine", dim=4)
    index.init_index(max_elements=12, random_seed=42)
    index.add_items(corpus)
    vector = np.array([3, 4, 2, 1], dtype=dtype)
    before = vector.copy()
    first = index.knn_query(vector, k=10)
    second = index.knn_query(vector, k=10)
    np.testing.assert_array_equal(vector, before)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_hits_reuse_exact_array_but_search_each_time(make_predictor):
    predictor = make_predictor()
    results = [signature(predictor._find_semantic_matches("same raw query", Prediction()))
               for _ in range(3)]
    assert results == [results[0]] * 3
    predictor.embedder.encode.assert_called_once_with("same raw query")
    assert predictor.index.knn_query.call_count == 3
    vector = predictor._query_embedding_cache["same raw query"]
    assert all(call.args[0] is vector for call in predictor.index.knn_query.call_args_list)
    np.testing.assert_array_equal(vector, np.array([3, 4, 2, 1], dtype=np.float32))


def test_exact_whitespace_and_case_keys_and_instance_isolation(make_predictor):
    first, second = make_predictor(), make_predictor()
    texts = ["raw query text", " raw query text", "raw query text ", "RAW query text",
             "raw  query text", "raw\tquery text"]
    for text in texts * 2:
        first._find_semantic_matches(text, Prediction())
    second._find_semantic_matches(texts[0], Prediction())
    assert [call.args[0] for call in first.embedder.encode.call_args_list] == texts
    second.embedder.encode.assert_called_once_with(texts[0])
    assert first._query_embedding_cache is not second._query_embedding_cache


def test_lru_bound_hit_recency_and_least_recent_eviction(make_predictor):
    predictor = make_predictor()
    for i in range(128):
        predictor._find_semantic_matches(f"query number {i}", Prediction())
    assert len(predictor._query_embedding_cache) == 128
    predictor._find_semantic_matches("query number 0", Prediction())
    assert next(reversed(predictor._query_embedding_cache)) == "query number 0"
    predictor._find_semantic_matches("query number 128", Prediction())
    assert len(predictor._query_embedding_cache) == 128
    assert "query number 1" not in predictor._query_embedding_cache
    assert "query number 0" in predictor._query_embedding_cache
    predictor._find_semantic_matches("query number 1", Prediction())
    assert predictor.embedder.encode.call_count == 130
    assert len(predictor._query_embedding_cache) == 128


def test_configure_resets_before_model_setup_even_if_setup_fails(make_predictor, monkeypatch):
    predictor = make_predictor()
    predictor._find_semantic_matches("old model query", Prediction())

    def setup(*args, **kwargs):
        assert not predictor._query_embedding_cache
        raise RuntimeError("model setup failed")

    monkeypatch.setattr(module, "SentenceTransformer", setup)
    with pytest.raises(RuntimeError, match="model setup failed"):
        predictor.configure()
    assert not predictor._query_embedding_cache


def test_successful_configure_discards_old_embeddings(make_predictor):
    predictor = make_predictor()
    predictor._find_semantic_matches("old model query", Prediction())
    old = predictor._query_embedding_cache
    predictor.configure()
    assert not predictor._query_embedding_cache
    assert predictor._query_embedding_cache is not old
    predictor.embedder = Mock()
    predictor.embedder.encode.return_value = np.zeros(4, dtype=np.float32)
    predictor.index.knn_query.return_value = (np.empty((1, 0), dtype=int), np.empty((1, 0)))
    predictor._find_semantic_matches("old model query", Prediction())
    predictor.embedder.encode.assert_called_once_with("old model query")


def test_failed_encode_leaves_no_entry_or_eviction(make_predictor):
    predictor = make_predictor()
    for i in range(128):
        predictor._find_semantic_matches(f"existing query {i}", Prediction())
    before = list(predictor._query_embedding_cache)
    predictor.embedder.encode.side_effect = RuntimeError("encode failed")
    with pytest.raises(RuntimeError, match="encode failed"):
        predictor._find_semantic_matches("failing new query", Prediction())
    assert list(predictor._query_embedding_cache) == before


def test_failed_search_retries_index_without_reencoding(make_predictor):
    predictor = make_predictor()
    result = predictor.index.knn_query.return_value
    predictor.index.knn_query.side_effect = [RuntimeError("index failed"), result]
    with pytest.raises(RuntimeError, match="index failed"):
        predictor._find_semantic_matches("same query text", Prediction())
    assert len(predictor._find_semantic_matches("same query text", Prediction())) == 2
    predictor.embedder.encode.assert_called_once_with("same query text")
    assert predictor.index.knn_query.call_count == 2


def test_learn_keeps_query_embedding_but_new_results_are_visible(make_predictor):
    predictor = make_predictor()
    predictor._find_semantic_matches("same query text", Prediction())
    vector = predictor._query_embedding_cache["same query text"]
    predictor.learn("new learned phrase")
    assert predictor._query_embedding_cache["same query text"] is vector
    predictor.index.add_items.assert_called_once()
    predictor.cannedData.learn.assert_called_once_with("new learned phrase")
    predictor.index.knn_query.return_value = (np.array([[2]]), np.array([[0.0]]))
    assert signature(predictor._find_semantic_matches("same query text", Prediction())) == [
        ("new learned phrase", 1.0, "semantic-test")]
    assert [call.args for call in predictor.embedder.encode.call_args_list] == [
        ("same query text",), ("new learned phrase",)]
    predictor.logger.error.assert_not_called()


def test_cached_native_queries_preserve_arrays_scores_and_order(make_predictor):
    predictor = make_predictor()
    corpus = np.arange(1, 49, dtype=np.float32).reshape(12, 4)
    predictor.corpus_phrases = [f"phrase {i}" for i in range(12)]
    index = hnswlib.Index(space="cosine", dim=4)
    index.init_index(max_elements=12, random_seed=42)
    index.add_items(corpus)
    predictor.index = index
    expected = signature(predictor._find_semantic_matches("native query text", Prediction()))
    vector = predictor._query_embedding_cache["native query text"]
    before = vector.copy()
    for _ in range(3):
        assert signature(predictor._find_semantic_matches("native query text", Prediction())) == expected
        np.testing.assert_array_equal(vector, before)
    predictor.embedder.encode.assert_called_once_with("native query text")
