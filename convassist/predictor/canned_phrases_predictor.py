# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import collections
import logging
import os
from pathlib import Path

import hnswlib
import joblib
import numpy
import torch
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer

from convassist.predictor.predictor import Predictor
from convassist.predictor.utilities.canned_data import cannedData
from convassist.predictor.utilities.prediction import Prediction
from convassist.predictor.utilities.suggestion import Suggestion


class CannedPhrasesPredictor(Predictor):
    """
    CannedPhrasesPredictor is a class that searches a database of canned phrases
    to find matching next words and sentences based on a given context.
    """

    def __init__(
        self,
        config,
        context_tracker,
        predictor_name,
        logger: logging.Logger | None = None,
    ):
        # Only check for GPU one time must be done before predictor is loaded
        if torch.cuda.is_available():
            self.device = "cuda"
            self.n_gpu = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.n_gpu = torch.mps.device_count()
        else:
            self.device = "cpu"
            self.n_gpu = 0

        super().__init__(config, context_tracker, predictor_name, logger)

    def configure(self):
        self.corpus_phrases = []
        self.corpus_embeddings = []

        self._model_loaded = False
        self.stemmer = PorterStemmer()

        if os.path.exists(os.path.join(self._static_resources_path, self.sbertmodel)):
            localfiles = True
        else:
            localfiles = False

        self.embedder = SentenceTransformer(
            self.sbertmodel,
            device=self.device,
            local_files_only=localfiles,
            tokenizer_kwargs={"clean_up_tokenization_spaces": True},
        )

        self.cannedData = cannedData(self.sentences_db_path, self.personalized_cannedphrases)

        if self.cannedData.length > 0:
            if not os.path.isfile(self.embedding_cache_path):
                self.logger.debug(f"{self.embedding_cache_path} does not exist, creating")

                canned_data = self.cannedData.all_phrases_as_list()
                corpus_embeddings = self.embedder.encode(
                    canned_data, show_progress_bar=True, convert_to_numpy=True
                )
                joblib.dump(
                    {"sentences": canned_data, "embeddings": corpus_embeddings},
                    self.embedding_cache_path,
                )

            cache_data = joblib.load(self.embedding_cache_path)
            self.corpus_phrases = cache_data["sentences"]
            self.corpus_embeddings = cache_data["embeddings"]

            self.embedding_size = (
                self.corpus_embeddings[0].shape[0] if len(self.corpus_embeddings) > 0 else 0
            )
            self.index = hnswlib.Index(space="cosine", dim=self.embedding_size)

            # CHECK IF INDEX IS PRESENT
            if os.path.exists(self.index_path):
                self.logger.info("Loading index at ..." + self.index_path)
                self.index.load_index(self.index_path)
            else:
                # Create the HNSWLIB index
                self.logger.info("Start creating HNSWLIB index")
                self.index.init_index(max_elements=10000, ef_construction=400, M=64)
                self.index = self._create_index(self.index)
            self.index.set_ef(50)

        else:
            self.logger.warning("No canned phrases present.")

        self._model_loaded = True
        self.logger.debug(f"cannedPhrases count: {len(self.corpus_phrases)}")
        self.logger.info(f"Loaded {self.predictor_name} predictor.")

    def _create_index(self, ind):
        ind.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))
        self.logger.info("Saving index to:" + self.index_path)
        ind.save_index(self.index_path)
        return ind

    @property
    def sentences_db_path(self):
        return os.path.join(self._personalized_resources_path, self._sentences_db)

    @property
    def model_loaded(self):
        return self._model_loaded

    # Uses the semantic search index to find the top_k_hits
    def _find_semantic_matches(self, context, sent_prediction: Prediction) -> Prediction:
        self.logger.debug("Finding semantic matches")
        try:
            direct_matchedSentences = [s.word for s in sent_prediction]
            question_embedding = self.embedder.encode(context)

            # We use hnswlib knn_query method to find the top_k_hits
            corpus_ids, distances = self.index.knn_query(question_embedding, k=10)

            # We extract corpus ids and scores for the first query
            hits = [
                {"corpus_id": id, "score": 1 - score}
                for id, score in zip(corpus_ids[0], distances[0])
            ]
            hits = sorted(hits, key=lambda x: x["score"], reverse=True)
            for i in range(0, len(hits)):
                ret_sent = self.corpus_phrases[hits[i]["corpus_id"]]
                if ret_sent.strip() not in direct_matchedSentences:
                    sent_prediction.add_suggestion(
                        Suggestion(ret_sent.strip(), float(hits[i]["score"]), self.predictor_name)
                    )

        except Exception as e:
            self.logger.error(f"Exception in CannedPhrasePredictor find_semantic_matches {e}")
            raise e

        self.logger.debug(f"Found {len(sent_prediction)} semantic matches")
        return sent_prediction

    def _find_direct_matches(self, context, sent_prediction: Prediction) -> Prediction:
        self.logger.debug("Finding direct matches")
        try:
            canned_phrases = self.cannedData.all_phrases_as_dict()

            total_sent = sum(canned_phrases.values())
            context_StemmedWords = [self.stemmer.stem(w) for w in word_tokenize(context)]

            rows = []
            for k, _ in canned_phrases.items():
                matchfound = 0
                sentence_StemmedWords = [self.stemmer.stem(w) for w in word_tokenize(k)]
                for c in context_StemmedWords:
                    if c in sentence_StemmedWords:
                        matchfound = matchfound + 1
                new_row = {
                    "sentence": k,
                    "matches": matchfound,
                    "probability": float(canned_phrases[k] / total_sent),
                }
                rows.append(new_row)

            sorted_rows = sorted(
                rows, key=lambda x: (x["matches"], x["probability"]), reverse=True
            )
            for row in sorted_rows:
                if row["matches"] > 0:
                    sent_prediction.add_suggestion(
                        Suggestion(
                            row["sentence"],
                            row["matches"] + row["probability"],
                            self.predictor_name,
                        )
                    )
        except Exception as e:
            self.logger.error(f"Exception in CannedPhrasePredictor find_direct_matches: {e}")

        self.logger.debug(f"Found {len(sent_prediction)} direct matches")
        return sent_prediction

    def _getTopInitialPhrases(self, sent_prediction: Prediction, count=5) -> Prediction:
        canned_data = self.cannedData.all_phrases_as_dict()

        total_sent = sum(canned_data.values())
        probs = {}
        for k, v in canned_data.items():
            probs[k] = float(v / total_sent)

        sorted_x = collections.OrderedDict(
            sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        )
        for k, v in list(sorted_x.items())[:count]:
            sent_prediction.add_suggestion(Suggestion(k, v, self.predictor_name))
        return sent_prediction

    # base class method
    def predict(self, max_partial_prediction_size, filter=None):

        sent_prediction = Prediction()
        word_prediction = Prediction()  # Not used in this predictor

        try:
            context = self.context_tracker.context

            if context.strip() == "":
                # GET max_partial_prediction_size MOST FREQUENT SENTENCES
                sent_prediction = self._getTopInitialPhrases(
                    sent_prediction, max_partial_prediction_size
                )
                return sent_prediction, word_prediction

            # get matching sentences
            # First get direct matches based on both databases:
            sent_prediction = self._find_direct_matches(context, sent_prediction)

            # Only do a semantic search if we have more than 2 tokens in context
            if self.context_tracker.token_count >= 2:
                sent_prediction = self._find_semantic_matches(context, sent_prediction)

        except Exception as e:
            self.logger.error(f"Exception in cannedPhrases Predict: {e}")

        if len(sent_prediction) == 0:
            self.logger.error("No canned phrases found")

        self.logger.info(f"Got {len(sent_prediction)} sentence suggestions.")
        return sent_prediction[:max_partial_prediction_size], word_prediction

    def learn(self, phrase: str):
        # For the cannedPhrase predictor, learning adds the sentence to the PSMCannedPhrases
        if self.learn_enabled:
            try:
                # ADD THE NEW PHRASE TO THE EMBEDDINGS, AND RECREATE THE INDEX.
                phrase = phrase.strip()

                if phrase not in self.corpus_phrases:
                    phrase_emb = self.embedder.encode(phrase)
                    self.corpus_embeddings = numpy.vstack((self.corpus_embeddings, phrase_emb))
                    self.corpus_phrases.append(phrase)
                    joblib.dump(
                        {"sentences": self.corpus_phrases, "embeddings": self.corpus_embeddings},
                        self.embedding_cache_path,
                    )
                    self.index = self._create_index(self.index)

                # ADD THE NEW PHRASE TO THE DATABASE
                self.cannedData.learn(phrase)

            except Exception as e:
                self.logger.error(f"Exception in LEARN CANNED PHRASES SENTENCES. {e}")
