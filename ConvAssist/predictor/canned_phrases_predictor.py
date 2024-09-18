# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import collections
import logging
import os
from configparser import ConfigParser

import hnswlib
import joblib
import numpy
import pandas as pd
import torch
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer

from ConvAssist.context_tracker import ContextTracker
from ConvAssist.predictor.predictor import Predictor
from ConvAssist.predictor.utilities.prediction import Prediction
from ConvAssist.predictor.utilities.suggestion import Suggestion
from ConvAssist.utilities.databaseutils.sqllite_dbconnector import (
    SQLiteDatabaseConnector,
)


class CannedPhrasesPredictor(Predictor):
    """
    Searches the canned phrase database for matching next words and sentences
    """

    def configure(self):

        if torch.cuda.is_available():
            self.device = "cuda"
            self.n_gpu = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.n_gpu = torch.mps.device_count()
        else:
            self.device = "cpu"
            self.n_gpu = 0

        self.seed = 42
        self.cannedPhrases_counts = {}
        self.stemmer = PorterStemmer()
        self.embedder = SentenceTransformer(self.sbertmodel, device=self.device)

        with open(self.personalized_cannedphrases) as f:
            self.pers_cannedphrasesLines = [s.strip() for s in f.readlines()]

        # if(not os.path.isfile(self.sentences_db_path)):
        #     self.logger.debug(f"{self.sentences_db_path} does not exist, creating.")
        #     self.recreate_database()
        # else:
        #     self.logger.debug(f"{self.sentences_db_path} exists, not creating.")
        self.recreate_database()

        if not os.path.isfile(self.embedding_cache_path):
            self.logger.info(f"{self.embedding_cache_path} does not exist, creating")
            self.corpus_embeddings = self.embedder.encode(
                self.pers_cannedphrasesLines, show_progress_bar=True, convert_to_numpy=True
            )
            joblib.dump(
                {"sentences": self.pers_cannedphrasesLines, "embeddings": self.corpus_embeddings},
                self.embedding_cache_path,
            )

        else:
            self.logger.info(f"{self.embedding_cache_path} exists, loading embeddings from cache.")
            cache_data = joblib.load(self.embedding_cache_path)
            self.corpus_sentences = cache_data["sentences"]
            self.corpus_embeddings = cache_data["embeddings"]

        self.n_clusters = 20  # clusters for hnswlib index
        self.embedding_size = self.corpus_embeddings[0].shape[0]
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

        self.logger.debug(f"cannedPhrases count: {len(self.cannedPhrases_counts)}")

    def _create_index(self, ind):
        ind.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))
        self.logger.info("Saving index to:" + self.index_path)
        ind.save_index(self.index_path)
        return ind

    @property
    def sentences_db_path(self):
        return os.path.join(self._personalized_resources_path, self._sentences_db)

    def recreate_database(self):

        self.pers_cannedphrasesLines = self.read_personalized_corpus()

        try:
            # RETRIEVE ALL SENTENCES FROM THE DATABASE
            self.sentences_db = SQLiteDatabaseConnector(self.sentences_db_path)
            self.sentences_db.connect()

            if not os.path.isfile(self.sentences_db_path):
                columns = ["sentence TEXT UNIQUE", "count INTEGER"]
                self.sentences_db.create_table("sentences", columns)
            else:
                res_all = self.sentences_db.fetch_all("SELECT * FROM sentences")

            if res_all:
                for r in res_all:
                    self.cannedPhrases_counts[r[0]] = r[1]

            if os.path.isfile(self.embedding_cache_path):
                cache_data = joblib.load(self.embedding_cache_path)
                self.corpus_sentences = cache_data["sentences"]
                self.corpus_embeddings = cache_data["embeddings"]
            else:
                self.logger.info(
                    "In Recreate_DB of cannedPhrasesPredictor, EMBEDDINGS FILE DOES NOT EXIST!!! "
                )

            # check if cannedPhrases file has been modified!!!
            if set(self.corpus_sentences) != set(self.pers_cannedphrasesLines):
                self.logger.debug(
                    "Canned Phrases has been modified externally.. Recreating embeddings and indices"
                )
                phrasesToAdd = set(self.pers_cannedphrasesLines) - set(self.corpus_sentences)
                phrasesToRemove = set(self.corpus_sentences) - set(self.pers_cannedphrasesLines)
                self.logger.debug(
                    f"phrases to add Recreate_DB of cannedPhrasesPredictor = {str(phrasesToAdd)}"
                )
                self.logger.debug(
                    f"phrases to phrasesToRemove Recreate_DB of cannedPhrasesPredictor = {str(phrasesToRemove)}"
                )

                # update embeddings
                self.corpus_embeddings = self.embedder.encode(
                    self.pers_cannedphrasesLines, show_progress_bar=True, convert_to_numpy=True
                )
                joblib.dump(
                    {
                        "sentences": self.pers_cannedphrasesLines,
                        "embeddings": self.corpus_embeddings,
                    },
                    self.embedding_cache_path,
                )

                # update index:
                self.index = self._create_index(self.index)
            else:
                self.logger.info(
                    "Recreate_DB of cannedPhrasesPredictor: NO modifications to cannedPhrases"
                )

        except Exception as e:
            self.logger.error(f"CannedPhrasePredictor recreateDB: {e}")

        finally:
            self.sentences_db.close()

    def _find_semantic_matches(self, context, sent_prediction: Prediction, cannedph) -> Prediction:
        try:
            direct_matchedSentences = [s.word for s in sent_prediction]
            question_embedding = self.embedder.encode(context)
            # We use hnswlib knn_query method to find the top_k_hits
            corpus_ids, distances = self.index.knn_query(question_embedding, k=5)
            # We extract corpus ids and scores for the first query
            hits = [
                {"corpus_id": id, "score": 1 - score}
                for id, score in zip(corpus_ids[0], distances[0])
            ]
            hits = sorted(hits, key=lambda x: x["score"], reverse=True)
            for i in range(0, len(hits)):
                ret_sent = self.pers_cannedphrasesLines[hits[i]["corpus_id"]]
                if ret_sent.strip() not in direct_matchedSentences:
                    sent_prediction.add_suggestion(
                        Suggestion(ret_sent.strip(), hits[i]["score"], self.predictor_name)
                    )

        except Exception as e:
            self.logger.error(f"Exception in CannedPhrasePredictor find_semantic_matches {e}")
            raise e

        return sent_prediction

    def _find_direct_matches(self, context, lines, sent_prediction, cannedph):
        try:
            total_sent = sum(cannedph.values())
            context_StemmedWords = [self.stemmer.stem(w) for w in context.split()]
            num_contextWords = len(context_StemmedWords)
            rows = []
            for k, v in cannedph.items():
                matchfound = 0
                sentence_StemmedWords = [self.stemmer.stem(w) for w in word_tokenize(k)]
                for c in context_StemmedWords:
                    if c in sentence_StemmedWords:
                        matchfound = matchfound + 1
                new_row = {
                    "sentence": k,
                    "matches": matchfound,
                    "probability": float(cannedph[k] / total_sent),
                }
                rows.append(new_row)
            scores = pd.DataFrame.from_records(rows)
            sorted_df = scores.sort_values(by=["matches", "probability"], ascending=[False, False])
            for index, row in sorted_df.iterrows():
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

        return sent_prediction

    def _getTopInitialPhrases(self, cannedph, sent_prediction, count=5):
        total_sent = sum(cannedph.values())
        probs = {}
        for k, v in cannedph.items():
            probs[k] = float(v / total_sent)

        sorted_x = collections.OrderedDict(
            sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        )
        for k, v in list(sorted_x.items())[:count]:
            sent_prediction.add_suggestion(Suggestion(k, v, self.predictor_name))
        return sent_prediction

    # base class method
    def predict(self, max_partial_prediction_size, filter):

        sent_prediction = Prediction()
        word_prediction = Prediction()  # Not used in this predictor
        try:
            context = self.context_tracker.context.strip()

            if context == "":
                # GET 5 MOST FREQUENT SENTENCES
                sent_prediction = self._getTopInitialPhrases(
                    self.cannedPhrases_counts, sent_prediction
                )
                return sent_prediction, word_prediction

            # get matching sentences
            # First get direct matches based on both databases:
            sent_prediction = self._find_direct_matches(
                context, self.pers_cannedphrasesLines, sent_prediction, self.cannedPhrases_counts
            )

            # Get semantic matches based on both databases:
            sent_prediction = self._find_semantic_matches(
                context, sent_prediction, self.cannedPhrases_counts
            )

        except Exception as e:
            self.logger.error("Exception in cannedPhrases Predict: {e} ")

        if len(sent_prediction) == 0:
            self.logger.error("No canned phrases found")

        self.logger.info(
            f"End prediction. got {len(word_prediction)} word suggestions and {len(sent_prediction)} sentence suggestions"
        )
        return sent_prediction, word_prediction

    # base class method
    def learn(self, change_tokens):
        # For the cannedPhrase predictor, learning adds the sentence to the PSMCannedPhrases
        if self.learn_enabled:
            try:
                # ADD THE NEW PHRASE TO THE EMBEDDINGS, AND RECREATE THE INDEX.
                if change_tokens not in self.corpus_sentences:
                    phrase_emb = self.embedder.encode(change_tokens.strip())
                    self.corpus_embeddings = numpy.vstack((self.corpus_embeddings, phrase_emb))
                    self.corpus_sentences.append(change_tokens.strip())
                    joblib.dump(
                        {"sentences": self.corpus_sentences, "embeddings": self.corpus_embeddings},
                        self.embedding_cache_path,
                    )
                    self.index = self._create_index(self.index)

                self.sentences_db.connect()
                count = 0
                # TODO: Move all database funtionality to a separate class
                # CHECK IF SENTENCE EXISITS IN THE DATABASE
                res = self.sentences_db.fetch_all(
                    "SELECT count FROM sentences WHERE sentence = ?", (change_tokens,)
                )

                if res and len(res) > 0:
                    if len(res[0]) > 0:
                        count = int(res[0][0])

                # IF SENTENCE DOES NOT EXIST, ADD INTO DATABASE WITH COUNT = 1
                if count == 0:
                    self.pers_cannedphrasesLines.append(change_tokens)
                    fout = open(self.personalized_cannedphrases, "w")
                    for line in self.pers_cannedphrasesLines:
                        fout.write(line + "\n")
                    fout.close()
                    self.sentences_db.execute_query(
                        """
                    INSERT INTO sentences (sentence, count)
                    VALUES (?,?)""",
                        (change_tokens, 1),
                    )

                    self.cannedPhrases_counts[change_tokens] = 1
                # ELSE, IF SENTENCE EXIST, ADD INTO DATABASE WITH UPDATED COUNT
                else:
                    self.sentences_db.execute_query(
                        """
                    UPDATE sentences SET count = ? where sentence = ?""",
                        (count + 1, change_tokens),
                    )

                    self.cannedPhrases_counts[change_tokens] = count + 1
                self.sentences_db.commit()
            except Exception as e:
                self.logger.error("Exception in LEARN CANNED PHRASES SENTENCES  = {e}")
            finally:
                self.sentences_db.close()
