# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: GPL-3.0-or-later

import collections
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import hnswlib
import joblib
import nltk
import numpy
import torch
import transformers
from nltk import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer

from ..utilities.databaseutils.sqllite_dbconnector import SQLiteDatabaseConnector
from .predictor import Predictor
from .utilities.nlp import NLP
from .utilities.prediction import Prediction, Suggestion


class SentenceCompletionPredictor(Predictor):
    """
    SentenceCompletionPredictor is a class that provides functionality for sentence completion prediction using a pre-trained language model and a corpus of sentences.

    Methods:
        configure(self):
            Configures the predictor by loading the necessary models, embeddings, and indexes.
        load_model(self) -> None:
            Loads the pre-trained language model for sentence generation.
        retrieve(self):
            Property to get the retrieve attribute.
        retrieve(self, value):
            Property to set the retrieve attribute.
        _set_seed(self, seed):
            Sets the random seed for reproducibility.
        _read_personalized_toxic_words(self):
            Reads personalized allowed toxic words from a file.
        _extract_svo(self, sent):
            Extracts subject-verb-object (SVO) from a given sentence.
        _ngram_to_string(self, ngram):
            Converts an n-gram to a string.
        _filter_text(self, text):
            Filters the text to check for blacklisted words.
        _textInCorpus(self, text):
            Checks if the given text is in the corpus and returns the similarity score.
        _retrieve_fromDataset(self, context):
            Retrieves sentences from the dataset that match the given context.
        _checkRepetition(self, text):
            Checks for repetitive bigrams and trigrams in the text.
        _generate(self, context: str, num_gen: int) -> Prediction:
            Generates sentence completions for the given context.
        model_loaded(self):
            Property to check if the model is loaded.
        predict(self, max_partial_prediction_size: int, filter: Optional[str] = None):
            Predicts sentence completions based on the given context.
        learn(self, change_tokens):
            Learns from the given change tokens by adding them to the database.
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._retrieveaac = None
                self._model_loaded = False
    """

    def configure(self):
        self._model_loaded = False
        self.corpus_sentences = []
        self.sentence_generator: transformers.Pipeline | None = None

        self.nlp = NLP().get_nlp()

        if torch.cuda.is_available():
            self.device = "cuda"
            self.n_gpu = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.n_gpu = torch.mps.device_count()
        else:
            self.device = "cpu"
            self.n_gpu = 0

        # check if saved torch model exists
        self.load_model()
        self.stemmer = PorterStemmer()

        # CREATE INDEX TO QUERY DATABASE
        self.embedder = SentenceTransformer(
            str(self._sentence_transformer_model),
            device=self.device,
            local_files_only=True,
            tokenizer_kwargs={"clean_up_tokenization_spaces": "True"},
        )
        self.embedding_size = 384  # Size of embeddings
        self.top_k_hits = 2  # Output k hits
        self.n_clusters = 350

        # We use Inner Product (dot-product) as Index.
        # We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        self.index = hnswlib.Index(space="cosine", dim=self.embedding_size)

        with open(self.retrieve_database) as f:
            self.corpus_sentences = [s.strip() for s in f.readlines()]

        with open(self.blacklist_file) as f:
            self.blacklist_words = [s.strip() for s in f.readlines()]

        self.personalized_allowed_toxicwords = self._read_personalized_toxic_words()

        self.OBJECT_DEPS = {
            "dobj",
            "pobj",
            "dative",
            "attr",
            "oprd",
            "npadvmod",
            "amod",
            "acomp",
            "advmod",
        }
        self.SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}

        # tags that define wether the word is wh-
        self.WH_WORDS = {"WP", "WP$", "WRB"}
        self.stopwords = []
        stoplist = open(self.stopwordsFile).readlines()
        for s in stoplist:
            self.stopwords.append(s.strip())

        if not Path.is_file(Path(self.embedding_cache_path)):
            self.corpus_embeddings = self.embedder.encode(
                self.corpus_sentences, show_progress_bar=True, convert_to_numpy=True
            )
            joblib.dump(
                {"sentences": self.corpus_sentences, "embeddings": self.corpus_embeddings},
                self.embedding_cache_path,
            )

        else:
            cache_data = joblib.load(self.embedding_cache_path)
            self.corpus_sentences = cache_data["sentences"]
            self.corpus_embeddings = cache_data["embeddings"]

        # LOAD INDEX IF EXISTS, ELSE CREATE INDEX
        if Path.exists(Path(self.index_path)):
            self.logger.debug("Loading index...")
            self.index.load_index(str(self.index_path))

        else:
            # creating the embeddings pkl file
            self.logger.debug(" index does not exist, creating index")

            # Create the HNSWLIB index
            self.logger.debug("Start creating HNSWLIB index")
            self.index.init_index(max_elements=20000, ef_construction=400, M=64)

            # Then we train the index to find a suitable clustering
            self.index.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))

            self.logger.debug(f"Saving index to: {self.index_path}")
            self.index.save_index(self.index_path)

        # Controlling the recall by setting ef:
        self.index.set_ef(50)  # ef should always be > top_k_hits

        if not Path.is_file(Path(self.sent_database)):
            self.logger.debug(f"{self.sent_database} not found, creating it")
            columns = ["sentence TEXT UNIQUE", "count INTEGER"]
            SQLiteDatabaseConnector(self.sent_database).create_table("sentences", columns)

    def load_model(self) -> None:
        self.logger.debug(f"{__name__} loading model {str(self._modelname)}")

        if os.path.exists(self.modelname):
            try:
                self.logger.debug(f"Loading gpt2 model from {str(self.modelname)}")

                device = 0 if self.device == "cuda" or self.device == "mps" else -1

                tokenizer = transformers.GPT2Tokenizer.from_pretrained(self.tokenizer)
                assert tokenizer is not None
                model = transformers.GPT2LMHeadModel.from_pretrained(self.modelname)
                assert model is not None

                tokenizer.pad_token_id = tokenizer.eos_token_id

                self.sentence_generator = transformers.pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )
                assert self.sentence_generator is not None
                self._model_loaded = True
            except Exception as e:
                self.logger.error(f"Exception in SentenceCompletionPredictor load_model = {e}")
                self._model_loaded = False

        self.logger.debug(f"SentenceCompletionPredictor MODEL status: {self.model_loaded}")

    @property
    def retrieve(self):
        return self._retrieveaac

    @retrieve.setter
    def retrieve(self, value):
        self._retrieveaac = value

    def _set_seed(self, seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def _read_personalized_toxic_words(self):

        path = Path(self.personalized_allowed_toxicwords_file)
        if not path.exists():
            with open(self.personalized_allowed_toxicwords_file, "w") as f:
                pass

        with open(self.personalized_allowed_toxicwords_file) as f:
            self.personalized_allowed_toxicwords = f.readlines()

        self.personalized_allowed_toxicwords = [
            s.strip() for s in self.personalized_allowed_toxicwords
        ]
        self.logger.debug(f"UPDATED TOXIC WORDS = {self.personalized_allowed_toxicwords}")
        return self.personalized_allowed_toxicwords

    def _extract_svo(self, sent):
        doc = self.nlp(sent)
        sub = []
        at = []
        ve = []
        imp_tokens = []
        for token in doc:
            # is this a verb?
            if token.pos_ == "VERB":
                ve.append(token.text)
                if (
                    token.text.lower() not in self.stopwords
                    and token.text.lower() not in imp_tokens
                ):
                    imp_tokens.append(token.text.lower())
            # is this the object?
            if token.dep_ in self.OBJECT_DEPS or token.head.dep_ in self.OBJECT_DEPS:
                at.append(token.text)
                if (
                    token.text.lower() not in self.stopwords
                    and token.text.lower() not in imp_tokens
                ):
                    imp_tokens.append(token.text.lower())
            # is this the subject?
            if token.dep_ in self.SUBJECT_DEPS or token.head.dep_ in self.SUBJECT_DEPS:
                sub.append(token.text)
                if (
                    token.text.lower() not in self.stopwords
                    and token.text.lower() not in imp_tokens
                ):
                    imp_tokens.append(token.text.lower())
        return imp_tokens

    def _ngram_to_string(self, ngram):
        "|".join(ngram)

    def _filter_text(self, text):
        res = False
        words = []
        toxicwordsList = list(
            set(self.blacklist_words) - set(self.personalized_allowed_toxicwords)
        )

        if any(x in text.lower().split() for x in toxicwordsList):
            words = list(set(text.lower().split()) & set(toxicwordsList))
            self.logger.warning("blacklisted word is present!!")
            res = True
        return (res, words)

    def _textInCorpus(self, text):
        query_embedding = self.embedder.encode(text)

        # We use hnswlib knn_query method to find the top_k_hits
        corpus_ids, distances = self.index.knn_query(query_embedding, k=self.top_k_hits)
        hits = [
            {"corpus_id": id, "score": 1 - score} for id, score in zip(corpus_ids[0], distances[0])
        ]
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)
        self.logger.debug(
            f"score = {hits[0]['score']}, corpus_id = {hits[0]['corpus_id']}, len(corpus_sentences) = {len(self.corpus_sentences)}, corpus_embeddings.shape = {self.corpus_embeddings[0].shape}"
        )
        self.logger.debug(
            f"text = {text}, score = {hits[0]['score']}, sentence = {self.corpus_sentences[hits[0]['corpus_id']]}"
        )
        return hits[0]["score"]

    def _retrieve_fromDataset(self, context):
        pred = Prediction()
        probs = {}

        lines = open(self.retrieve_database).readlines()
        retrieved = []
        totalsent = len(lines)
        for each in lines:
            # if(each.lower().find(context.lower())!=-1):
            if each.lower().startswith(context.lower()):
                each = re.split("[.\n?!]", each)[0]
                retrieved.append(each)
        retrieved_set = set(retrieved)
        self.logger.debug(
            f"len(retrieved_set = {len(retrieved_set)}) len(retrieved) = {len(retrieved)}"
        )
        for s in retrieved_set:
            probs[s] = float(retrieved.count(s)) / totalsent
        try:
            pers_results: dict[str, float] = {}
            totalsentences = 0
            dbconn = SQLiteDatabaseConnector(self.sent_database)
            dbconn.connect()
            count = 0
            # CHECK IF SENTENCE EXISTS IN THE DATABASE
            res = dbconn.fetch_all("SELECT * FROM sentences")
            if res:
                for r in res:
                    sent = r[0]
                    totalsentences = totalsentences + r[1]
                    if sent.lower().startswith(context.lower()):
                        sent = re.split("[.\n?!]", sent)[0]
                        if sent in pers_results:
                            pers_results[sent] = pers_results[sent] + r[1]
                        else:
                            pers_results[sent] = r[1]
                for k, v in pers_results.items():
                    probs[k] = float(v / totalsentences)

            sorted_x = collections.OrderedDict(
                sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            )
            count = 0
            addedcompletions = []
            for k, v in sorted_x.items():
                if count > 5:
                    break
                k = k[len(context) :]
                if k not in addedcompletions:
                    pred.add_suggestion(Suggestion(k, v, self.predictor_name))
                    addedcompletions.append(k)
                    count = count + 1
        except Exception as e:
            self.logger.error(f"Exception in {self.predictor_name}, {__name__} function {e}")

        return pred

    def _checkRepetition(self, text):
        tokens = nltk.word_tokenize(text)

        # Create bigrams and check if there are any repeititions.
        bgs = nltk.bigrams(tokens)
        # compute frequency distribution for all the bigrams in the text
        fdist = nltk.FreqDist(bgs)
        fdist = {k: v for (k, v) in fdist.items() if v >= 2}
        if fdist != {}:
            return True

        # Create trigrams and check if there are any repeititions.
        bgs = nltk.trigrams(tokens)
        # compute frequency distribution for all the trigrams in the text
        fdist = nltk.FreqDist(bgs)
        fdist = {k: v for (k, v) in fdist.items() if v >= 2}
        if fdist != {}:
            return True
        return False

    def _generate(self, context: str, num_gen: int) -> Prediction:
        """
        _generate: generates completions for the given context
        Args:
            context: str: context for which completions are to be generated
            num_gen: int: number of completions to be generated
            predi: Prediction: object to store the generated completions
        Returns:
            predi: Prediction: object with generated completions
        """
        try:
            predictions = Prediction()

            gen_context = "<bos>" + context

            if gen_context.endswith((".", "!", "?")):
                gen_context = gen_context + " <eos>"

            result = self.sentence_generator(
                gen_context,
                do_sample=False,
                max_new_tokens=20,
                num_return_sequences=10,
                num_beams=10,
                num_beam_groups=10,
                diversity_penalty=1.5,
                repetition_penalty=1.1,
            )

            if isinstance(result, List) and all(isinstance(item, dict) for item in result):
                generated_text: List[Dict[str, Any]] = result
            else:
                raise TypeError(f"Unexpected type for result: {type(result)}")

            allsent: list[str] = []
            counts: Dict[str, float] = {}
            totalsent = 0

            for text in generated_text:
                sentence = text["generated_text"]
                self.logger.debug(f"Generated Text: {sentence}")

                # Cleanup and split the generated text into sentences
                # clean_sentences = self._clean_generated_text(sentence)

                # Get the first sentence between the markers
                clean_sentence = self.extract_text_between_markers(sentence)

                self.logger.debug(f"Full generated Text = {clean_sentence}")
                # gen_text_sent = sent_tokenize(clean_sentences[1])
                # currSentence = gen_text_sent[num_context_sentences - 1]

                # check for repetitive sentences
                if self._checkRepetition(clean_sentence):
                    self.logger.warning(f"Repetition in the sentence: {clean_sentence}")
                    continue

                remainderText = clean_sentence[len(context) :]
                remainderTextForFilter = re.sub(r"[?,!.\n]", "", remainderText.strip())

                filtered, _ = self._filter_text(remainderTextForFilter)

                if not filtered:
                    score = self._textInCorpus(clean_sentence.strip())

                    # TODO: DO WE THRESHOLD SCORES?
                    # TODO: DETOXIFY

                    if remainderTextForFilter != "":
                        if clean_sentence not in allsent:
                            imp_tokens = self._extract_svo(clean_sentence)
                            imp_tokens_reminder = []
                            # get important tokens only of the generated completion
                            for imp in imp_tokens:
                                if imp in word_tokenize(remainderTextForFilter):
                                    imp_tokens_reminder.append(imp)
                            present = False
                            for a in allsent:
                                for it in imp_tokens_reminder:
                                    if self.stemmer.stem(it) in [
                                        self.stemmer.stem(w)
                                        for w in word_tokenize(a[len(context) :])
                                    ]:
                                        present = True
                                        break
                            if not present:
                                allsent.append(clean_sentence)
                                counts[remainderText] = 1 * score
                                totalsent = totalsent + 1
                        else:
                            counts[remainderTextForFilter] = (
                                counts[remainderTextForFilter] + 1 * score
                            )
                            totalsent = totalsent + 1

            # toxic_filtered_sent = self.detoxify(allsent)
            self.logger.warning("toxic_filtered_sent not called.")

            for k, v in counts.items():
                counts[k] = float(v) / totalsent

            sorted_x = collections.OrderedDict(
                sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            )
            count = 0
            for k, v in sorted_x.items():

                if count == num_gen:
                    break
                self.logger.debug(f"sentence = {k} score = {v}")
                predictions.add_suggestion(Suggestion(k, v, self.predictor_name))
                count = count + 1

        except Exception as e:
            self.logger.error(f"Exception in SentenceCompletionPredictor.{e}")

        return predictions

    def _clean_generated_text(self, gentext):
        # Just clean the generated text of all the potential rubbish
        # return re.sub(r"[<>\[\]\d\n\t]|<bos>|<eos>|bos|eos", "", gentext)
        return re.split(r"[<>\[\]\d\n\t]|<bos>|<eos>|bos|eos", gentext)
        # return re.split(r"<bos> |<eos> |bos|eos|<bos>|<eos>|<|>|\[|\]|\d", gentext)

    def extract_text_between_markers(
        self, text: str, start_marker: str = "<bos>", end_marker: str = "<eos>"
    ) -> str:
        """
        Extract text between start_marker and end_marker. If end_marker is not present,
        stop at the first punctuation mark.

        Args:
            text (str): The input text.
            start_marker (str): The marker indicating the start of the text to extract.
            end_marker (str): The marker indicating the end of the text to extract.

        Returns:
            str: The extracted text.
        """
        # Find the start marker
        start_index = text.find(start_marker)
        if start_index == -1:
            return ""  # Start marker not found

        # Move the index to the end of the start marker
        start_index += len(start_marker)

        # Find the end marker
        end_index = text.find(end_marker, start_index)
        if end_index == -1:
            # End marker not found, stop at the first punctuation mark
            match = re.search(r"[.?!]", text[start_index:])
            if match:
                end_index = start_index + match.start()
            else:
                end_index = len(text)  # No punctuation found, take the rest of the text

        return text[start_index:end_index].strip()

    @property
    def model_loaded(self):
        return self._model_loaded

    def predict(self, max_partial_prediction_size: int, filter: Optional[str] = None):
        sentence_predictions = Prediction()
        word_predictions = Prediction()  # not used in this predictor

        context = self.context_tracker.context.lstrip()
        self.logger.debug(f"context inside predictor predict = {context}")

        if not context:
            # TODO: Fix this
            self.logger.debug(f"context is empty, loading top sentences from {self._startsents}")

            sentence_predictions = self.load_n_start_sentences(max_partial_prediction_size)

            if not sentence_predictions:
                self.logger.warning("No start sentences found.  File may be missing or corrupted.")

        else:
            # If we are testing generation models
            if self.test_generalsentenceprediction:
                self.logger.warning("Testing general sentence prediction")
                sentence_predictions = self._generate(
                    "<bos> " + context.strip(), max_partial_prediction_size
                )

            else:
                sentence_predictions = self._retrieve_fromDataset(context)
                self.logger.debug(
                    f"retrieved {len(sentence_predictions)} sentences in {str(sentence_predictions)}"
                )
                remaining_predicitions_needed = max_partial_prediction_size - len(
                    sentence_predictions
                )
                self.logger.debug(
                    f"remaining_predicitions_needed = {remaining_predicitions_needed}"
                )

                # ONLY IF THE GENERATION MODEL IS LOADED, GENERATE MODEL BASED PREDICTIONS
                if not self.retrieve and self.model_loaded and remaining_predicitions_needed > 0:
                    self.logger.debug(
                        f"generating {remaining_predicitions_needed} more predictions"
                    )
                    sentence_predictions = self._generate(
                        context.strip(), remaining_predicitions_needed
                    )

        return sentence_predictions, word_predictions

    def load_n_start_sentences(self, max_partial_prediction_size=-1):
        predictions = Prediction()

        with open(self.startsents) as f:
            data = f.readlines(max_partial_prediction_size)
            for sentence in data:
                predictions.add_suggestion(
                    Suggestion(sentence.strip(), float(1 / len(data)), self.predictor_name)
                )
        return predictions

    # Base class method
    def learn(self, change_tokens):
        # For the sentence completion predictor, learning adds the sentence to the database
        if self.learn_enabled:
            change_tokens = change_tokens.strip()
            self.logger.debug(f"learning, {change_tokens}")
            # add to sentence database
            try:
                dbconn = SQLiteDatabaseConnector(self.sent_database)
                dbconn.connect()
                count = 0
                # CHECK IF SENTENCE EXISTS IN THE DATABASE
                res = dbconn.fetch_all(
                    "SELECT count FROM sentences WHERE sentence = ?", (change_tokens,)
                )
                if res and len(res) > 0:
                    if len(res[0]) > 0:
                        count = int(res[0][0])

                # IF SENTENCE DOES NOT EXIST, ADD INTO DATABASE WITH COUNT = 1
                if count == 0:
                    self.logger.debug("count is 0, inserting into database")
                    dbconn.execute_query(
                        """
                    INSERT INTO sentences (sentence, count)
                    VALUES (?,?)""",
                        (change_tokens, 1),
                    )
                    # update retrieval index:
                    # self.index.load_index(self.index_path)
                    self.logger.debug(
                        "shape before: {} len*self.corpus_sentences = {}".format(
                            self.corpus_embeddings[0].shape, len(self.corpus_sentences)
                        )
                    )

                    self.logger.debug(
                        "sentence  {} not present, adding to embeddings and creating new index".format(
                            change_tokens
                        )
                    )
                    phrase_emb = self.embedder.encode(change_tokens.strip())
                    phrase_id = len(self.corpus_embeddings)
                    self.corpus_embeddings = numpy.vstack((self.corpus_embeddings, phrase_emb))
                    self.corpus_sentences.append(change_tokens.strip())
                    # np.save(self.embedding_cache_path,{'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings})
                    joblib.dump(
                        {"sentences": self.corpus_sentences, "embeddings": self.corpus_embeddings},
                        self.embedding_cache_path,
                    )
                    # with open(self.embedding_cache_path, "wb") as fOut:
                    #     pickle.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, fOut)

                    # Then we train the index to find a suitable clustering
                    self.logger.debug(
                        "phrase_emb.shape = {} id= {}".format(
                            str(phrase_emb[0].shape), str(len(self.corpus_embeddings))
                        )
                    )
                    self.index.add_items(phrase_emb, phrase_id)

                    self.logger.debug(f"Saving index to:{self.index_path}")
                    self.index.save_index(self.index_path)
                    self.logger.debug(
                        "shape after: {} len*self.corpus_sentences =  {}".format(
                            str(self.corpus_embeddings.shape), str(len(self.corpus_sentences))
                        )
                    )

                    # DEALING WITH PERSONALIZED, ALLOWED TOXIC WORDS
                    # if sentence to be learnt contains a toxic word, add the toxic word to the "allowed" word list
                    res, words = self._filter_text(change_tokens)
                    if res:
                        for tox in words:
                            self.logger.debug(f"toxic words to be added to personalized db: {tox}")
                            if tox not in self.personalized_allowed_toxicwords:
                                self.personalized_allowed_toxicwords.append(tox)
                                fout = open(self.personalized_allowed_toxicwords_file, "w")
                                for tox_word in self.personalized_allowed_toxicwords:
                                    fout.write(tox_word + "\n")
                                fout.close()

                # ELSE, IF SENTENCE EXIST, ADD INTO DATABASE WITH UPDATED COUNT
                else:
                    self.logger.debug("sentence exists, updating count")
                    dbconn.execute_query(
                        """
                    UPDATE sentences SET count = ? where sentence = ?""",
                        (count + 1, change_tokens),
                    )
                dbconn.commit()
            except Exception as e:
                self.logger.error(f"Exception in SentenceCompletionPredictor learn  = {e}")
            finally:
                dbconn.close()
