# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import collections
import joblib
import hnswlib
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
from sentence_transformers import SentenceTransformer

from ConvAssist.predictor.utilities.prediction import Prediction
from ConvAssist.predictor import Predictor
from ConvAssist.utilities.suggestion import Suggestion

class CannedPhrasesPredictor(Predictor):
    """
    Searches the canned phrase database for matching next words and sentences
    """

    def __init__(
            self, 
            config, 
            context_tracker, 
            predictor_name, 
            short_desc=None, 
            long_desc=None, 
            dbconnection=None
        ):
        Predictor.__init__(
            self, config, context_tracker, predictor_name, short_desc, long_desc
        )
        self.db = None
        self.dbconnection = dbconnection
        self.cardinality = 3
        self.learn_mode_set = False
        self.dbclass = None
        self.dbuser = None
        self.dbpass = None
        self.dbhost = None
        self.dbport = None
        self.MODEL_LOADED = False
        self._database = None
        self._deltas = None
        self._learn_mode = None
        self.config = config
        self.name = predictor_name
        self.context_tracker = context_tracker
        self._read_config()
        self.seed = 42
        self.cannedPhrases_counts={}
        self.stemmer = PorterStemmer()
        self.embedder = SentenceTransformer(self.sbertmodel)
        self.pers_cannedphrasesLines = open(self.personalized_cannedphrases, "r").readlines()
        self.pers_cannedphrasesLines = [s.strip() for s in self.pers_cannedphrasesLines]

        self.log.info("Logging inside canned phrases init!!!")
        if(not os.path.isfile(self.sentences_db)):
            self.createSentDB(self.sentences_db)


        if(not os.path.isfile(self.embedding_cache_path)):
            self.corpus_embeddings = self.embedder.encode(self.pers_cannedphrasesLines, show_progress_bar=True, convert_to_numpy=True)
            # np.save(self.embedding_cache_path,{'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings})
            joblib.dump({'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings},self.embedding_cache_path)

        else:
            # cache_data = np.load(self.embedding_cache_path)

            cache_data = joblib.load(self.embedding_cache_path)
            self.corpus_sentences = cache_data['sentences']
            self.corpus_embeddings = cache_data['embeddings']



        # #### IF CANNED PHRASES EMBEDDING NOT FOUND, 
        # if(not os.path.isfile(self.embedding_cache_path)):
        #     self.corpus_embeddings = self.embedder.encode(self.pers_cannedphrasesLines, show_progress_bar=True, convert_to_numpy=True)
        #     with open(self.embedding_cache_path, "wb") as fOut:
        #         pickle.dump({'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings}, fOut)

        # else:
        #     with open(self.embedding_cache_path, "rb") as fIn:
        #         cache_data = pickle.load(fIn)
        #         self.corpus_embeddings = cache_data['embeddings']
        #         self.corpus_sentences = cache_data['sentences']

        self.n_clusters = 20     ### clusters for hnswlib index
        self.embedding_size = self.corpus_embeddings.shape[1]
        self.index = hnswlib.Index(space = 'cosine', dim = self.embedding_size)

        ####### CHECK IF INDEX IS PRESENT
        if os.path.exists(self.index_path):
            self.log.info("Loading index at ..."+ self.index_path)
            self.index.load_index(self.index_path)
        else:
            ### Create the HNSWLIB index
            self.log.info("Start creating HNSWLIB index")
            self.index.init_index(max_elements = 10000, ef_construction = 400, M = 64)
            self.index = self.create_index(self.index)
        self.index.set_ef(50)

        self.MODEL_LOADED = True

    def create_index(self, ind):
        ind.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))
        self.log.info("Saving index to:"+ self.index_path)
        ind.save_index(self.index_path)
        return ind

    def is_model_loaded(self):
        return self.MODEL_LOADED

    def insert_into_tables(self, conn, sql, task):
        cur = conn.cursor()
        cur.execute(sql, task)
        conn.commit()

        return cur.lastrowid

    def recreate_canned_db(self, personalized_corpus):
        self.log.info("inside CannedPhrasesPredictor recreate_canned_db")

        self.corpus_sentences = []
        self.pers_cannedphrasesLines = open(self.personalized_cannedphrases, "r").readlines()
        self.pers_cannedphrasesLines = [s.strip() for s in self.pers_cannedphrasesLines]

        try:
            raise NotImplementedError
            #### RETRIEVE ALL SENTENCES FROM THE DATABASE
            conn_sent = sqlite3.connect(self.sentences_db)
            c = conn_sent.cursor()
            c.execute("SELECT * FROM sentences")
            res_all = c.fetchall()
            #self.log.debug("len(sent_db) = "+ str(len(res_all))+ " len(self.pers_cannedphrases) = "+ str(len(self.pers_cannedphrasesLines)))
            for r in res_all:
                self.cannedPhrases_counts[r[0]] = r[1]

            if(os.path.isfile(self.embedding_cache_path)):
                # cache_data = np.load(self.embedding_cache_path)
                cache_data = joblib.load(self.embedding_cache_path)
                self.corpus_sentences = cache_data['sentences']
                self.corpus_embeddings = cache_data['embeddings']


            # if(os.path.isfile(self.embedding_cache_path)):
            #     with open(self.embedding_cache_path, "rb") as fIn:
            #         cache_data = pickle.load(fIn)
            #         self.corpus_embeddings = cache_data['embeddings']
            #         self.corpus_sentences = cache_data['sentences']
            else:
                self.log.info("In Recreate_DB of cannedPhrasesPredictor, EMBEDDINGS FILE DOES NOT EXIST!!! ")


            ###### check if cannedPhrases file has been modified!!! 
            if(set(self.corpus_sentences) != set(self.pers_cannedphrasesLines) ):
                self.log.info("Canned Phrases has been modified externally.. Recreating embeddings and indices")
                phrasesToAdd = set(self.pers_cannedphrasesLines) - set(self.corpus_sentences)
                phrasesToRemove = set(self.corpus_sentences) - set(self.pers_cannedphrasesLines)
                print("phrases to add = ", str(phrasesToAdd))
                print("phrases to remove = ", str(phrasesToRemove))
                self.log.info("phrases to add Recreate_DB of cannedPhrasesPredictor = "+ str(phrasesToAdd))
                self.log.info("phrases to phrasesToRemove Recreate_DB of cannedPhrasesPredictor= "+ str(phrasesToRemove))
                #### update embeddings, 
                self.corpus_embeddings = self.embedder.encode(self.pers_cannedphrasesLines, show_progress_bar=True, convert_to_numpy=True)
                # np.save(self.embedding_cache_path,{'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings})
                joblib.dump({'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings}, self.embedding_cache_path)
                # with open(self.embedding_cache_path, "wb") as fOut:
                #     pickle.dump({'sentences': self.pers_cannedphrasesLines, 'embeddings': self.corpus_embeddings}, fOut)

                #### update index:
                self.index = self.create_index(self.index)
            else:
                self.log.info("Recreate_DB of cannedPhrasesPredictor: NO modifications to cannedPhrases= ")

        except Exception as e:
            self.log.info("Exception in CannedPhrasePredictor recreateDB  = "+e)

    def createSentDB(self, dbname):
        self.log.info("IN createSentDB")
        try:
            raise NotImplementedError
            #self.log.debug("creating db = "+ dbname)
            conn = sqlite3.connect(dbname)
            c = conn.cursor()
            c.execute('''
                    CREATE TABLE IF NOT EXISTS sentences
                    (sentence TEXT UNIQUE, count INTEGER)
                    ''')
            conn.commit()
        except Exception as e:
            #self.log.debug("Exception in createSentDB  = "+str(e))
            pass

    def find_semantic_matches(self,context, sent_prediction, cannedph):
        try:
            direct_matchedSentences = [s.word for s in sent_prediction]
            question_embedding = self.embedder.encode(context)
            #We use hnswlib knn_query method to find the top_k_hits
            corpus_ids, distances = self.index.knn_query(question_embedding, k=5)
            # We extract corpus ids and scores for the first query
            hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
            hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            for i in range(0, len(hits)):
                ret_sent = self.pers_cannedphrasesLines[hits[i]['corpus_id']]
                if ret_sent.strip() not in direct_matchedSentences:
                    sent_prediction.add_suggestion(Suggestion(ret_sent.strip(), hits[i]["score"], self.name))
        except Exception as e:
            #self.log.debug("Exception in CannedPhrasePredictor find_semantic_matches  = "+e)
            pass

        return sent_prediction

    def find_direct_matches(self,context, lines, sent_prediction, cannedph):
        try:
            total_sent = sum(cannedph.values())
            context_StemmedWords = [self.stemmer.stem(w) for w in context.split()]
            num_contextWords = len(context_StemmedWords)
            rows = []
            for k,v in cannedph.items():
                matchfound = 0
                sentence_StemmedWords = [self.stemmer.stem(w) for w in word_tokenize(k)]
                for c in context_StemmedWords:
                    if c in sentence_StemmedWords:
                        matchfound = matchfound+1
                new_row = {'sentence':k, 'matches':matchfound, 'probability':float(cannedph[k]/total_sent)}
                rows.append(new_row)
            scores = pd.DataFrame.from_records(rows)
            sorted_df = scores.sort_values(by = ['matches', 'probability'], ascending = [False, False])
            for index, row in sorted_df.iterrows():
                if(row["matches"]>0):
                    sent_prediction.add_suggestion(Suggestion(row['sentence'], row["matches"]+row["probability"], self.name))
        except Exception as e:
            self.log.debug("Exception in CannedPhrasePredictor find_direct_matches  = "+e)

        return sent_prediction

    def getTop5InitialPhrases(self, cannedph, sent_prediction):
        total_sent = sum(cannedph.values())
        probs = {}
        for k,v in cannedph.items():
            probs[k] = float(v/total_sent)

        sorted_x = collections.OrderedDict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
        count = 0
        for k,v in sorted_x.items():
            if(count==5):
                break
            sent_prediction.add_suggestion(Suggestion(k, v, self.name))
            count = count+1
        return sent_prediction

    def predict(self, max_partial_prediction_size, filter):

        tokens = [""] * self.cardinality
        sent_prediction = Prediction()
        word_prediction = Prediction()
        try:
            context = self.context_tracker.past_stream().strip()

            if(context==""):
                ##### GET 5 MOST FREQUENT SENTENCES 
                sent_prediction = self.getTop5InitialPhrases(self.cannedPhrases_counts, sent_prediction)
                return sent_prediction, word_prediction

            ###### get matching sentences 
            ###### First get direct matches based on both databases: 

            sent_prediction = self.find_direct_matches(context, self.pers_cannedphrasesLines, sent_prediction, self.cannedPhrases_counts)

            ###### Get semantic matches based on both databases: 
            sent_prediction = self.find_semantic_matches(context, sent_prediction, self.cannedPhrases_counts)
            #self.log.debug("sent_prediction = "+str(sent_prediction))

        except Exception as e:
            self.log.debug("Exception in cannedPhrases Predict = "+e)

        return sent_prediction, word_prediction

    def close_database(self):
        raise NotImplementedError
        self.db.close_database()

    def learn(self, change_tokens):
        #### For the cannedPhrase predictor, learning adds the sentence to the PSMCannedPhrases 
        if self.learn_mode == "True":
            #self.log.debug("learning ..."+change_tokens)
            try:
                raise NotImplementedError
                #### ADD THE NEW PHRASE TO THE EMBEDDINGS, AND RECREATE THE INDEX. 
                if(change_tokens not in self.corpus_sentences):
                    #self.log.debug("phrase "+ change_tokens+ " not present, adding to embeddings and creating new index")
                    phrase_emb = self.embedder.encode(change_tokens.strip())
                    self.corpus_embeddings = np.vstack((self.corpus_embeddings, phrase_emb))
                    self.corpus_sentences.append(change_tokens.strip())
                    # np.save(self.embedding_cache_path,{'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings})
                    joblib.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, self.embedding_cache_path)
                    # with open(self.embedding_cache_path, "wb") as fOut:
                    #     pickle.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, fOut)
                    self.index = self.create_index(self.index)

                conn = sqlite3.connect(self.sentences_db)
                c = conn.cursor()
                count = 0
                #### CHECK IF SENTENCE EXISITS IN THE DATABASE
                c.execute("SELECT count FROM sentences WHERE sentence = ?", (change_tokens,))
                res = c.fetchall()
                if len(res) > 0:
                    if len(res[0]) > 0:
                        count = int(res[0][0])

                ### IF SENTENCE DOES NOT EXIST, ADD INTO DATABASE WITH COUNT = 1
                if count==0:
                    self.pers_cannedphrasesLines.append(change_tokens)
                    fout = open(self.personalized_cannedphrases, "w")
                    for l in self.pers_cannedphrasesLines:
                        fout.write(l+"\n")
                    fout.close()
                    c.execute('''
                    INSERT INTO sentences (sentence, count)
                    VALUES (?,?)''', (change_tokens, 1))

                    self.cannedPhrases_counts[change_tokens] = 1
                ### ELSE, IF SENTENCE EXIST, ADD INTO DATABASE WITH UPDATED COUNT
                else:
                    c.execute('''
                    UPDATE sentences SET count = ? where sentence = ?''', (count+1, change_tokens))

                    self.cannedPhrases_counts[change_tokens] = count +1
                conn.commit()
            except Exception as e:
                self.log.debug("Exception in LEARN CANNED PHRASES SENTENCES  = "+e)

    def _read_config(self):
        self.static_resources_path = self.config.get(self.name, "static_resources_path")
        self.personalized_resources_path = self.config.get(self.name, "personalized_resources_path")
        self.learn_mode = self.config.get(self.name, "learn")
        self.personalized_cannedphrases = os.path.join(self.personalized_resources_path, self.config.get(self.name, "personalized_cannedphrases"))
        self.sentences_db  = os.path.join(self.personalized_resources_path, self.config.get(self.name, "sentences_db"))
        self.embedding_cache_path = os.path.join(self.personalized_resources_path, self.config.get(self.name, "embedding_cache_path"))
        self.index_path = os.path.join(self.personalized_resources_path, self.config.get(self.name, "index_path"))
        self.sbertmodel = os.path.join(self.static_resources_path, self.config.get(self.name, "sbertmodel"))