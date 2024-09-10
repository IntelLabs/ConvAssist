# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Dict, List, Optional
import numpy
from numpy import shape
from pathlib import Path
import torch
import time
import re
import hnswlib
import nltk
import joblib
import collections
from pathlib import Path
from nltk import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
from sentence_transformers import SentenceTransformer
from transformers import pipeline, Pipeline

from ConvAssist.predictor import Predictor
from ConvAssist.utilities.suggestion import Suggestion
from ConvAssist.utilities.nlp import NLP
from ConvAssist.predictor.utilities.prediction import Prediction

from ConvAssist.utilities.databaseutils.sqllite_dbconnector import SQLiteDatabaseConnector

class SentenceCompletionPredictor(Predictor):
    """
    Calculates prediction from n-gram model using gpt-2.
    """

    def set_seed(self, seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def __init__(self, 
                 config, 
                 context_tracker, 
                 predictor_name, 
                 short_desc=None, 
                 long_desc=None, 
                 logger=None):

        super().__init__( 
            config, 
            context_tracker, 
            predictor_name,
            short_desc, 
            long_desc,
            logger
        )

        self.context_tracker = context_tracker
        self.name = predictor_name
        # self.db = None

        # Sentence Completion Attributes - Defaults
        #[Database]
        self._class:str = "SentenceCompletionPredictor"

        #[PredictorRegistry]
        self._predictors: str = "SentenceCompletionPredictor"

        # [SentenceCompletionPredictor]
        self._predictor_class:str = "SentenceCompletionPredictor"
        self._learn:bool = True
        self._static_resources_path:str = ""
        self._personalized_resources_path:str = ""
        self._test_generalsentenceprediction:bool = False
        self._retrieveaac:bool = True
        self._sent_database:str = "sent_database.db"
        self._retrieve_database:str = "all_aac.txt"
        self._modelname:str = "aac_gpt2"
        self._tokenizer:str = "aac_gpt2"
        self._startsents:str = "startSentences.txt"
        self._embedding_cache_path:str = "all_aac_embeddings.pkl"
        self._sentence_transformer_model:str = "multi-qa-MiniLM-L6-cos-v1"
        self._index_path:str = "all_aac_semanticSearch.index"
        self._blacklist_file:str = "filter_words.txt"
        self._stopwords:str = "stopwords.txt"
        self._personalized_allowed_toxicwords_file:str = "personalized_allowed_toxicwords.txt"

        self._read_config()
        self.model_loaded = False
        self.corpus_sentences=[]
        self.generator: Pipeline
        
        self.nlp = NLP().get_nlp()
        
        ################ check if saved torch model exists  
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.n_gpu = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.n_gpu = torch.mps.device_count()
        else:
            self.device = torch.device("cpu")
            self.n_gpu = 0
            
        self.load_model(self.test_generalSentencePrediction, self.retrieve)            
        self.stemmer = PorterStemmer()
        
        ####### CREATE INDEX TO QUERY DATABASE
        self.embedder = SentenceTransformer(str(self._sentence_transformer_model))
        self.embedding_size = 384    #Size of embeddings
        self.top_k_hits = 2       #Output k hits
        self.n_clusters = 350
        
        #We use Inner Product (dot-product) as Index. 
        #We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        self.index = hnswlib.Index(space = 'cosine', dim = self.embedding_size)

        self.corpus_sentences = open(self.retrieve_database).readlines()
        self.corpus_sentences = [s.strip() for s in self.corpus_sentences]

        self.blacklist_words = open(self.blacklist_file).readlines()
        self.blacklist_words = [s.strip() for s in self.blacklist_words]

        self.personalized_allowed_toxicwords = self.read_personalized_toxic_words()

        self.OBJECT_DEPS = {"dobj","pobj", "dative", "attr", "oprd", "npadvmod", "amod","acomp","advmod"}
        self.SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
        
        # tags that define wether the word is wh-
        self.WH_WORDS = {"WP", "WP$", "WRB"}
        self.stopwords = []
        stoplist = open(self.stopwordsFile,"r").readlines()
        for s in stoplist:
            self.stopwords.append(s.strip())

        if(not Path.is_file(Path(self.embedding_cache_path))):
            self.corpus_embeddings = self.embedder.encode(self.corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
            joblib.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, self.embedding_cache_path)
            # np.save(self.embedding_cache_path,{'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings})

        else:
            # cache_data = np.load(self.embedding_cache_path)
            cache_data = joblib.load(self.embedding_cache_path)
            self.corpus_sentences = cache_data['sentences']
            self.corpus_embeddings = cache_data['embeddings']

        ###### LOAD INDEX IF EXISTS, ELSE CREATE INDEX 
        if Path.exists(Path(self.index_path)):
            self.logger.debug("Loading index...")
            self.index.load_index(str(self.index_path))

        else:
            ## creating the embeddings pkl file
            self.logger.debug(" index does not exist, creating index")

            ### Create the HNSWLIB index
            self.logger.debug("Start creating HNSWLIB index")
            self.index.init_index(max_elements = 20000, ef_construction = 400, M = 64)

            # Then we train the index to find a suitable clustering
            self.index.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))

            self.logger.debug(f"Saving index to: {self.index_path}")
            self.index.save_index(self.index_path)
            
        # Controlling the recall by setting ef:
        self.index.set_ef(50)  # ef should always be > top_k_hits

        if(not Path.is_file(Path(self.sent_database))) :
            self.logger.debug(f"{self.sent_database} not found, creating it")
            columns = ['sentence TEXT UNIQUE', 'count INTEGER']
            self.createTable(self.sent_database, "sentences", columns)

    @property
    def learn_mode(self):
        return self._learn

    @property
    def modelname(self):
        return self._modelname
    
    @property
    def personalized_resources_path(self):
        return self._personalized_resources_path

    @property
    def static_resources_path(self):
        return self._static_resources_path

    @property
    def test_generalSentencePrediction(self):
        return self._test_generalsentenceprediction
    
    @test_generalSentencePrediction.setter
    def test_generalSentencePrediction(self, value):
        self._test_generalsentenceprediction = value

    @property
    def sent_database(self):
        return os.path.join(self.personalized_resources_path, self._sent_database)
    
    @property
    def retrieve_database(self):
        return os.path.join(self.static_resources_path, self._retrieve_database)
    
    @property
    def blacklist_file(self):
        return os.path.join(self.static_resources_path, self._blacklist_file)
    
    @property
    def embedding_cache_path(self):
        return os.path.join(self.personalized_resources_path, self._embedding_cache_path)
    
    @property
    def index_path(self):
        return os.path.join(self.personalized_resources_path, self._index_path)

    @property
    def stopwordsFile(self):
        return os.path.join(self.static_resources_path, self._stopwords)
    
    @property
    def personalized_allowed_toxicwords_file(self):
        return os.path.join(self.personalized_resources_path, self._personalized_allowed_toxicwords_file)
    
    @property
    def startsents(self):
        return os.path.join(self.personalized_resources_path, self._startsents)
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def retrieve(self):
        return self._retrieveaac
    
    @retrieve.setter
    def retrieve(self, value):
        self._retrieveaac = value

    def read_personalized_toxic_words(self):
        path = Path(self.personalized_allowed_toxicwords_file)
        if not Path.exists(path):
            f = open(self.personalized_allowed_toxicwords_file, "w")
            f.close()
        self.personalized_allowed_toxicwords = open(self.personalized_allowed_toxicwords_file, "r").readlines()
        self.personalized_allowed_toxicwords = [s.strip() for s in self.personalized_allowed_toxicwords]
        self.logger.debug(f"UPDATED TOXIC WORDS = {self.personalized_allowed_toxicwords}")
        return self.personalized_allowed_toxicwords

    def extract_svo(self, sent):
        doc = self.nlp(sent)
        sub = []
        at = []
        ve = []
        imp_tokens = []
        for token in doc:
            # is this a verb?
            if token.pos_ == "VERB":
                ve.append(token.text)
                if(token.text.lower() not in self.stopwords and token.text.lower() not in imp_tokens):
                    imp_tokens.append(token.text.lower())
            # is this the object?
            if token.dep_ in self.OBJECT_DEPS or token.head.dep_ in self.OBJECT_DEPS:
                at.append(token.text)
                if(token.text.lower() not in self.stopwords and token.text.lower() not in imp_tokens):
                    imp_tokens.append(token.text.lower())
            # is this the subject?
            if token.dep_ in self.SUBJECT_DEPS or token.head.dep_ in self.SUBJECT_DEPS:
                sub.append(token.text)
                if(token.text.lower() not in self.stopwords and token.text.lower() not in imp_tokens):
                    imp_tokens.append(token.text.lower())
        return imp_tokens

    def load_model(self, test_generalSentencePrediction, retrieve): 
        self.logger.debug(f"SentenceCompletionPredictor LOAD MODEL: {str(os.path.exists(self.modelname))}")
        
        self.test_generalSentencePrediction = test_generalSentencePrediction
        self.retrieve = retrieve
        
        #### if we are only testing the models or only retrieving from the AAC dataset, no need to load the model
        if (self.test_generalSentencePrediction) or (not self.retrieve):
            if(os.path.exists(self.modelname)):
                self.logger.debug(f"Loading gpt2 model from {str(self.modelname)}")
                self.generator = pipeline('text-generation', model=self.modelname, tokenizer=self.tokenizer)
                self.model_loaded = True

        elif (self.retrieve):
                self.model_loaded = True

        self.logger.debug(f"SentenceCompletionPredictor MODEL loaded: {self.model_loaded}")

    def is_model_loaded(self):
        return self.model_loaded

    def ngram_to_string(self, ngram):
        "|".join(ngram)

    def filter_text(self, text):
        res = False
        words = []
        toxicwordsList = list(set(self.blacklist_words) - set(self.personalized_allowed_toxicwords))

        if(any(x in text.lower().split() for x in toxicwordsList)):
            words = list(set(text.lower().split()) & set(toxicwordsList))
            self.logger.warning("blacklisted word is present!!")
            res = True
        return (res, words)

    def textInCorpus(self, text):
        query_embedding = self.embedder.encode(text)
        
        #We use hnswlib knn_query method to find the top_k_hits
        corpus_ids, distances = self.index.knn_query(query_embedding, k=self.top_k_hits)
        hits = [{'corpus_id': id, 'score': 1-score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        self.logger.debug(f"score = {hits[0]['score']}, corpus_id = {hits[0]['corpus_id']}, len(corpus_sentences) = {len(self.corpus_sentences)}, corpus_embeddings.shape = {self.corpus_embeddings[0].shape}")
        self.logger.debug(f"text = {text}, score = {hits[0]['score']}, sentence = {self.corpus_sentences[hits[0]['corpus_id']]}")
        return hits[0]['score']
        
    def retrieve_fromDataset(self, context):
        pred = Prediction()
        probs = {}

        lines = open(self.retrieve_database, "r").readlines()
        retrieved = []
        totalsent= len(lines)
        for each in lines:
            # if(each.lower().find(context.lower())!=-1):
            if(each.lower().startswith(context.lower())):
                each = re.split('[.\n?!]',each)[0]
                retrieved.append(each)
        retrieved_set = set(retrieved)
        self.logger.debug(f"len(retrieved_set = {len(retrieved_set)}) len(retrieved) = {len(retrieved)}")
        for s in retrieved_set:
            probs[s] = float(retrieved.count(s))/totalsent
        try:
            pers_results: dict[str, float] = {}
            totalsentences = 0
            dbconn = SQLiteDatabaseConnector(self.sent_database, self.logger)
            dbconn.connect()
            count = 0
            #### CHECK IF SENTENCE EXISITS IN THE DATABASE
            res = dbconn.fetch_all("SELECT * FROM sentences")
            if res:
                for r in res:
                    sent = r[0]
                    totalsentences = totalsentences+r[1]
                    if(sent.lower().startswith(context.lower())):
                        sent = re.split('[.\n?!]',sent)[0]
                        if(sent in pers_results):
                            pers_results[sent] = pers_results[sent]+r[1]
                        else:
                            pers_results[sent] = r[1]
                for k,v in pers_results.items():
                    probs[k] = float(v/totalsentences)


            sorted_x = collections.OrderedDict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
            count = 0
            addedcompletions = []
            for k,v in sorted_x.items():
                if(count > 5):
                    break
                k = k[len(context):]
                if (not k in addedcompletions):
                    pred.add_suggestion(Suggestion(k, v, self.name))
                    addedcompletions.append(k)
                    count = count+1
        except Exception as e:
            self.logger.debug(f"Exception in SentenceCompletionPredictor, retrieveFromDataset function {e}")

        return pred

    def checkRepetition(self, text):
        tokens = nltk.word_tokenize(text)

        #Create bigrams and check if there are any repeititions. 
        bgs = nltk.bigrams(tokens)
        #compute frequency distribution for all the bigrams in the text
        fdist = nltk.FreqDist(bgs)
        fdist = {k:v for (k,v) in fdist.items() if v >= 2}
        if (fdist!={}):
            return True

        #Create trigrams and check if there are any repeititions. 
        bgs = nltk.trigrams(tokens)
        #compute frequency distribution for all the trigrams in the text
        fdist = nltk.FreqDist(bgs)
        fdist = {k:v for (k,v) in fdist.items() if v >= 2}
        if (fdist!={}):
            return True        
        return False

    def generate(self, context, num_gen, predi):
        try:
            start = time.perf_counter()
            result = self.generator(context, do_sample=False, max_new_tokens=20, num_return_sequences=10, num_beams = 10, num_beam_groups=10, diversity_penalty=1.5, repetition_penalty = 1.1) 

            if isinstance(result, List) and all(isinstance(item, dict) for item in result):
                generated_text:List[Dict[str, Any]] = result
            else:
                raise TypeError(f"Unexpected type for result: {type(result)}")
            
            inputContext = context
            allsent: list[str] = []
            counts: Dict[str, float] = {}
            totalsent = 0
            if(num_gen<5):
                num_gen = 5
            num_gen = 10
            inputContext = inputContext.replace("<bos> ","")
            contextList = sent_tokenize(inputContext)
            num_context_sent = len(contextList)
            self.logger.debug (f"num_context_sent = {num_context_sent}")
            for o in generated_text:
                self.logger.debug(f"Generated Text: {o['generated_text']}")
                gentext = o["generated_text"]
                newgen = re.split(r'<bos> |<eos> |bos|eos|<bos>|<eos>|<|>|\[|\]|\d',gentext)
                self.logger.debug(f"Full generated Text = {newgen[1]}")
                gen_text_sent = sent_tokenize(newgen[1])
                currSentence = gen_text_sent[num_context_sent-1]
                
                ### check for repetitive sentences
                if (self.checkRepetition(currSentence)):
                    self.logger.warning(f"Repetition in the sentence: {currSentence}")
                    continue

                reminderText = currSentence[len(contextList[-1]):]
                reminderTextForFilter = re.sub(r'[?,!.\n]', '', reminderText.strip())

                if(self.filter_text(reminderTextForFilter)[0]!=True):
                    reminderText = re.sub(r'[?!.\n]', '', reminderText.strip())
                    score = self.textInCorpus(currSentence.strip())

                    #TODO: DO WE THRESHOLD SCORES?
                    #TODO: DETOXIFY

                    if reminderText!='':
                        if(currSentence not in allsent):
                            imp_tokens = self.extract_svo(currSentence)
                            imp_tokens_reminder = []
                            #### get important tokens only of the generated completion
                            for imp in imp_tokens:
                                if imp in word_tokenize(reminderText):
                                    imp_tokens_reminder.append(imp)
                            present = False
                            for a in allsent:
                                for it in imp_tokens_reminder:
                                    if(self.stemmer.stem(it) in [self.stemmer.stem(w) for w in word_tokenize(a[len(contextList[-1]):])]):
                                        present = True
                                        break
                            if(present==False):
                                allsent.append(currSentence)
                                counts[reminderText] = 1*score
                                totalsent = totalsent + 1 
                        else:
                            counts[reminderText] = counts[reminderText]+1*score
                            totalsent = totalsent + 1 

            # toxic_filtered_sent = self.detoxify(allsent)
            self.logger.warning(f"toxic_filtered_sent not called.")

            for k, v in counts.items():
                counts[k] = float(v)/totalsent

            sorted_x = collections.OrderedDict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
            count = 0
            for k,v in sorted_x.items():
                
                if(count==num_gen):
                    break
                self.logger.debug(f"sentence = {k} score = {v}")
                predi.add_suggestion(Suggestion(k, v, self.name))
                count = count+1

            self.logger.debug(f"latency in generation = {time.perf_counter()-start}")
        except Exception as e:
            self.logger.debug(f"Exception in SentenceCompletionPredictor.{e}")

        return predi

    def predict(self, max_partial_prediction_size, filter):
        super().predict(max_partial_prediction_size, filter)
        
        sentence_prediction = Prediction()
        word_prediction = Prediction()   # not used in this predictor

        # tokens = [""] * self.cardinality

        context = self.context_tracker.context.lstrip()
        if(context == "" or context==" "):
            self.logger.debug(f"context is empty, loading top sentences from {self._startsents}")
            if (not Path.is_file(Path(self.startsents))):
                self.logger.error(f"{self.startsents} not found!!!")

            ##### retrieve top5 from startsentFile 
            data = open(self.startsents,"r").readlines()
            for k in data:
                sentence_prediction.add_suggestion(Suggestion(k.strip(), float(1/len(data)), self.name))
            return sentence_prediction, word_prediction
        start = time.perf_counter()
        self.logger.debug("context inside predictor predict = {context}")

        #### If we are testing generation models
        if (self.test_generalSentencePrediction):
            sentence_prediction = self.generate("<bos> "+context.strip(),5, sentence_prediction)

        #### if we want to Only retrieve from AAC dataset
        elif(self.retrieve):
            self.logger.debug("retireve is True - retrieving from database")
            sentence_prediction = self.retrieve_fromDataset(context)

        #### Hybrid retrieve mode  elif(self.retrieve=="hybrid"):
        elif(self.retrieve=="False"):
            self.logger.debug("Hybrid retrieval - AAC dataset + model generation")
            sentence_prediction = self.retrieve_fromDataset(context)
            self.logger.debug(f"retrieved {len(sentence_prediction)} sentences in {str(sentence_prediction)}")
            
            ##### ONLY IF THE GENERATION MODEL IS LOADED, GENERATE MODEL BASED PREDICTIONS
            if(len(sentence_prediction)<5 and self.model_loaded):
                self.logger.debug(f"generating {5-len(sentence_prediction)} more predictions")
                sentence_prediction = self.generate("<bos> "+context.strip(),5-len(sentence_prediction), sentence_prediction)

        latency = time.perf_counter() - start 
        self.logger.debug(f"latency = {latency}")
        self.logger.debug(f"prediction = {sentence_prediction}")
        
        return sentence_prediction, word_prediction


    def close_database(self):
        pass

    def learn(self, change_tokens):
        #### For the sentence completion predictor, learning adds the sentence to the database
        if self.learn_mode:
            change_tokens = change_tokens.strip()
            self.logger.debug(f"learning, {change_tokens}")
            #### add to sentence database
            try:
                dbconn = SQLiteDatabaseConnector(self.sent_database, self.logger) 
                dbconn.connect()
                count = 0
                #### CHECK IF SENTENCE EXISITS IN THE DATABASE
                res = dbconn.fetch_all("SELECT count FROM sentences WHERE sentence = ?", (change_tokens,))
                if res and len(res) > 0:
                    if len(res[0]) > 0:
                        count = int(res[0][0])

                ### IF SENTENCE DOES NOT EXIST, ADD INTO DATABASE WITH COUNT = 1
                if count==0:
                    self.logger.debug("count is 0, inserting into database")
                    dbconn.execute_query('''
                    INSERT INTO sentences (sentence, count)
                    VALUES (?,?)''', (change_tokens, 1))
                    ### update retrieval index: 
                    # self.index.load_index(self.index_path)
                    self.logger.debug("shape before: {} len*self.corpus_sentences = {}".format(self.corpus_embeddings[0].shape, len(self.corpus_sentences)))
                    
                    self.logger.debug("sentence  {} not present, adding to embeddings and creating new index".format(change_tokens))
                    phrase_emb = self.embedder.encode(change_tokens.strip())
                    phrase_id = len(self.corpus_embeddings)
                    self.corpus_embeddings = numpy.vstack((self.corpus_embeddings, phrase_emb))
                    self.corpus_sentences.append(change_tokens.strip())
                    # np.save(self.embedding_cache_path,{'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings})
                    joblib.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, self.embedding_cache_path)
                    # with open(self.embedding_cache_path, "wb") as fOut:
                    #     pickle.dump({'sentences': self.corpus_sentences, 'embeddings': self.corpus_embeddings}, fOut)
                    
                    # Then we train the index to find a suitable clustering
                    self.logger.debug("phrase_emb.shape = {} id= {}".format(str(phrase_emb[0].shape), str(len(self.corpus_embeddings))))
                    self.index.add_items(phrase_emb, phrase_id)

                    self.logger.debug("Saving index to:{}".format(self.index_path))
                    self.index.save_index(self.index_path)
                    self.logger.debug("shape after: {} len*self.corpus_sentences =  {}".format(str(self.corpus_embeddings.shape), str(len(self.corpus_sentences))))


                    #### DEALING WITH PERSONALIZED, ALLOWED TOXIC WORDS
                    #### if sentence to be learnt contains a toxic word, add the toxic word to the "allowed" word list
                    res, words = self.filter_text(change_tokens)
                    if(res==True):
                        for tox in words:
                            self.logger.debug(f"toxic words to be added to personalized db: {tox}")
                            if(tox not in self.personalized_allowed_toxicwords):
                                self.personalized_allowed_toxicwords.append(tox)
                                fout = open(self.personalized_allowed_toxicwords_file, "w")
                                for tox_word in self.personalized_allowed_toxicwords:
                                    fout.write(tox_word+"\n")
                                fout.close()

                ### ELSE, IF SENTENCE EXIST, ADD INTO DATABASE WITH UPDATED COUNT
                else:
                    self.logger.debug("sentence exists, updating count")
                    dbconn.execute_query('''
                    UPDATE sentences SET count = ? where sentence = ?''', (count+1, change_tokens))
                dbconn.commit()
            except Exception as e:
                self.logger.debug(f"Exception in SentenceCompletionPredictor learn  = {e}")
            finally:
                dbconn.close()

    def _read_config(self):
        try:
            self.logger.debug(f"Reading config for {self.name}")

            for attr, default in vars(self).items():
                if attr.startswith("_"):
                    for section in self.config.sections():
                        if attr[1:] in self.config.options(section):
                            new_value = default
                            try:
                                new_value = self.config.getboolean(section, attr[1:],fallback=default)
                            except ValueError:
                                try:
                                    new_value = self.config.getint(section, attr[1:],fallback=default)
                                except ValueError:
                                    new_value = self.config.get(section, attr[1:],fallback=default)

                            setattr(self, attr, new_value )
                            break
            
        except Exception as e:
            self.logger.error(f"Exception in SentenceCompletionPredictor._read_config = {e}")
            raise e
        
        # self.logger.debug(f"SENTENCE MODE CONFIGURATIONS")
        # #self.logger.debug(f"using Onnx model for inference {self.use_onnx_model}")
        # self.logger.debug(f"test_generalSentencePrediction {self.test_generalSentencePrediction}")
        # self.logger.debug(f"model = {self.modelname} tokenizer = {self.tokenizer}")