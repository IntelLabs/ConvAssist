from .predictor import Predictor
from .utilities import PredictorResponse
from .utilities.models import Predictions, Suggestion
import transformers
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import hnswlib
import numpy as np
import joblib
from pathlib import Path
from convassist.utilities.databaseutils.sqllite_dbconnector import (
    SQLiteDatabaseConnector,
)

#### This predictor is designed to do multiple things: 
# 1) After ASR recieves visitors voice input (and the user has not typed anything), this predictor will be called to generate : 
    ## a) A set of possible responses to the visitor's voice input + conversation history
    ## b) A set of keywords that the user could select from to continue the conversation
    ## c) A set of "Next words" that the user could select from to continue the conversation

# 2) After ASR recieves visitors voice input and the user has started to type something, this predictor will be called to generate :
    ## a) A set of possible response "continuations" to the user-input given conversation history
    ## b) A set of word-completions that the user could select from to continue the conversation

## keyword generation takes conversation history as input and generates keywords
## word generation takes conversation history + typed text as input and generates next words
## sentence generation takes conversation history as input and generates full sentences OR
## takes conversation history + typed text as input and generates sentence completions

### For sentence generation, we will use a combination of dialog retrieval and
###  pre-trained language model (LLM) to generate the responses.

### INitialization: 
### We will initialize a SentenceTransformer model for generating embeddings for dialog context.
### We will initialize a SQLite database to store dialog context-response pairs.
### We will initialize an HNSWLIB index for efficient similarity search of dialog context embeddings.

### Configuration:
### 1. Load the LLM model and tokenizer
### 2. Load the SentenceTransformer model for dialog context embeddings
### 3. Initialize the dialog context database
### 4. Initialize the HNSWLIB index

### Prediction:
### 1. Retrieve relevant responses from the database using HNSWLIB for efficient similarity search
###         a) Generate embedding for the query context
###         b) Use HNSWLIB for efficient similarity search
###         c) Sort by score
### 2. If we don't have enough predictions, generate more using LLMs

### Learning:
### 1. Store the context-response pair in the database
### 2. Update the HNSWLIB index with the new context-response pair

### Dialog Context Management:
### 1. Maintain a conversation history of the last 5 turns
### 2. Generate embeddings for the conversation history
### 3. Use the embeddings to retrieve relevant responses from the database
### 4. Use the embeddings to update the HNSWLIB index


class ContextAwarePredictor(Predictor):
    @property
    def modelname(self):
        return self._modelname

    @modelname.setter
    def modelname(self, value):
        self._modelname = value

    @property
    def generatorPipeline(self):
        return self._generatorPipeline
    
    @generatorPipeline.setter
    def generatorPipeline(self, value):
        self._generatorPipeline = value

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    @property
    def llmLoaded(self):
        return self._llmLoaded
    
    @llmLoaded.setter
    def llmLoaded(self, value):
        self._llmLoaded = value

    @property
    def prompt(self):
        return self._prompt
    
    @prompt.setter
    def prompt(self, value):
        self._prompt = value

    @property
    def predictiontype(self):
        return self._predictiontype
    
    @predictiontype.setter
    def predictiontype(self, value):
        self._predictiontype = value

    def __init__(self, *args, **kwargs):
        self._prompt: str = ""
        self._modelname: str = ""
        self._device: str = ""
        self._n_gpu: int = 0
        self._generatorPipeline: transformers.pipelines.Pipeline = None
        self._tokenizer: AutoTokenizer = None
        self._llmLoaded: bool = False
        self._predictiontype: str = ""


        # For dialog context management
        self._embedder = None
        self._dialogs_db = None
        self._conversation_history = []
        self._max_history_turns = 5  # Store last 5 turns for context
        self._dialog_embedding_dim = 384  # Default embedding dimension


        super().__init__(*args, **kwargs)
        
    def configure(self) -> None:

        try:
            self.generatorPipeline = transformers.pipeline("text-generation", model=self.modelname, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

            self.llmLoaded = True
            self.logger.info(f"Model {self.modelname} loaded successfully")

            # Configure embedding model for dialog context
            self.embedder = SentenceTransformer(
                self.sentence_transformer_model if hasattr(self, 'sentence_transformer_model') else "all-MiniLM-L6-v2",
                device=self._device
            )
            self._dialog_embedding_dim = self.embedder.get_sentence_embedding_dimension()
            self.top_k_hits = getattr(self, 'top_k_hits', 5)


            # Initialize dialog database
            from convassist.utilities.databaseutils.sqllite_dbconnector import SQLiteDatabaseConnector
            self.dialogs_db_path = getattr(self, 'dialogs_db_path', 'dialogs.db')
            self.dialogs_db = SQLiteDatabaseConnector(self.dialogs_db_path)
            
            # Create dialogs table if it doesn't exist
            self.dialogs_db.connect()
            self.dialogs_db.execute_query('''
                CREATE TABLE IF NOT EXISTS dialogs (
                context_text TEXT,
                response_text TEXT,
                context_embedding BLOB,
                count INTEGER DEFAULT 1,
                PRIMARY KEY (context_text, response_text)
                )
            ''')
            self.dialogs_db.close()
            self.logger.info("Dialog context database initialized")

            # Initialize HNSWLIB index
            self.index_path = getattr(self, 'dialog_index_path', 'dialog_index.bin')
            self.embedding_cache_path = getattr(
                self, 'dialog_embedding_cache_path', 'dialog_embeddings.pkl'
            )
            
            # Initialize collections
            self._dialog_contexts = []
            self._context_embeddings = np.empty((0, self.embedding_size))
            
            # We use cosine similarity for the index
            self.index = hnswlib.Index(space="cosine", dim=self.embedding_size)
            
            # Load existing embeddings if available
            if Path.is_file(Path(self.embedding_cache_path)):
                cache_data = joblib.load(self.embedding_cache_path)
                self._dialog_contexts = cache_data.get("dialog_contexts", [])
                self._context_embeddings = cache_data.get("embeddings", np.empty((0, self.embedding_size)))
                
                # Load existing index if available, else create a new one
                if Path.exists(Path(self.index_path)):
                    self.logger.debug("Loading existing dialog index...")
                    self.index.load_index(str(self.index_path))
                else:
                    self.logger.debug("Creating new index from existing embeddings...")
                    self._create_new_index()
            else:
                # If no embeddings file, initialize from database
                self._initialize_from_database()
            
            # Set search parameters
            self.index.set_ef(50)  # ef should always be > top_k_hits
            
            self.logger.info("ContextAwarePredictor configured successfully")
        except Exception as e:
            self.logger.error(f"Error configuring ContextAwarePredictor: {e}")

    def _initialize_from_database(self):
        """Initialize the index and embeddings from the database"""
        try:
            dbconn = SQLiteDatabaseConnector(self.dialogs_db_path)
            dbconn.connect()
            all_dialogs = dbconn.fetch_all(
                "SELECT context_text, response_text, context_embedding FROM dialogs"
            )
            dbconn.close()
            
            if not all_dialogs:
                # Create empty index
                self._create_new_index()
                return
                
            # Process all dialogs from database
            self._dialog_contexts = []
            embeddings_list = []
            
            for context_text, response_text, embedding_bytes in all_dialogs:
                self._dialog_contexts.append({
                    "context": context_text,
                    "response": response_text
                })
                
                # Convert blob to numpy array
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                embeddings_list.append(embedding)
            
            # Convert list to numpy array if we have embeddings
            if embeddings_list:
                self._context_embeddings = np.vstack(embeddings_list)
            
            # Create and save index
            self._create_new_index()
            
        except Exception as e:
            self.logger.error(f"Error initializing from database: {e}")
            # Create empty index
            self._create_new_index()

    def _create_new_index(self):
        """Create a new HNSWLIB index from existing embeddings"""
        try:
            max_elements = max(20000, len(self._dialog_contexts) * 2)  # Allow room for growth
            
            self.index.init_index(
                max_elements=max_elements,
                ef_construction=400,
                M=64
            )
            
            if len(self._context_embeddings) > 0:
                self.index.add_items(
                    self._context_embeddings, 
                    list(range(len(self._context_embeddings)))
                )
                
            self.index.save_index(self.index_path)
            joblib.dump(
                {
                    "dialog_contexts": self._dialog_contexts, 
                    "embeddings": self._context_embeddings
                },
                self.embedding_cache_path
            )
            
            self.logger.debug(f"Created new index with {len(self._dialog_contexts)} dialogs")
        except Exception as e:
            self.logger.error(f"Error creating new index: {e}")


















        except Exception as e:
            self.logger.error(f"Error loading model {self.modelname}: {e}")

    def predict(self, max_partial_prediction_size=None, filter=None) -> PredictorResponse:
        responses:PredictorResponse = PredictorResponse()

        match self.predictiontype:
            case "sentences":
                predictions = responses.sentencePredictions
            case "words":
                predictions = responses.wordPredictions
            case "keywords":
                predictions = responses.keywordPredictions
            case _:
                raise ValueError(f"Invalid prediction type: {self.predictiontype}")

        if not self.llmLoaded:
            self.logger.error("Model not loaded")
            return responses
        
        context = self.context_tracker.context.lstrip()
        #### TODO: Process the context to get the conversation history and user written text
        if self.predictiontype == "sentences":
            # Get responses from retrieval methods
            matched_responses = self._retrieve_dialog_responses(
                context, 
                max_results=max_partial_prediction_size
            )
            
            # Add matched responses to predictions
            for response, score in matched_responses:
                predictions.add_suggestion(
                    Suggestion(response, score, f"{self.predictor_name}")
                )
            
            # If we don't have enough predictions , generate more using LLMs
            if len(predictions.suggestions) < max_partial_prediction_size:
                remaining = max_partial_prediction_size - len(predictions.suggestions)
                
                predictions = self._generate(self.prompt, context, predictions)

        else:
            predictions = self._generate(self.prompt, context, predictions)

        return responses
    
    def _generate(self, prompt, context: str, predictions: Predictions) -> Predictions:
        prompt = self.prompt.replace("{context}", context)

        self.logger.debug(f"Prompt: {prompt}")
        self.logger.debug(f"Context: {context}")

        try:
            outputs = self.generatorPipeline(
                prompt,
                do_sample=False,
                num_return_sequences=5,
                num_beams=5,
                num_beam_groups=5,
                diversity_penalty=1.5,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=100,
                return_full_text=False,
            )
 

        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return []
                
        for index, generated_text in enumerate(outputs):
            #TODO: Figure out how best to put the generated text into the response
            
            sentence_text = generated_text['generated_text']
            predictions.add_suggestion(Suggestion(sentence_text, 1/len(outputs), self.predictor_name))

        return predictions

    def _retrieve_dialog_responses(self, dialog_context, max_results=5):
        """Retrieve relevant responses using HNSWLIB for efficient similarity search"""
        if not dialog_context or len(self._dialog_contexts) == 0:
            return []
            
        try:
            # Generate embedding for the query context
            query_embedding = self.embedder.encode(dialog_context)
            
            # Use HNSWLIB for efficient similarity search
            context_ids, distances = self.index.knn_query(
                query_embedding, 
                k=min(max_results, len(self._dialog_contexts))
            )
            
            # Format results
            results = []
            for idx, distance in zip(context_ids[0], distances[0]):
                response = self._dialog_contexts[idx]["response"]
                similarity_score = 1 - distance  # Convert distance to similarity
                
                # Get count information from database to boost frequently used responses
                dbconn = SQLiteDatabaseConnector(self.dialogs_db_path)
                dbconn.connect()
                res = dbconn.fetch_all(
                    "SELECT count FROM dialogs WHERE context_text = ? AND response_text = ?",
                    (self._dialog_contexts[idx]["context"], response)
                )
                dbconn.close()
                
                count = 1
                if res and len(res) > 0 and len(res[0]) > 0:
                    count = int(res[0][0])
                
                # Boost score based on frequency
                boosted_score = similarity_score * (1 + np.log(count) * 0.1)
                
                results.append((response, boosted_score))
                
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving dialog responses: {e}")
            return []

    ## TODO: Need to figure out how learn is called. Typically the user written text is sent
    ## But here we need to send the conversation history + user written text
    def learn(self, context: str, response: str) -> None:
        try:

            ### We need a response to store into the database. 
            ### Empty context is okay, in case User starts a new conversation. 
            if not response:
                self.logger.warning("Missing response for learning")
                return

            # Generate embedding for the context
            context_embedding = self.embedder.encode(context)
            context_embedding_bytes = context_embedding.tobytes()
            
            # Store in database
            self.dialogs_db.connect()

            # Check if this context-response pair already exists
            existing = self.dialogs_db.fetch_all(
                "SELECT count FROM dialogs WHERE context_text = ? AND response_text = ?", 
                (context, response)
            )

            if existing and len(existing) > 0:
                # Update count for existing dialog
                current_count = existing[0][0]
                self.dialogs_db.execute_query(
                    "UPDATE dialogs SET count = ?, updated_at = CURRENT_TIMESTAMP WHERE context_text = ? AND response_text = ?",
                    (current_count + 1, context, response)
                )
                self.logger.debug(f"Updated existing dialog, new count: {current_count + 1}")
            else:
                # Insert new dialog
                self.dialogs_db.execute_query(
                    """
                    INSERT INTO dialogs 
                    (context_text, response_text, context_embedding, count) 
                    VALUES (?, ?, ?, ?)
                    """,
                    (context, response, context_embedding_bytes, 1)
                )
                self.logger.debug(f"Inserted new dialog context-response pair")
                
            self.dialogs_db.close()
        
        except Exception as e:



            self.logger.error(f"Error learning dialog: {e}")


