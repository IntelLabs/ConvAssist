import kagglehub
from tqdm import tqdm
import pandas as pd
import string
import re
import nltk
from nltk.tokenize import sent_tokenize

datasets = [
    {
        'NAME' : 'atharvjairath/personachat',
        'DATASET_FILENAME' : 'personality.csv',
        'OUTPUT_FILENAME' : './personachat/Persona_Chat_Data.txt',
        'DELIMITER' : ',',
        'COLS' : ['chat'],
    },
    {
        'NAME' : 'projjal1/human-conversation-training-data',
        'DATASET_FILENAME' : 'human_chat.txt',
        'OUTPUT_FILENAME' : './human-conversation/Human_Conversation_Data.txt',
        'DELIMITER' : ':',
        'COLS' : [1],
    }
]


def clean_sentence(sentence: str):
    # Skip sentences containing redacted terms
    if '<' in sentence and '>' in sentence:
        return None

    # Create a translation table that removes punctuation except apostrophes
    punct_to_remove = string.punctuation.replace("'", "")
    trans_table = str.maketrans("", "", punct_to_remove)
    cleaned = sentence.translate(trans_table)
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.replace(" ' ", " ")

    return f"{cleaned}\n"


def parse(text):
    sentences = []

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    pattern = r'\s([,.!?;:])'
    text = re.sub(pattern, r'\1', text)

    for sent in sent_tokenize(text):
        sent = clean_sentence(sent)
        if sent:
            sentences.extend(sent)
        else:
            continue

    return sentences


for dataset in datasets:
    # Download latest version
    path = kagglehub.dataset_download(dataset['NAME'])

    df = pd.read_csv(
        f"{path}/{dataset['DATASET_FILENAME']}",
        delimiter=dataset['DELIMITER'],
        encoding='ascii',
        encoding_errors='ignore',
        usecols=dataset['COLS']
    )

    with open(dataset['OUTPUT_FILENAME'], 'w') as fout:
        for row in tqdm(df.itertuples(index=False), desc="Processing sentences"):
            sentences = parse(row[0])
            previous_row = row[0]
            if sentences:
                fout.writelines(sentences)
            else:
                continue
