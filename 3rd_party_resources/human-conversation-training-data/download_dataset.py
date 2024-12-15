import kagglehub

def clean_sentence(sentence):
    # Skip sentences containing redacted terms
    if '<' in sentence and '>' in sentence:
        return None
        
    # Remove punctuation except apostrophes
    punctuation = '.,!?;:¦"""()-'
    for p in punctuation:
        sentence = sentence.replace(p, '')
    
    # Remove extra spaces and convert to lowercase
    return ' '.join(sentence.split()).lower()

def parse_conversation(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_sentences = []
    
    for line in lines:
        if line.startswith('Human'):
            _, message = line.split(':', 1)
            clean_sentence_result = clean_sentence(message)
            if clean_sentence_result:  # Only add non-empty and non-redacted sentences
                clean_sentences.append(clean_sentence_result)
    
    return clean_sentences

# Download latest version
path = kagglehub.dataset_download("projjal1/human-conversation-training-data")

with open(f"{path}/human_chat.txt") as f:
    content = f.read()

    #strip any emoji's
    content = content.encode('ascii', 'ignore').decode('ascii')

    sentences = parse_conversation(content)
    print(sentences[1:5])

with open('processed_Human_Conversation_dataset.txt', 'w') as fout:
    for sentence in sentences:
        fout.write(sentence + '\n')
