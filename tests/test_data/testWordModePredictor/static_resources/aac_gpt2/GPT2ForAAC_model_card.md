
# Model Card for GPT2ForAAC

<!-- Provide a quick summary of what the model is/does. [Optional] -->
This model is developed as a part of the Assistive Context Aware Toolkit (ACAT) whose goal is to build a system that allows users with disabilities to communicate. As a part of this, we have built a system that can quickly suggest utterances/sentences that the user can use to have a social conversation or communicate with their caregiver in near real time. In other words, given a context from the user, the model generates possible sentence completions for the user to chose from. 

# Model Details

<!-- Provide a longer summary of what this model is/does. -->
This model is developed as a part of the Assistive Context Aware Toolkit (ACAT) whose goal is to build a system that allows users with disabilities to communicate. As a part of this, we have built a system that can quickly suggest utterances/sentences that the user can use to have a social conversation or communicate with their caregiver in near real time. In other words, given a context from the user, the model generates possible sentence completions for the user to chose from. 

- **Developed by:** Intel Corporation 
- **Model type:** Language model for text generation
- **Language(s) supported (NLP):** English
- **License:** Apache-2.0
- **Parent Model:** OpenAI GPT2
- **Input** Models input text only.
- **Output** Models generate text only.
- **Status** This is a static model trained on an offline dataset. Future versions of the tuned models will be released as we improve model performance and safety with community feedback.
- **Model Architecture** GPT2ForAAC is an auto-regressive language model that uses the GPT2 transformer architecture, specifically OpenAI's GPT2 (small) with 12-layers, 768-hidden, 12-heads, 117M parameters.. The model has been finetuned on Alternative and Augmentative Communications datasets (more details below). 



# Uses

## Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->

The model is a part of ACAT's ConvAssist package, that has been built to enable social conversations for users with speech or motor neuron disabilities such as ALS. It supports capabilities such as word prediction, sentence prediction and semantic search for quick communication. The model specifically will be used by users for its sentence suggestion capabilities. 

The model could also be further fine-tuned by developers to enhance the sentence completion performance. 




# Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

This model is based on OpenAI's GPT2 model that has been specifically fine-tuned on an Alternative and Augmentative Communication (AAC) dataset, to support ACAT’s assistive usage. ​The AI base model, GPT2, was pretrained with vast amounts of internet data possibly containing biased/toxic and factually inaccurate text. While we have taken great efforts to remove harmful or hurtful generation in our fine-tuned model, it is possible that our system's responses at times could be biased, toxic, inaccurate or contextually inappropriate. ​


## Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->


In ACAT’s usage, the users retain full control of whether to choose from the AI generated content or not and also have the ability to edit and confirm all responses to ensure they reflect their intended message. AI-generated content should not be used or marked as an official record until a human review of the content has been conducted to ensure accuracy. It is users’ responsibility to edit for tone was well as facts. ​​Users are responsible for any decisions made or actions taken on the basis of genAI content. 


# Training Details

## Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The training data is a set of AAC-like communication utterances crowdsourced using Amazon Mechanical Turk. 
(Keith Vertanen and Per Ola Kristensson. The Imagination of Crowds: Conversational AAC Language Modeling using Crowdsourcing and Large Data Sources. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP). ACL: 700-711, 2011.)


# Evaluation

We incorporate this model in the ACAT framework with additional post-processing to make it more suitable for our use-case. Specifically we also add relevance ranking module that re-ranks the model's generation based on the AAC usecase. We also have a toxicity filtering module that filters out toxic generations (Please refer to the image below). 

![image](https://github.com/intel-sandbox/ConvAssist/assets/89480559/d93ac63a-1e3d-454b-920f-54c7f0bfd631)


## Model Evaluation on Toxicity
Evaluations of the overall backend is done using Keystroke Savings Rate (KSR), that measures how many keystrokes are required to type a sentence that a user intends to. In this model card, we only present model-related evaluations. We evaluate model for the toxicity levels - based on our toxicity filtering and relevance ranking module. We use Perspective API to measure the different toxicity attributes such as identity attack, profanity, etc.. Below table presents the results. 


We evalute the base model GPT2, GPT2 finetuned on the AAC dataset, and 

||Toxicity| Identity Attack | Profanity | Obscene|Average Toxicity|
|---|---|---|---|---| ---|
||Perspective API|Perspective API|Perspective API| Perspective API| HuggingFace Evaluate Library|
|GPT2 Base Model| 0.378|0.0634|0.2992| 0.3901|0.253|
|GPT2 Finetuned on AAC​|0.2438​ | 0.0322​|0.1811​|0.2342​|0.1666|
|GPT2 Finetuned on AAC + Relevance Ranking + Toxicity Filtering​|**0.1364**​|**0.0173**​|**0.0818​**|**0.1186**|​**0.0696**|

## Measuring Bias using Honest score 
The HONEST score aims to measure hurtful sentence completions in language models. It aims to quantify how often sentences are completed with a hurtful word, and if there is a difference between groups (e.g. genders, sexual orientations, etc.).
We compare the base GPT2 model and our fine-tuned model and observe that the gap between positive and negative regard scores between male and female gender is reduced in our fine-tuned model. 
||Male <br> Positive| Female <br> Positive| Male <br> Negative| Female<br> Negative|
|---|---|---|---|---|
|GPT2 Base Model| 0.42|0.39|0.15| 0.166|
|GPT2 Finetuned on AAC​|**0.50** | **0.515​**|**0.102​**|**0.101​**|

# Forseeable Misuses

Uses that require high levels of factuality, high stakes situations, mental health or medical applications, generating misinformation or disinformation, impersonating others, facilitating or inciting harassment or violence, any use that could lead to the violation of a human right under the UN Declaration of Human Rights.  





