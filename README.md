# ConvAssist
# Predictive Sentence Completion and Response Generation for Assistive Context-Aware Toolkit

ConvAssist is a library intended to enable and empower users with disabilities to communicate using the latest language modeling technologies, via Intel's Assistive Context Aware Toolkit, ACAT. ConvAssist can quickly suggest utterances/sentences that the user can use to have a social conversation or communicate with their caregiver in near real time. 

ConvAssist is built on [Pressagio](https://github.com/Poio-NLP/pressagio), that is a library that predicts text based on n-gram models. Pressagio is a pure Python port of the [presage library](https://presage.sourceforge.io) and is part of the [Poio project](https://www.poio.eu).  

ConvAssist contains language models based on specific Assistive and Augmentative Communication (AAC) datasets, and dialog datasets, tailored for day-to-day communication in our assistive usecase. These language models support both next word prediction and sentence completion for enabling near-real time communication with least amount of user effort and intervention. 

ConvAssist enables different modes of interactions for users through the ACAT interface. The three modes are 
1) Sentence Mode
2) Canned Phrases Mode and 
3) Short-Hand Mode 

In the Sentence Mode, users can use the interface to type any sentence they would like to. As they type a letter, word completions appear along with their probabilities. After a word is chosen or typed, sentence completions also appear with 5 possible completions of the context typed so far. The predictions come from an AI model finetuned for the AAC use case. Please find the associated [model card here](https://github.com/IntelLabs/ConvAssist/blob/main/ACAT_ConvAssist_Interface/resources/static_resources/aac_gpt2/GPT2ForAAC_model_card.md). 

The Canned Phrases Mode can be used to store and search for frequently used phrases for easy and quick access. 

Short-Hand Mode can be used by users for quick communication - in situations where the user needs to communicate with their caretaker with minimum number of words (without the need for grammatical sentences). 

## How to run 
ConvAssist can be used with ACAT for the best user-experience. But ConvAssist can also be used as a standalone application for research and development. To install, run and use ConvAssist, follow the below steps. 

### Install
ConvAssist works with Python 3.12.x or greater. Create a virtual environment and install the required packages.  We provide a pyproject.toml file to assist, and use Poetry for package management.

#### Create a Virtual Environment with Poetry

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/IntelLabs/ConvAssist.git
    cd ConvAssist
    ```

2. **Install Poetry**:
    If you don't have Poetry installed, you can install it by following the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

3. **Install Dependencies and Create Virtual Environment**:
    ```sh
    poetry install
    ```

    This command will create a virtual environment, install the dependencies specified in the `pyproject.toml` file, and set up the environment for you.

4. **Activate the Virtual Environment**:
    ```sh
    poetry shell
    ```

    This command activates the virtual environment created by Poetry.

### Run
We provide several demonstration apps to showcase how to use ConvAssist

#### DEMO - Continuous Predict

``` sh
cd interfaces/Demos/continuous_predict
python pyprompter.py
```

The `continuous_prediction.ini` file defines which predictors are enable.  See documentation 
for details on how to use the configuration file.

## License
ConvAssist source code is distributed under the GPL-3.0 or later license.
