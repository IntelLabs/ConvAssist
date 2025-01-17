# ConvAssist
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/ConvAssist/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/ConvAssist)
[![Unit Tests](https://github.com/IntelLabs/ConvAssist/actions/workflows/run_unittests.yaml/badge.svg?branch=covassist-cleanup)](https://github.com/IntelLabs/ConvAssist/actions/workflows/run_unittests.yaml)
[![pytorch](https://img.shields.io/badge/PyTorch-v2.4.1-green?logo=pytorch)](https://pytorch.org/get-started/locally/)
![GitHub License](https://img.shields.io/github/license/IntelLabs/ConvAssist)
![python-support](https://img.shields.io/badge/Python-3.12-3?logo=python)


ConvAssist is a library intended to enable and empower users with disabilities to communicate using the latest language modeling technologies, via Intel's Assistive Context Aware Toolkit, ACAT. ConvAssist can quickly suggest utterances/sentences that the user can use to have a social conversation or communicate with their caregiver in near real time. 

ConvAssist is built on [Pressagio](https://github.com/Poio-NLP/pressagio), that is a library that predicts text based on n-gram models. Pressagio is a pure Python port of the [presage library](https://presage.sourceforge.io) and is part of the [Poio project](https://www.poio.eu).  

ConvAssist contains language models based on specific Assistive and Augmentative Communication (AAC) datasets, and dialog datasets, tailored for day-to-day communication in our assistive usecase. These language models support both next word prediction and sentence completion for enabling near-real time communication with least amount of user effort and intervention. 

## Predictor Classes

### Sentence Completion Predictor
```SentenceCompletionPredictor``` is a class that provides sentence completion predictions using a combination of n-gram models and GPT-2.

### Spell Correct Predictor
```SpellCorrectPredictor``` is a class that extends the Predictor class to provide spell correction functionality. It uses a spell checker to generate suggestions for the last token in the context.

### Canned Phrases Predictor
```CannedPhrasesPredictor``` is a class that searches a database of canned phrases to find matching next words and sentences based on a given context. 

### Canned Word Predictor
```CannedWordPredictor``` is a specialized predictor that extends the SmoothedNgramPredictor.  It is designed to handle canned responses using natural language processing (NLP) techniques.

### Smoothed NGram Predictor
```SmoothedNgramPredictor``` is a class that extends the Predictor class to provide functionality for predicting the next word(s) in a sequence using smoothed n-grams.

### General Word Predictor
```GeneralWordPredictor``` is a class that extends SmoothedNgramPredictor to provide word predictions based on a precomputed set of most frequent starting words from an AAC dataset.

## Installation and use
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

## Installation and use
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

The `continuous_prediction.ini` file defines which predictors are enable.  See documentation for continuous_predict for details on how to use the configuration file.

## License
ConvAssist source code is distributed under the GPL-3.0 or later license.
