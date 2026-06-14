# Dickens Classifier

A text classification project that distinguishes between writing by Charles Dickens and ChatGPT-generated text that imitates his style. Built as a senior research project using NLTK and scikit-learn.

## Overview

This project explores whether NLP-based classifiers can reliably tell apart authentic Dickens prose from AI-generated imitations. Three classification approaches are implemented and compared:

- **PMI Bigrams** (`classifyPmi.py`) — Naive Bayes classifier using Pointwise Mutual Information to select the most distinctive bigrams
- **Chi-Squared Bigrams** (`classifyChi.py`) — Naive Bayes classifier using Chi-Squared scoring for bigram feature selection
- **Information Gain** (`informationGain.py`) — Decision Tree classifier using ASCII-based token features (token sum, first/last letter, word length)
- **Combined** (`classify.py`) — Runs PMI and Chi-Squared classifiers side by side for direct comparison

## Project Structure

```
dickens-classifier/
├── classify.py             # Main script, runs PMI and Chi-Squared together
├── classifyPmi.py          # Standalone PMI classifier
├── classifyChi.py          # Standalone Chi-Squared classifier
├── informationGain.py      # Standalone Decision Tree classifier
├── config.py               # File paths and sample data configuration
├── Dickens_portion.txt     # Dickens training corpus
├── chatGPT_portion.txt     # ChatGPT training corpus
├── combinedPartsOfSpeech.csv  # POS-tagged data for informationGain.py
└── sample-data/
    ├── dickens/            # Full Dickens texts for testing
    └── chatgpt/            # ChatGPT-generated texts for testing
```

## Requirements

- Python 3.x
- nltk
- scikit-learn
- pandas

Install dependencies with:

```bash
pip install nltk scikit-learn pandas
```

Some NLTK corpora are also required. These are downloaded automatically when running `classify.py`, but can also be installed manually:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

Run the main comparison script:
```bash
python classify.py
```

Or run individual classifiers:
```bash
python classifyPmi.py
python classifyChi.py
python informationGain.py
```

The output will show each classifier's predictions for both ChatGPT-generated and authentic Dickens test files, along with the most informative features.

## Data

Training data consists of:
- Excerpts from Dickens novels sourced from [Project Gutenberg](https://www.gutenberg.org/)
- ChatGPT-generated texts prompted to imitate Dickens' style (A Christmas Carol, A Tale of Two Cities, Oliver Twist, and others)

Test files are configured in `config.py`.

## Author

Senior Research Project 2023
Nicole Lane (Hardy)Senior Research Project — 2023
