# Author: Nicole Hardy
# Last Edited: 23 Oct 2023
# The program will implement bigrams as its feature selection method
import itertools
import re
import nltk
import random
import config
from nltk.tokenize import TweetTokenizer
# Need this specific tokenizer because the default breaks on apostrophes
from nltk.corpus import stopwords
from nltk.collocations import *
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# create bigram measures, lemmatizer, and tokenizer
bigram_measures = nltk.collocations.BigramAssocMeasures()
lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

dickensFile = "Dickens_portion.txt"
chatFile = "chatGPT_portion.txt"

# Converts a file of text into tokens
def convertToTokens(fileName):
    # Read the entire file into a string, replacing new lines with a space
    with(open(fileName, encoding='utf8')) as file:
        fileText = file.read().replace("\n", ' ')

    # Replace any non-alphanumeric characters with a space
    fileText = re.sub(r'[^a-zA-Z0-9_\s’]', ' ', fileText)
    # Replace 2 or more spaces with one space
    fileText = re.sub(r'\s{2,}', ' ', fileText)
    # Replace any orphan single quotes with a space
    fileText = re.sub(r'\s’\s', ' ', fileText)
    # Replace backticks to single quotes
    fileText = re.sub(r'’', '\'', fileText)

    tokens = tokenizer.tokenize(fileText)

    return tokens

# Tokenize text
dickensTokens = convertToTokens(dickensFile)
chatTokens = convertToTokens(chatFile)

stop_words = set(stopwords.words("english"))

# remove stop words from dickens corpus
no_stop_words_dickens = []
for word in dickensTokens:
    if word not in stop_words:
        no_stop_words_dickens.append(lem.lemmatize(word))

# remove stopwords from chatgpt corpus
no_stop_words_chat = []
for word in chatTokens:
    if word not in stop_words:
        no_stop_words_chat.append(lem.lemmatize(word))

# Create bigrams, pull out those that occur more than three times, and sort the top 100 scores by their frequency.
dickensFeatures = BigramCollocationFinder.from_words(no_stop_words_dickens)
chatFeatures = BigramCollocationFinder.from_words(no_stop_words_chat)

# Filter out bigrams containing appearing less than 3 times
dickensFeatures.apply_freq_filter(3)
chatFeatures.apply_freq_filter(3)

# Of the bigrams that appear the most, select the first 150 using specified measures
dickensNBBigrams = dickensFeatures.nbest(bigram_measures.pmi, 150)
chatNBBigrams = chatFeatures.nbest(bigram_measures.pmi, 150)
dickensChiBigrams = dickensFeatures.nbest(bigram_measures.chi_sq, 150)
chatChiBigrams = chatFeatures.nbest(bigram_measures.chi_sq, 150)

# Tags
dickens = 'Dickens'
chat = 'ChatGPT'

#https://stackoverflow.com/questions/20827741/nltk-naivebayesclassifier-training-for-sentiment-analysis
# featuresets for training:
# List of tuples that define what the tuple is, whether or not it appears, and the label
# For example, the bigram ('bread', 'butter') appear in a text written by Dickens, so the tuple is created as:
# (({"('bread', 'butter')": True}, "Dickens"))
nbTrainingData = []
chiTrainingData = []
#List of bigrams from both datasets. These will be compared to the input data
nbTrainingBigrams = []
chiTrainingBigrams = []

for bigram in dickensNBBigrams:
    nbTrainingBigrams.append(bigram)
    nbTrainingData.append(tuple(({str(bigram): True}, dickens)))

for bigram in chatNBBigrams:
    nbTrainingBigrams.append(bigram)
    nbTrainingData.append(tuple(({str(bigram): True}, chat)))

for bigram in dickensChiBigrams:
    chiTrainingBigrams.append(bigram)
    chiTrainingData.append(tuple(({str(bigram): True}, dickens)))

for bigram in chatChiBigrams:
    chiTrainingBigrams.append(bigram)
    chiTrainingData.append(tuple(({str(bigram): True}, chat)))

# Create classifier
nbClassifier = nltk.NaiveBayesClassifier.train(nbTrainingData)
chiClassifier = nltk.NaiveBayesClassifier.train(chiTrainingData)

print("Naive Bayes")
nbClassifier.show_most_informative_features(10)
print("Chi Squared")
chiClassifier.show_most_informative_features(10)

# "nb" = Naive Bayes Classifier
# "chi" = Chi Squared

def classifyText(classifierType, inputFile):
    # Create bigrams of the input text, similar to how the test data was made
    inputTokens = convertToTokens(inputFile)

    no_stop_words_input = []
    for word in inputTokens:
        if word not in stop_words:
            no_stop_words_input.append(lem.lemmatize(word))

    inputFeatures = BigramCollocationFinder.from_words(no_stop_words_input)

    # Depending on the measure, determine the bigram and classifier
    inputBigrams = None
    trainingBigrams = None
    classifier = None
    if classifierType == "nb":
        inputBigrams = inputFeatures.nbest(bigram_measures.pmi, 150)
        trainingBigrams = nbTrainingBigrams
        classifier = nbClassifier
    elif classifierType == "chi":
        inputBigrams = inputFeatures.nbest(bigram_measures.chi_sq, 150)
        trainingBigrams = chiTrainingBigrams
        classifier = chiClassifier
    else:
        inputBigrams = inputFeatures.nbest(bigram_measures.pmi, 150)
        trainingBigrams = nbTrainingBigrams
        classifier = nbClassifier

    inputData = {}

    for bigram in inputBigrams:
        if bigram in trainingBigrams:
            inputData[str(bigram)] = True
        else:
            inputData[str(bigram)] = False

    result = classifier.classify(inputData)
    return result

chatFiles = config.files["chatgpt"]
dickensFiles = config.files["dickens"]

print("NB      | Chi     | File")
print("--------------------------------------------------")
for file in chatFiles:
    nbResult = classifyText("nb", config.chatDir+file["name"])
    chiResult = classifyText("chi", config.chatDir+file["name"])
    print(nbResult+" | "+chiResult+"  |"+file["name"])
print("")
print("NB      | Chi      | File")
print("--------------------------------------------------")
for file in dickensFiles:
    nbResult = classifyText("nb", config.dickensDir+file["name"])
    chiResult = classifyText("chi", config.dickensDir+file["name"])
    print(nbResult+" | "+chiResult+"  | "+file["name"])
