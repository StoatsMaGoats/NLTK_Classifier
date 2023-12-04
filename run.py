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
# Tag the bigrams
dickensFeatures = BigramCollocationFinder.from_words(no_stop_words_dickens)
chatFeatures = BigramCollocationFinder.from_words(no_stop_words_chat)

dickensFeatures.apply_freq_filter(3)
chatFeatures.apply_freq_filter(3)

dickensBigrams = dickensFeatures.nbest(bigram_measures.pmi, 150)
chatBigrams = chatFeatures.nbest(bigram_measures.pmi, 150)
dickensChiFeatures = dickensFeatures.nbest(bigram_measures.chi_sq, 150)
chatChiFeatures = chatFeatures.nbest(bigram_measures.chi_sq,150)

dickens = 'Dickens'
chat = 'ChatGPT'

trainingData = []
trainingChiData = []
trainingBigrams = []

for bigram in dickensBigrams:
    trainingBigrams.append(bigram)
    trainingData.append(tuple(({str(bigram): True}, dickens)))
    trainingChiData.append(tuple(({str(bigram): True}, dickens)))

for bigram in chatBigrams:
    trainingBigrams.append(bigram)
    trainingData.append(tuple(({str(bigram): True}, chat)))
    trainingChiData.append(tuple(({str(bigram): True}, chat)))

classifier = nltk.NaiveBayesClassifier.train(trainingData)
chiClassifier = nltk.NaiveBayesClassifier.train(trainingChiData)

def classifyText(classifier, inputFile):

    inputTokens = convertToTokens(inputFile)

    no_stop_words_input = []
    for word in inputTokens:
        if word not in stop_words:
            no_stop_words_input.append(lem.lemmatize(word))

    inputFeatures = BigramCollocationFinder.from_words(no_stop_words_input)

    inputBigrams = inputFeatures.nbest(bigram_measures.pmi, 150)

    #https://stackoverflow.com/questions/20827741/nltk-naivebayesclassifier-training-for-sentiment-analysis
    inputData = {}

    for bigram in inputBigrams:
        if bigram in trainingBigrams:
            #print(str(bigram) + " is in training data")
            inputData[str(bigram)] = True
        else:
            inputData[str(bigram)] = False

    #classifier.show_most_informative_features(10)
    result = classifier.classify(inputData)
    return result

chatFiles = config.files["chatgpt"]
dickensFiles = config.files["dickens"]

print("NB      | Chi     | File")
print("--------------------------------------------------")
for file in chatFiles:
    nbResult = classifyText(classifier, config.chatDir+file["name"])
    chiResult = classifyText(chiClassifier, config.chatDir+file["name"])
    print(nbResult+" | "+chiResult+"  |"+file["name"])
print("")
print("NB      | Chi      | File")
print("--------------------------------------------------")
for file in dickensFiles:
    nbResult = classifyText(classifier, config.dickensDir+file["name"])
    chiResult = classifyText(chiClassifier, config.dickensDir+file["name"])
    print(nbResult+" | "+chiResult+"  | "+file["name"])
