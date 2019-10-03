# Chapter 6 - Handling Text
# Adapted from the book: Machine Learning with Python Cookbook by Chris Albon
# 6.0 Introduction
"""
Unstructured text data, like the contents of a book or a tweet, is both one of the most interesting sources of 
features and one of the most complex to handle. In this chapter, we will cover strategies for transforming text 
into information-rich features. This is not to say that the recipes covered here are comprehensive. There exist 
entire academic disciplines focused on handling this and similar data, and the contents of all their techniques 
would fill a small library. Despite this, there are some commonly used techniques, and a knowledge of these will
add valuable tools to our preprocessing toolbox.
"""

# 6.1 Cleaning Text
# When you have some unstructured text data and want to complete some basic cleaning.
# Most basic text cleaning operations should only replace Python's core string operations, in particular 'strip', 
# 'replace', and 'split':
# Create text
text_data = ["    Interrobang. By Aishwarya Henriete    ",
             "Parking And Going. By Karl Gautier",
             "   Today Is The night. By Jarek Prakash   "]

# Strip whitespace
strip_whitespace = [string.strip() for string in text_data]

# Show text
strip_whitespace

# Remove periods
remove_periods = [string.replace(".", "") for string in strip_whitespace]

# Show text again
remove_periods

# We can also create and apply a custom transformation function:
# Create function
def capitalizer(string: str) -> str:
    return string.upper()

# Apply function
capitalize_string = [capitalizer(string) for string in remove_periods]

capitalize_string

# Finally, we can use regular expressions to make powerful string operations:
# Import library
import re

# Create function
def replace_letters_with_X(string: str) -> str:
    return re.sub(r"[a-zA-Z]", "X", string)

# Apply function
Xs = [replace_letters_with_X(string) for string in remove_periods]

Xs

# Most text data will need to be cleaned before we can use it to build features. Most basic text cleaning can be
# completed using Python's standard string operations. In the real world we will most likely define a custom 
# cleaning function (e.g., 'capitalizer') combining some cleaning task and apply that to the text data.

# 6.2 Parsing and Cleaning HTML
# When you have text data with HTML elements and want to extract just the text.
# Use Beautiful Soup's extensive set of options to parse and extract from HTML:
# Load library
from bs4 import BeautifulSoup

# Create some HTML code
html = """
       <div class='full_name'><span style='font-weight:bold'>
       Masego</span> Azra</div>"
       """

# Parse html
soup = BeautifulSoup(html, "lxml")

# Find the div with the class "full_name", show text
print(soup.find("div", {"class" : "full_name"}).text)

# Despite the strange name, Beautiful Soup is a powerful Python library designed for scraping HTML. Typically 
# Beautiful Soup is used to scrape live websites, but we can just as easily use it to extract text data embedded in
# HTML. The full range of Beautiful Soup operations is beyond the scope of this book, but even the few methods used 
# in this solution show how easily we can parse HTML code to extract the data we want.

# 6.3 Removing Punctuation
# When you have a feature of text data and want to remove punctustion.
# Define a function that uses 'translate' with a dictionary of punctuation characters:
# Load libraries
import unicodedata
import sys

# Create text
text_data = ['Hi"""" I. Love. This, Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']

# Create a dictionary of punctuation characters
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))

# For each string, remove any punctuation characters
print([string.translate(punctuation) for string in text_data])

# 'translate' is a Python method popular due to its blazing speed. In our solution, first we created a dictionary, 
# 'punctuation', with all punctuation characters according to Unicode as its key and 'None' as its values. Next we 
# translate all characters in the string that are in 'punctuation' into 'None', effectively removing them. There are
# more readable ways to remove punctuation, but this somewhat hacky solution has the advantage of being far faster 
# than alternatives.

# It is important to be conscious of the fact that punctuation contains information (e.g., "Right?" versus "Right!").
# Removing punctuation is often a necessary evil to create features; however, if the punctuation is important we 
# should make sure to take that into account.

# 6.4 Tokenizing Text
# When you have text and want to break it up into individual words.
# Natural Language Toolkit for Python (NLTK) has a powerful set of text manipulation operations, including word 
# tokenizing.
# Load library
from nltk.tokenize import word_tokenize

# Create text
string = "The science of today is the technology of tomorrow"

# Tokenize words
word_tokenize(string)

# We can also tokenize into sentences
# Load library
from nltk.tokenize import sent_tokenize

# Create text
string = "The science of today is the technology of tomorrow. Tomorrow is today."

# Tokenize sentences
sent_tokenize(string)

# Tokenization, especially word tokenization, is a common task after cleaning text data because it is the forst 
# step in the process of turning the text into data we will use to construct useful features.

# 6.5 Removing Stop Words
# When you're given tokenized text data, you want to remove extremely common words (e.g., a, is, of, on) that 
# contain little informationalvalue.
# Use NLTK's 'stopwords':
# Load library
from nltk.corpus import stopwords

# You will have to download the set of stop words the first time
# import nltk
# nltk.download('stopwords')

# Create word tokens
tokenized_words = ["i",
                   "am",
                   "going",
                   "to",
                   "go",
                   "to",
                   "the",
                   "store",
                   "and",
                   "park",]

# Load stop words
stop_words = stopwords.words('english')

# Remove stop words
[word for word in tokenized_words if word not in stop_words]

# While "stop words" can refer to any set of words we want to remove before processing, frequently the term refers 
# to extremely common words that themselves contain little information value. NLTK has a list of common stop words 
# that we can use to find and remove stop words in our tokenized words:
# Show stop words
stop_words[:5]
# Note: NLTK's 'stopwords' assumes the tokenized words are all lowercased.

# 6.6 Stemming Words
# When you have tokenized words and want to convert them into their root forms.
# Use NLTK's 'PorterStemmer':
# Load library
from nltk.stem.porter import PorterStemmer

# Create word tokens
tokenized_words = ["i", "am", "humbled", "by", "this", "traditional", "meeting"]

# Create stemmer
porter = PorterStemmer()

# Apply stemmer
[porter.stem(word) for word in tokenized_words]

# Stemming reduces a word to its stem by identifying and removing affixes (e.g., gerunds) while keeping the root 
# meaning of the word. For example, both "tradition" and traditional have "tradit" as their stem, indicating that 
# while they are different words they represent the same general concept. By stemming our text data, we transform
# it to something less readable, but closer to its base meaning and thus more suitable for comparison across 
# observations. NLTK's 'PorterStemmer' implements the widely used Porter stemming algorithm to remove or replace
# common suffixes to produce the word stem.

# 6.7 Tagging Parts of Speech
# When you have text data and want to tag each word or character with its part of speech.
# Using NLTK's pre-trained parts-of-speech tagger:
# Load libraries
from nltk import pos_tag
from nltk import word_tokenize

# Create text
text_data = "Chris loved outdoor running"

# Use pre-trained part of speech tagger
text_tagged = pos_tag(word_tokenize(text_data))

# Show parts of speech
text_tagged

# The output is a list of tuples with the word and the tag of the part of speech. NLTK uses the Penn Treebank parts 
# for speech tags. Some examples of the Penn Treebank tags are:
# - NNP = Proper noun, singular
# - NN  = Noun, singular or mass
# - RB  = Adverb
# - VBD = Verb, past tense
# - VBG = Verb, gerund or present participle
# - JJ  = Adjective
# - PRP = Personal pronoun

# Once the text has been tagged, we can use the tags to find certain parts of speech. For example, here are all nouns:
# Filter words
[word for word, tag in text_tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]

# A more realistic situation would be that we have data where every observation cantains a tweet and we want to 
# convert those sentences into features for individual parts of speech (e.g., a feature with 1 if a proper noun
# is present, and 0 otherwise):
# Import libraries
from sklearn.preprocessing import MultiLabelBinarizer

# Create text
tweets = ["I am eating a burrito for breakfast",
          "Political science is an amazing field",
          "San Francisco is an awesome city"]

# Create list
tagged_tweets = []

# Tag each word and each tweet
for tweet in tweets:
    tweet_tag = pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])

# Use one-hot encoding to convert the tags into features
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)

# Using 'classes_' we can see that each feature is a part-of-speech tag:
# Show feature names
one_hot_multi.classes_

# If our text is in English and not on a specialized topic (e.g., medicine) the simplest solution is to use NLTK's 
# pre-trained parts-of-speech tagger. However, if pos_tag is not very accurate, NLTK also gives the ability to train
# our own tagger. The major downside of training a tagger is that we need a large corpus (a large and structured set 
# of texts) of text where the tag of each word is known. Constructing this tagged corpus is obviously labor intensive
# and is probably going to be a last resort.

# All that said, if we had a tagged corpus and wanted to train a tagger, the following is an example of how we could 
# do it. The corpus we are using is the Brown Corpus, one of the most popular sources of tagged text. Here we use a 
# backoff n-gram tagger, where n is the number of previous words we take into account when predicting a word's 
# part-of-speech tag. First we take into account the previous two words using 'TrigramTagger'; if two words are not 
# present, we "back off" and take into account the tag of the previous word using 'BigramTagger', and finally if that
# fails we only look at the word itself using 'UnigramTagger'. To examine the accuracy of our tagger, we split our 
# text into two parts, train our tagger on one part, and test how well it predicts the tags of the second part:
# Load library
from nltk.corpus import brown
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

# Get some text from the Brown Corpus, broken into sentences
sentences = brown.tagged_sents(categories = 'news')

# Split into 4000 sentences for training and 623 for testing
train = sentences[:4000]
test = sentences[4000:]

# Create backoff tagger
unigram = UnigramTagger(train)
bigram = BigramTagger(train, backoff = unigram)
trigram = TrigramTagger(train, backoff = bigram)

# Show accuracy
trigram.evaluate(test)

# 6.8 Encoding Text as a Bag of Words (https://en.wikipedia.org/wiki/N-gram & http://bit.ly/2HRba5v)
# When you have text data and want to create a set of features indicating the number of times an observation's text
# contains a particular word.
# Use scikit-learn's 'CountVectorizer':

# Load libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# Create the bag of words features matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Show feature matrix
bag_of_words

# This output is a sparse array, which is often necessary when we have a large amount of text. However, in our toy
# example we can use 'toarray' to view a matrix of word counts for each observation:
bag_of_words.toarray()

# We can use the 'vocabulary_' method to view the word associated with each feature:
# Show feature names
count.get_feature_names()

# One of the most common methods of transforming text into features is by using a bag-of-words model. Bag-of-words
# models output a feature for every unique word in text data, with each feature containing a count of occurances in
# observations. For example, in our solution the sentence 'I love Brazil. Brazil!' has a value of 2 in the "brazil"
# feature because the word brazil appears two lines.

# The text data in our solution was purposely small. In the real world, a single observation of text data could be 
# the contents of an entire book! Since our bag-of-words model creates a feature for every unique word in the data,
# the resulting matrix can contain thousands of features. This means that the size of the matrix can sometimes become
# very large in memory. However, luckily we can expliot a common characteristicof bag-of-words feature matrices to
# reduce the amount of data we need to store.

# Most words likely do not occur in most observations, and therefore bag-of-words feature matrices will contain 
# mostly 0s as values. We call these types of matrices "sparse". Instead of storing all values of the matrix, we can
# only store nonezero values and then assume all other values are 0. This will save us memory when we have large
# feature matrices. One of the nice features of CountVectorizer is that the output is a sparse matrox by default.

# 'CoutVectorizer' comes with a number of useful parameters to make creating bag-of-words feature matrices easy. 
# First, while by default every feature is a word, that does not have to be the case. Instead we cam set every 
# feature to be the combination of two words (called a 2-gram) or even three words (3-gram). 'ngram_range' sets the
# minimum and maximum size of our n-grams. For example, (2, 3) will return all 2-grams and 3-grams. Second, we can 
# easily remove low-information filler words using stop_words either with a built-in list or a custom list. Finally, 
# we can restrict the words or phrases we want to consider to a certain list of words using 'vocabulary'. For example,
# we could create a bag-of-words feature matrix for only occurrences of country names:

# Create feature matrix with arguments
count_2gram = CountVectorizer(ngram_range = (1, 2),
                              stop_words = "english",
                              vocabulary = ['brazil'])

bag = count_2gram.fit_transform(text_data)

# View feature matrix
bag.toarray()

# View the 1-grams and 2-grams
count_2gram.vocabulary_

# 6.9 Weighting Word Importance (http://bit.ly/2HT2wmW)
# When you want a bag of words, but with words weighted by their importance to an observation.
# Compare the frequency of the word in a document (a tweet, movie review, speed transcript, etc.) with the frequency
# of the word in all other documents using term frequency-inverse document frequency (tf-idf). scikit-learn makes
# this easy with TfidVectorizer:

# Load libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# Create the tf-idf feature matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

# Show tf-idf feature matrix
feature_matrix

# Just as in recipe 6.8 the output is a spare matrix. However, if we want to view the output as a dense matrix, we 
# can use .toarray:

# Show tf-idf feature matrix as dense matrix
feature_matrix.toarray()

# 'vocabulary_' shows us the word of each feature:
# Show feature names
tfidf.vocabulary_

# Discussion
# The more a word appears in a document, the more likely it is important to that document. For example, if the word 
# economy appears frequently, it is evidence that the document might be about economics. We call this term 
# frequency (tf).

# In contrast, if a word appears in many documents, it is likely less important to any individual document. For 
# example, if every document in some text data contains the word after then it is probably an unimportant word. We
# call this document frequency (df).

# By combining these two statistics, we can assign a score to every word representing how important that word is in 
# a document. Specifically, we multiply tf to the inverse of the document frequency (idf):

#            tf-idf(t, d) = tf(t, d) X idf(t)

# where t is a word and d is a document. There are a number of variations in how tf and idf are calculated. In 
# scikit-learn, tf is simply the number of times a word appears in the document and idf calculated as:

#            idf(t) = log((1 + n(sub'd')/(1 + df(d, t)))) + 1

# where n(sub'd') is the number of documents and df(d, t) is term, t's document frequency (i.e., number of documents
# where the term appears).

# By default, scikit-learn then normalizes the tf-idf vectors using the Euclidean norm (L2 norm). The higher the 
# resulting value, the more important the word is to a document.