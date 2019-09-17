
# 6.9 Weighting Word Importance
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
print(tfidf.vocabulary_)