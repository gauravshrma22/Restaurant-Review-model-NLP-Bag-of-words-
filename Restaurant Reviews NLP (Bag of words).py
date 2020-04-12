#Natural Language Processing...
#NLP model for Classifying Positive and Negative Reviews..

#Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the datasets
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t' , quoting = 3)

#Clearing the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range (0,1000):
  review = re.sub('[^a-zA-z]', '  ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = '  '.join(review)
  corpus.append(review)

#Creating the Bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Splitting the Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier  = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting the Test Set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
