 # -*- coding: utf-8 -*-

#python version: Python 3.6.1
# genral
import pandas as pd 
import numpy as np 
import sys
import os


# nlp
import re   #for regex
import nltk
import spacy
import gensim
import string
from string import punctuation
from nltk.corpus import stopwords
#from nltk import pos_tag
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# feature engineering
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_union



def clean_text(comment_text,remove_stopwords=False,stem_words=False):
    comment_list = []
    for text in comment_text:
        # lowercase
        text = text.lower()
        # remove non-alphabet and numbers
        text = re.sub(r"[^A-Za-z0-9(),!?@&$\'\`\"\_\n]", " ", text)
        text = re.sub(r"\n", " ", text)
        
        # deal with apostrophe
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        
        # special symbol
        text = text.replace('&', ' and')
        text = text.replace('@', ' at')
        text = text.replace('$', ' dollar')

        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        
        comment_list.append(text)
    return comment_list

train = pd.read_csv('input/train.csv').fillna(' ')
test = pd.read_csv('input/test.csv').fillna(' ')

train["clean_comment_text"] = clean_text(train['comment_text'])
test["clean_comment_text"] = clean_text(test['comment_text'])

train_text = train["clean_comment_text"]
test_text = test["clean_comment_text"]

all_text = pd.concat([train_text,test_text])

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=30000)
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=30000)
vectorizer = make_union(word_vectorizer, char_vectorizer)

vectorizer.fit(all_text)
train_features = vectorizer.transform(train_text)
test_features = vectorizer.transform(test_text)

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})
class_names=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(solver='sag')

    cv_score = np.mean(cross_val_score(
        classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('submission.csv', index=False)