import pandas as pd
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


def tokenize(X):
    transformed_X = []
    for document in X:
        tokenized_doc = []
        for sent in nltk.sent_tokenize(document):
            tokenized_doc += nltk.word_tokenize(sent)
        transformed_X.append(tokenized_doc)
    return transformed_X


def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    train_tokens = tokenize(train['comment_text'])
    test_tokens = tokenize(test['comment_text'])

    tfidf = TfidfVectorizer(analyzer='word',
                            tokenizer=lambda x: x,
                            preprocessor=lambda x: x,
                            token_pattern=None)
    tfidf.fit(train_tokens)

    l = []
    for _l in train_tokens:
        l += _l
    counts = Counter(l)

    dt = {'vocabulary': tfidf.vocabulary_,
          'idf': tfidf.idf_.tolist(),
          'train_tokens': train_tokens,
          'test_tokens': test_tokens,
          'counter': counts,
          }
    with open("data/data.pkl", "wb") as f:
        pickle.dump(dt, f)


def load():
    return pickle.load(open("data/data.pkl","rb"))


if __name__ == '__main__':
    main()
