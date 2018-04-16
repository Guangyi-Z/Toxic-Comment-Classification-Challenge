import pickle
import gensim
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import time
import preprocess
from itertools import product


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, vocab, idf):
        self.word2vec = word2vec
        self.word2weight = None
        self.vocab = vocab
        self.idf = idf
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.vectors[0])

    def fit(self, X, y):
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(self.idf)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, self.idf[i]) for w, i in self.vocab.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


def grid_search_train(train, test, subm):
    '''
    Arguments:
    sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, 
    sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5,
    cbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, 
    sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=()
    '''
    data = preprocess.load()
    for sg,size,window,min_count,hs,neg,iter,sample in product( [1,0],
                                                                [100,300],
                                                                [5,10],
                                                                [1],
                                                                [0,1],
                                                                [5,10],
                                                                [5,25],
                                                                [0.1,0.01,0.001]):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), sg,size,window,min_count,hs,neg,iter,sample)
        
        model = gensim.models.word2vec.Word2Vec(data['train_tokens'], 
                                                sg=sg, size=size, window=window, min_count=min_count,
                                                hs=hs, negative=neg, iter=iter, sample=sample)
        model_name = 'sg{0}-sz{1}-win{2}-minc{3}-hs{4}-neg{5}-iter{6}-samp{7}'.format(sg,
                                                                      size, window, min_count,
                                                                      hs, neg, iter, sample)
        model.save('data/w2v-' + model_name + '.model')

        embedding_vectorizer = TfidfEmbeddingVectorizer(model.wv, data['vocabulary'], data['idf'])
        train_embedded = embedding_vectorizer.fit(data['train_tokens'], None)
        train_embedded = embedding_vectorizer.transform(data['train_tokens'])
        test_embedded = embedding_vectorizer.transform(data['test_tokens'])

        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        preds = np.zeros((len(test), len(label_cols)))
        for i, j in enumerate(label_cols):
            m = LogisticRegression()
            mf = m.fit(train_embedded, train[j])
            preds[:,i] = mf.predict_proba(test_embedded)[:,1]

        submid = pd.DataFrame({'id': subm["id"]})
        submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
        submission.to_csv('submission/submission-toxicw2v-doctfidf-lr-{}.csv'.format(model_name), index=False)
    

if '__main__' == __name__:
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    subm = pd.read_csv('data/sample_submission.csv')
    
    grid_search_train(train, test, subm)
    