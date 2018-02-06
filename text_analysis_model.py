from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
from gensim import corpora
# import Stemmer
# english_stemmer = Stemmer.Stemmer('en')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer - super(TfidfVectorizer,self).build_analyzer()
        return lambda doc: english_stemmer.stemWords(analyzer(doc))
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer,self).build_analyzer()
        return lambda doc: english_stemmer.stemWords

class TextAnalysis:
    """A class that holds the model that performs the primary text analysis. Predicted probability.This
    model has the ability to fit a classifier."""
    def __init__(self,classifier,tokenizer = None,method = 'count',):
        self.classifier = classifier
        self.vocabulary = None
        self.vectorizer = None
        self.trained_model = None
        self.train_predictions = None
        # self.n_kfolds = n_kfolds
        self.method = method
        self.tokenizer = tokenizer
        self.word_list = []



    def make_vectorizer(self,X,ngram_range,max_features=20000,vocabulary=None):
        if self.method == 'count':
            vectorizer = CountVectorizer(tokenizer=self.tokenizer,
                                         stop_words='english',
                                         ngram_range=ngram_range,
                                         vocabulary=vocabulary,
                                         max_features=max_features)
            vectorizer.fit(X)

        elif self.method == 'tf_idf':
            vectorizer = TfidfVectorizer(tokenizer = self.tokenizer,
                                         ngram_range= ngram_range,
                                         stop_words= 'english',
                                         vocabulary=vocabulary,
                                         max_features=max_features)
            vectorizer.fit(X)
        elif self.method == 'hash':
            vectorizer = HashingVectorizer(tokenizer = self.tokenizer,
                                         ngram_range= ngram_range,
                                         stop_words= 'english')
        return vectorizer


    def get_vocabulary(self,X_train,y_train,ngram,n_words):
        """Uses Random Forests to determine the feature importance for words. This is not currently being used in
        the model, but it might be adapted to be used"""
        vect = self.make_vectorizer(X_train,ngram_range=ngram)
        X_train = vect.transform(X_train)
        X_train = X_train[:, None]
        rf = RandomForestClassifier()
        rf.fit(X_train,y_train)
        feature_importance = rf.feature_importances_
        dict = {v:k for k,v in vect.vocabulary_.iteritems()}
        self.word_list.extend(vocab[np.argsort(feature_importance)][:n_words])




    def make_training_predictions(self,X,y,n_kfolds,ngram = (1,1),max_features=20000):
        kf = KFold(n_splits=n_kfolds, shuffle=True, random_state=5)
        predictions = np.zeros(len(X))
        i=0
        for train_index, test_index in kf.split(X):
            print('this is split # {} out of {}'.format(i,n_kfolds))
            vect = self.make_vectorizer(X[train_index].values.astype('U'), ngram_range=ngram,max_features=max_features)
            X_train_pred = vect.transform(X[train_index].values.astype('U'))
            X_test_pred = vect.transform(X[test_index].values.astype('U'))
            self.classifier.fit(X_train_pred, y.iloc[train_index])
            predicted_values = self.classifier.predict_proba(X_test_pred)[:,]
            predictions[[test_index]] = predicted_values
            i += 1
        self.train_predictions = predictions




    def get_vectorizer(self,X,max_features=20000):
        self.vectorizer = self.make_vectorizer(X,ngram_range=(1,1),vocabulary=None,max_features=max_features)


    def fit(self,X_train,y_train):

        X_train = self.vectorizer.fit_transform(X_train.values.astype('U'))
        self.trained_model = self.classifier.fit(X_train,y_train)

    def test_predictions(self,X_test):
        X_test = self.vectorizer.transform(X_test)
        predictions = self.trained_model.predict(X_test)[:,0]
        return predictions

