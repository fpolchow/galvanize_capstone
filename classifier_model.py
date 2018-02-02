

class TextAnalysis:
    def __init__(self,classifier,n_words,method = 'count', n_kfolds ,tokenizer):
        self.classifier = classifier
        self.vocabulary = None
        self.vectorizer = None
        self.trained_model = None
        self.n_words = n_words
        self.n_kfolds = n_kfolds
        self.method = method
        self.tokenizer = tokenizer
        self.word_list = []



    def make_vectorizer(self,X,ngram_range,vocabulary=None):
        if self.method == 'count':
            vectorizer = CountVectorizer(tokenizer=self.tokenizer,
                                         stop_words='english',
                                         ngram_range=ngram_range,
                                         max_features=max_features,
                                         vocabulary=vocabulary)
            vectorizer.fit(X)

        elif self.method == 'tf_idf':
            vectorizer = TfidfVectorizer(tokenizer = self.tokenizer,
                                         ngram_range=ngram_range,
                                         max_features=max_features,
                                         vocabulary=vocabulary)
            vectorizer.fit(X)
        return vectorizer


    def get_vocabulary(X_train,y_train,ngram,n_words):

        vect = self.make_vectorizor(X_train, method=self.method,ngram_range=ngram)
        X_train = vect.transform(X_train)
        rf = RandomForestClassifier()
        rf.fit(X_train,y_train)
        feature_importance = rf.feature_importances_
        dict = {v:k for k,v in vect.vocabulary_.iteritems()}
        self.word_list.extend(vocab[np.argsort(feature_importance)][:n_words])




    def train_predictions(self,X,y,ngram = (1,2)):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=5)
        predictions = np.zeros(len(X))
        for train_index, test_index in kf.split(X):
            vect = make_vectorizer(X[train], method=self.method, ngram_range=ngram)
            X_train_pred, X_test_pred = vect.transform(X[train]),vect.transform(X[test])
            self.classifier.fit(X_train_pred, y[train_index])
            predicted_values = self.classifier.predict_proba(X_test_pred)
            predictions[test_index] = predicted_values
        return predictions




    def get_vectorizer(self,X):
        self.vectorizer = self.make_vectorizer(X,ngram_range=(1,2),vocabulary=self.word_list)


    def fit(self,X_train,y_train):

        X_train = self.vectorizer.fit_transform(X_train)
        self.trained_model = self.classifier(X_train,y_train)

    def test_predictions(self,X_test):
        X_test = self.vectorizer.transform(X_test)
        self.trained_model.predict(X_test)
