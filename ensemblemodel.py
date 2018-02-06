# import cleaning_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV





class RedditEnsembleModel:
    """ A model that will be used to create the final predictions"""
    def __init__(self,model,text_analysis_model=None):
        self.model = model
        self.tf_idf_vectorized = None
        self.text_analysis_model = text_analysis_model

    # def find_predictions(self,):
    #     predictions = self.classifier_model.train_predictions(X_train,y_train,ngram = ngram)
    #     X_train['prediction'] = predictions

    def fit(self, X_train,y_train):
        if self.text_analysis_model:
            X_train['prediction'] = self.text_analysis_model.train_predictions
            self.model = self.model.fit(X_train,y_train)
        else:
            self.model = self.model.fit(X_train,y_train)




    def predict(self,X_test):
        text = self.text_analysis_model.vectorizer.transform(X_test['cleaned'].values.astype('U'))
        X_test['text_predictions'] = self.text_analysis_model.test_predictions(text)
        X_test_with_predictions = X_test.drop(labels='cleaned',axis=1)
        return self.model.predict(X_test_with_predictions)

