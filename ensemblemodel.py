import cleaning_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV




class RedditEnsembleModel:
    def __init__(self,classifier_model,model):
        self.model = model
        self.tf_idf_vectorized = None
        self.classifier_model = classifier_model
        self.model = None

    # def find_predictions(self,):
    #     predictions = self.classifier_model.train_predictions(X_train,y_train,ngram = ngram)
    #     X_train['prediction'] = predictions

    def fit(self, X_train,y_train,ngram = (1,2)):
        X_train['prediction'] = self.classifier_model.train_predictions
        pipeline = Pipeline(steps=[('scale',StandardScaler()),('svm',SVR())])
        self.model = pipeline.fit(X_train,y_train)




    def predict(self,X_test):
        X_test['text_classifier'] = self.classifier_model.predict(X_test)
        return self.model.predict(X_test)

