from models import Base
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class RandomForest(Base):
    def __init__(self):
        super().__init__(model_name='Random Forest Classifier')
        self.params = {
            'max_features' : 6
        }
        # Initialize model
        self.lr = RandomForestClassifier(
            max_features = self.params['max_features'],
            random_state=27,
            verbose=1     
        )
        self.y_pred = None
    
    def fit(self, X_train, y_train):
        self.lr.fit(X_train, y_train)
    
    def predict(self, X_test):
        y_pred = self.lr.predict(X_test)
        self.y_pred = y_pred
        
        return y_pred
    
class RandomForest_tuned(Base):
    def __init__(self):
        super().__init__(model_name='Random Forest Classifier (tuned)')
        self.params = [
            {'classifier' : [RandomForestClassifier()],
             'classifier__n_estimators' : list(range(10,101,10)),
             'classifier__max_features' : list(range(6,32,5))}
        ]
        # Initialize model
        rf = Pipeline(
            [
                (
                    'classifier',
                    RandomForestClassifier(random_state = 27)
                )
            ]
        )
        self.clf = GridSearchCV(rf, self.params, cv = 5, scoring = ['recall','f1'], refit = 'f1', verbose = True, n_jobs = -1)
        self.y_pred = None
    
    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        print(sorted(self.clf.cv_results_.keys()))
        print(sorted(self.clf.cv_results_))
    
    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)
        self.y_pred = y_pred
        
        return y_pred