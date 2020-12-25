from models import Base
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class Logistic(Base):
    def __init__(self):
        super().__init__(model_name='Logistic Regression')
        self.params = {
            'solver':'liblinear'
        }
        # Initialize model
        self.lr = LogisticRegression(solver = self.params['solver'], random_state = 27)
        self.y_pred = None
    
    def fit(self, X_train, y_train):
        self.lr.fit(X_train, y_train)
    
    def predict(self, X_test):
        y_pred = self.lr.predict(X_test)
        self.y_pred = y_pred
        
        return y_pred
    
class Logistic_tuned(Base):
    def __init__(self):
        super().__init__(model_name='Logistic Regression (tuned)')
        self.params = [
            {
                'classifier' : [LogisticRegression()],
                'classifier__penalty' : ['l1', 'l2'],
                'classifier__C' : np.logspace(-4, 4, 20),
                'classifier__solver' : ['liblinear']
            }
        ]
        # Initialize model
        lr = Pipeline(
            [
                (
                    'classifier',
                    LogisticRegression(random_state = 27)
                )
            ]
        )
        self.clf = GridSearchCV(lr, self.params, cv = 5, scoring = ['recall','f1'], refit = 'f1', verbose = True, n_jobs = -1)
        self.y_pred = None
    
    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        print(sorted(self.clf.cv_results_.keys()))
        print(sorted(self.clf.cv_results_))
    
    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)
        self.y_pred = y_pred
        
        return y_pred