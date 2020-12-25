# from typing import final
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

class Base:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.scores = {}
        self.confusion_matrix = None
    # @final
    def score(self, y_test, method = 'f1'): #y_pred, 
        if method == 'f1':
            score = f1_score(y_test, self.y_pred)
            self.scores['F1'] = score
        elif method == 'accuracy':
            score = accuracy_score(y_test, self.y_pred)
            self.scores['Accuracy'] = score
        elif method == 'recall':
            score = recall_score(y_test, self.y_pred)
            self.scores['Recall'] = score
        elif method == 'confusion_matrix':
            self.confusion_matrix = pd.DataFrame(confusion_matrix(y_test, self.y_pred))
            print(self.confusion_matrix)
        elif method == 'all':
            
            self.scores['F1'] = f1_score(y_test, self.y_pred)
            self.scores['Accuracy'] = accuracy_score(y_test, self.y_pred)
            self.scores['Recall'] = recall_score(y_test, self.y_pred)
            self.confusion_matrix = pd.DataFrame(confusion_matrix(y_test, self.y_pred))
            print('Scores of model "{}":'.format(self.model_name))
            print(json.dumps(self.scores, sort_keys = True, indent = 4))
            print('Confusion matrix of model "{}":'.format(self.model_name))
            print(self.confusion_matrix)
            print()
        else:
            print('Cannot compute score "{}" for model "{}".'.format(method, self.model_name))
            
        