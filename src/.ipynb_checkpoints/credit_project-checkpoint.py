from predict_model import *
from train_model import *
import pandas_profiling
import numpy as np

class main:
    def __init__(self):
        self = self
        
    def read_from_csv(self, path):
        return pd.read_csv(path, index_col = 0)
    
    def trainAdaBoostPipeline(self, X_train, y_train, optimize = True):
        '''
        Build and train a pipeline that
        1. Imputes all missing X value with 0
        2. Scale all values
        3. Trains an AdaBoostClassifier.

        Input:
        - X_train (numpy array of x features with any number of rows)
        - y_train (numpy array of shape (1,))
        - optimize (boolean): If true, run gridsearchCV on n_estimators of AdaBoost

        Returns:
        - A trained sklearn pipeline
        '''
        imp = SimpleImputer(missing_values=np.nan, strategy='constant')

        pipe = Pipeline([('impute', SimpleImputer(np.nan, strategy='constant')),
                         ('scaler', StandardScaler()), 
                         ('ADA', AdaBoostClassifier(random_state=42))])

        if optimize==True:
            grid = GridSearchCV(pipe, cv=5, n_jobs=-1, param_grid={'ADA__n_estimators': [25,50,75,100]},scoring='roc_auc')
            grid.fit(X_train,y_train)
            return grid

        else:
            pipe.fit(X_train,y_train)
            return pipe

    def exportPipeline(pipe, version):
        '''
        Export a learnt pipeline to model/my_model_{{version}}.pkl

        Input:
        - pipeline (sklearn.pipeline)
        - version (string): name of file is ada_pipeline

        Output:
        Null
        '''
        path = f'../model/my_model_{version}.pkl'
        joblib.dump(pipe, path, compress=9)
        print(f'Model exported to {path}.')
    
    def load_model(self, version):
        path = f'../model/my_model_{version}.pkl'
        model = joblib.load(path)
        return model

    def predict_model(self, model, X_test):
        '''
        Returns the probability of default based on .
        '''
        return model.predict_proba(X_test)[:,1]