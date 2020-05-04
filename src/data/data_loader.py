import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.model_selection import train_test_split

# TODO decide if using confidence scores


class KTHDataLoader:
    def __init__(self, fpath, use_confidence_scores=True):
        ''' reads .csv file at fpath and stores dataset '''
        self.data = pd.read_csv(fpath)
        # convert frames to numpy arrays
        self.data['frames'] = self.data['frames'].apply(lambda x: np.asarray(literal_eval(x)))
        self.use_confidence_scores = use_confidence_scores

    def load_data(self, data=None):
        ''' loads data from dataframe data into two numpy arrays X (openpose keypoints), Y: labels 
            X.shape is (# samples,). Each sample in X is of shape (# frames, 25, 3) where 25 = # joints.
            if data is None, uses self.data (all data). Otherwise uses data passed in from filter_by_subjects or filter_by_scenarios
        '''
        if data is None:
            data = self.data

        if not self.use_confidence_scores:
            data['frames'] = data['frames'].apply(lambda x: x[:, :, :2])

        X = data['frames']
        Y = data['action']
        # TODO one hot encode

        return X, Y

    def filter_by_subjects(self, subjects):
        '''
            subjects: list of subjects to filter by e.g. [subject1, subject2, etc]
        '''
        df = self.data[self.data['subject'] in subjects]
        X, Y = self.load_data(df)

        return X, Y

    def filter_by_scenarios(self, scenarios):
        '''
            scenarios: list of scenarios to load from e.g. [scenario1, scenario2]
        '''
        df = self.data[self.data['scenario'] in scenarios]
        X, Y = self.load_data(df)

        return X, Y

    def split_train_test(self, X, Y, train=0.7):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train, random_state=1, stratify=Y)
        return X_train, Y_train, X_test, Y_test

    def split_train_val_test(self, X, Y, train=0.7, val=0.2, test=0.1):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test, random_state=1, stratify=Y)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=train/(train+val), \
                                                          random_state=1, stratify=Y_train)
        return X_train, Y_train, X_val, Y_val, X_test, Y_test


if __name__ == '__main__':
    data_loader = KTHDataLoader('../../datasets/kth_actions.csv', use_confidence_scores=False)
    X, Y = data_loader.load_data()
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_loader.split_train_val_test(X, Y)


