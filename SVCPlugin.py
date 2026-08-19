import os
import pandas as pd
import numpy as np
import PyPluMA
from sklearn import svm

class SVCPlugin:
    def input(self, inputfile):
        global parameters, spdata_train, spdata_test, ytrain
        parameters = pd.read_table(inputfile, sep="\t", header=None, index_col=0).squeeze("columns")
        if 'training' in parameters:
            spdata_train = pd.read_csv(os.path.join(PyPluMA.prefix(), parameters['training']), index_col=0)
            print("Training data loaded")
        else:
            print("Error: 'training' not found in parameters")
        if 'testing' in parameters:
            spdata_test = pd.read_csv(os.path.join(PyPluMA.prefix(), parameters['testing']), index_col=0)
            print("Testing data loaded")
        else:
            print("Error: 'testing' not found in parameters")
        if 'traininggroups' in parameters:
            ytrain = np.loadtxt(os.path.join(PyPluMA.prefix(), parameters['traininggroups']), delimiter=',')
            print("Training labels loaded")
        else:
            print("Error: 'traininggroups' not found in parameters")

    def run(self):
        global spdata_train, spdata_test, ytrain, pred
        if 'spdata_train' not in globals():
            print("Error: spdata_train not defined")
            pred = np.array([])
            return
        try:
            if not spdata_train.empty:
                # Remove zero columns
                spdata_train = spdata_train.loc[:, (spdata_train != 0).any(axis=0)]
                spdata_test = spdata_test.loc[:, (spdata_test != 0).any(axis=0)]
                # Fit SVM model
                #clf = svm.SVC(kernel='linear')
                clf = svm.SVC(kernel='poly', degree=3, C=1)
                clf.fit(spdata_train, ytrain)
                # Predict on test set
                pred = clf.predict(spdata_test)
            else:
                pred = np.array([])
        except NameError:
            print("Error: spdata_train not defined")
            pred = np.array([])

    def output(self, outputfile):
        global pred
        np.savetxt(outputfile, pred, delimiter=",")
