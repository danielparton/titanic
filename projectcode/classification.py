import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def run_rforest(train_data, test_data):
    # Create the random forest object which will include all the parameters
    # for the fit
    forest = RandomForestClassifier(n_estimators = 100)

    # Fit the training data to the Survived labels and create the decision trees
    # first param is feature columns
    # second param is Survived column
    forest = forest.fit(train_data[:, 1:], train_data[:, 0])

    # Take the same decision trees and run it on the test data
    output = forest.predict(test_data)
    return output

def write_output(index, predictions):
    df = pd.DataFrame({'Survived': predictions}, index=index)
    df.Survived = df.Survived.astype(int)
    ofilepath = os.path.join('predictions', 'predictions.csv')
    df.to_csv(ofilepath)
    return df
