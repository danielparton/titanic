import os
import pandas as pd
import numpy as np
import scipy.stats


def parse_orig_train_data():
    filepath = os.path.join('orig_data', 'train.csv')
    df = pd.read_csv(filepath, header=0)
    return df


def process_data(filename):
    df = pd.read_csv(filename, header=0)

    # =======
    # Make integer Gender column
    # =======
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # =======
    # Fill in missing Ages with median values
    # =======
    median_ages = np.zeros((2,3))
    for i in range(0,2):
        for j in range(0,3):
            median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

    for i in range(0,2):
        for j in range(0,3):
            df.loc[ df.Age.isnull() & (df.Gender == i) & (df.Pclass == j+1), 'Age' ] = median_ages[i,j]

    # df['AgeIsNull'] = df.Age.isnull().astype(int)

    # =======
    # Fill in missing Fares with median values
    # =======

    median_fares = np.zeros(3)
    for i in range(0,3):
        median_fares[i] = df[df['Pclass'] == i+1]['Fare'].dropna().median()

    for i in range(0,3):
        df_loc = df.loc[ df.Fare.isnull() & (df.Pclass == i+1), 'Fare' ]
        if len(df_loc) > 0:
            df.loc[ df.Fare.isnull() & (df.Pclass == i+1), 'Fare' ] = median_fares[i]

    # =======
    # Fill in Embarked (C, Q, S) and convert to integer (0,1,2)
    # =======
    embarked_mappings = {'C': 0, 'Q': 1, 'S': 2}
    # this will convert to floats, since there are NaNs present
    df['Embarked'] = df['Embarked'].map(embarked_mappings)

    # Embarked only has two null values, but might as well do this properly anyway
    # estimate probability distribution from non-null values
    embarked_probs = {}
    for x in df.Embarked.value_counts().iteritems():
        embarked_probs[x[0]] = float(x[1]) / float(df.Embarked.count())

    embarked_random_picker = scipy.stats.rv_discrete(values=(embarked_probs.keys(), embarked_probs.values()))

    for row in df[df.Embarked.isnull()].iterrows():
        row_index = row[0]
        new_value = embarked_random_picker.rvs()
        df.loc[df.index == row_index, 'Embarked'] = new_value

    # now convert to ints
    df['Embarked'] = df['Embarked'].astype(int)


    # =======
    # A couple of new features
    # =======
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Age*Class'] = df.Age * df.Pclass

    # =======
    # Drop unwanted columns
    # =======
    drop_columns = list(df.dtypes[df.dtypes.map(lambda x: x=='object')].keys()) # string-type columns
    df = df.drop(drop_columns, axis=1)

    return df
