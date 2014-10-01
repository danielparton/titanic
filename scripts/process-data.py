import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv', header=0)

df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

median_ages = np.zeros((2,3))
median_ages
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
median_ages

df['AgeFill'] = df['Age']
for i in range(0,2):
    for j in range(0,3):
        df.loc[ df.Age.isnull() & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill' ] = median_ages[i,j]

df['AgeIsNull'] = df.Age.isnull().astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

object_columns = list(df.dtypes[df.dtypes.map(lambda x: x=='object')].keys())

df = df.drop(object_columns, axis=1)

train_data = df.values
