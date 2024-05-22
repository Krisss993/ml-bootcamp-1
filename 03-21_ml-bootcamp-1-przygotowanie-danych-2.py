
import pandas as pd
import numpy as np

import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()



data = {
    'size': ['XL', 'L', 'M', 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'gender': ['female', 'male', 'male', 'female', 'female'],
    'price': [199.0, 89.0, 99.0, 129.0, 79.0],
    'weight': [500, 450, 300, 380, 410],
    'bought': ['yes', 'no', 'yes', 'no', 'yes']
}


df_raw = pd.DataFrame(data)
df = df_raw.copy()
df
df.info()

le = LabelEncoder()
df['bought'] = le.fit_transform(df['bought'])
df[['size', 'color', 'gender']] = df[['size', 'color', 'gender']].astype('category')
df.info()

df = pd.get_dummies(df, drop_first=True, dtype='int')
df

scaler = StandardScaler()
df[['price','weight']] = scaler.fit_transform(df[['price','weight']])

df



data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df_raw = pd.DataFrame(data=data)
df_raw
df = df_raw.copy()

df = df.dropna(axis=0,how='any', subset=['gender'])
df

df['size'] = df['size'].fillna('brak')
df

df.isnull()

df.isnull().sum().sum()
df.isnull().sum()/ len(df)


# UZUPELNIANIE DANYCH PRZEZ IMPUTER

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(df[['weight']])

# OBLICZONA WARTOSC - MEAN
imputer.statistics_

df[['weight']] = imputer.transform(df[['weight']])
df


df = df_raw.copy()
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='L')
imputer.fit(df[['size']])

df[['size']] = imputer.transform(df[['size']])

df



imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(df[['gender']])
df[['gender']] = imputer.transform(df[['gender']])

df


# WSZYSTKIE WIERSZE Z WARTOSCIAMI NAN w kolumnie weight
df[pd.isnull(df['weight'])]

# WSZYSTKIE PELNE WIERSZE w weight
df[~pd.isnull(df['weight'])]


pd.notnull(df)

df[pd.notnull(df['weight'])]













data = {'price': [108, 109, 110, 110, 109, np.nan, np.nan, 112, 111, 111]}
date_range = pd.date_range(start='01-01-2020 09:00', end='01-01-2020 18:00', periods=10)

df = pd.DataFrame(data=data, index=date_range)
df

plt.figure(figsize=(10, 4))
plt.title('Braki danych')
_ = plt.plot(df.index,df.price)


df['price'] = df['price'].bfill()

plt.plot(df.index, df.price)





