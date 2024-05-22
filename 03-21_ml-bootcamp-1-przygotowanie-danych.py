import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


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
df_raw
df = df_raw.copy()


df[['size','color','gender','bought']] = df[['size','color','gender','bought']].astype('category')

df.info()
df.describe().T
df.describe(include='category').T
df


# pd.get_dummies(df,columns=['size','color','gender','bought'])
# df.columns


# df

# MAPOWANIE KOLUMNY BOUGHT
# df['bought'] = df['bought'].apply(lambda x:1 if x=='yes' else 0)

le = LabelEncoder()

le.fit(df['bought'])
le.transform(df['bought'])
# FIT TRANSFORM ZASTEPUJE POPRZEDNIE 2 KROKI
le.fit_transform(df['bought'])

le.classes_

# PRZYPISANIE MAPOWANIA DO KLOMNUY
df['bought'] = le.fit_transform(df['bought'])

# PRZYWROCENIE POSTACI PRZED TRANSFORMACJA
df['bought'] = le.inverse_transform(df['bought'])
df

# sparse - True to macierz rzadka, przetrzymuje tylko dane zawierajace 1, bez danych 0
# sparse - False - zwraca całą macierz z 1 i 0
encoder = OneHotEncoder(sparse_output=False)

encoder.fit(df[['size']])
encoder.transform(df[['size']])


# ZWRACA NAZWY KOLUMN DLA DANYCH
encoder.categories_



# USUWAMY PIERWSZA KOLUMNE ABY NIE POPELNIC BLEDU ZERO JEDYNKOWEGO
# PO KODOWANIU 0-1 LICZBA KOLUMN N ZMNIEJSZA SIE DO N-1

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoder.fit_transform(df[['size']])


# DUMMIES PRZEWAZNIE LEPSZE OD OneHotEncoder
pd.get_dummies(df, drop_first=True, dtype='int', prefix='df', prefix_sep='-')




# STANDARYZACJA

sr = df['weight'].mean()
sr

os = np.std(df['weight'])
os
(df['weight']/sr).mean()


df['weight_std'] = (df['weight']-sr)/os

df['weight_std'].mean()
np.std(df['weight_std'])


def standarize(x):
    return (x-x.mean())/x.std()

df['weight_std'] = standarize(df['weight'])

df['weight_std'].mean()
df['weight_std'].std()

scaler = StandardScaler()
df[['weight','price']] = scaler.fit_transform(df[['weight','price']])
df


