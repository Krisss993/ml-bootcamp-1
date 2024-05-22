
import pandas as pd
import numpy as np

import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

sns.set()


raw_data = load_iris()
raw_data

df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
df
df['target'] = raw_data.target
df



df.describe().T.apply(lambda x:round(x,2))

df['target'].value_counts()

plt.pie(x=df['target'].value_counts())
df.target.value_counts().plot(kind='pie')




# df.pop() wyrywa kolumnę, zmieniając także data
data = df
target = df.pop('target')
# ==
data = df.iloc[:,:-1]
target = df.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')
print(f'\nTest ratio: {len(X_test) / len(data):.2f}')
print(f'\ny_train:\n{y_train.value_counts()}')
print(f'\ny_test:\n{y_test.value_counts()}')


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')
print(f'\nTest ratio: {len(X_test) / len(data):.2f}')
print(f'\ny_train:\n{y_train.value_counts()}')
print(f'\ny_test:\n{y_test.value_counts()}')


# stratify=target - ustawia równy podział ze względu na target, rozkład y_train jest taki sam jak rozkład y_test
X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.9, stratify=target)

print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')
print(f'\nTest ratio: {len(X_test) / len(data):.2f}')
print(f'\ny_train:\n{y_train.value_counts()}')
print(f'\ny_test:\n{y_test.value_counts()}')

plt.scatter(X_train.iloc[:,0], y_train)

sns.pairplot(df)



raw_data = load_breast_cancer()

data = raw_data.data
target = raw_data.target

df = pd.DataFrame(data, columns=raw_data.feature_names)
df = pd.concat([df,pd.Series(target)],axis=1)
df.rename(columns={0:'target'}, inplace=True)
df
df.target.value_counts()

data = df
target = df.pop('target')


X_train, X_test, y_train, y_test = train_test_split(data, target, stratify=target)
print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')
print(f'\nTest ratio: {len(X_test) / len(data):.2f}')
print(f'\ny_train:\n{y_train.value_counts()}')
print(f'\ny_test:\n{y_test.value_counts()}')




