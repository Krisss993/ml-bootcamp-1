import numpy as np
import pandas as pd
import sklearn

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)
sklearn.__version__


pogoda = ['słonecznie', 'deszczowo', 'pochmurno', 'deszczowo', 'słonecznie', 'słonecznie', 'pochmurno', 'pochmurno', 'słonecznie']
temperatura = ['ciepło', 'zimno', 'ciepło', 'ciepło', 'ciepło', 'umiarkowanie', 'umiarkowanie', 'ciepło', 'zimno']

spacer = ['tak', 'nie', 'tak', 'nie', 'tak', 'tak', 'nie', 'tak', 'nie']      

raw_df = pd.DataFrame(data={'pogoda': pogoda, 'temperatura': temperatura, 'spacer': spacer})
df = raw_df.copy()
df

le = LabelEncoder()
df['spacer'] = le.fit_transform(df['spacer'])
df

df = pd.get_dummies(df, columns = ['pogoda', 'temperatura'], drop_first=True, dtype='int')
df

data = df.copy()
target = data.pop('spacer')

data
target

model = GaussianNB()

model.fit(data,target)

model.score(data, target)

data.iloc[[0]]

model.predict(data.iloc[[0]])

model.predict_proba(data)

