import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import plotly_express as px
import plotly.offline as pyo

import statsmodels.api as sm

sns.set()

df_raw = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/insurance.csv')
df_raw.head()

df = df_raw.copy()
df.info()


for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')
df.info()

df[df.duplicated()]

df.drop_duplicates(inplace=True)
df.info()

df.isnull().sum()

df.describe(include='category').T

df = pd.get_dummies(df, drop_first=True, dtype='int')
df
df.info()

df.describe().T

df.sex_male.value_counts()
df.children.value_counts()


plt.pie(df.sex_male.value_counts())
df.sex_male.value_counts().plot(kind='pie')

plt.hist(df['charges'], bins=50)

# FACET DZIELI NA KOLUMNY/WIERSZE W ZALEZNOSCI OD ZMIENNEJ 0-1(KATEGORYCZNEJ)
fig = px.histogram(data_frame=df, x='charges', nbins=50, facet_col='smoker_yes', facet_row='sex_male')
pyo.plot(fig)

fig = px.histogram(data_frame=df, x='smoker_yes', facet_col='sex_male', color='sex_male')
pyo.plot(fig)

corr = df.corr()
corr

# MACIERZ KORELACJI
sns.set(style="white")
mask = np.zeros_like(corr, dtype='bool')
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

fig = plt.figure(figsize=(8, 6))
sns.heatmap(corr, square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmax=.5, center=0,)




# WARTOSCI KORELACJI POSZCZEGOLNYCH KOLUMN Z KOLUMNA CHARGES POSORTOWANE MALEJACO
df.corr()['charges'].sort_values(ascending=False)


sns.set()
df.corr()['charges'].sort_values(ascending=True).plot(kind='barh')

X = df.copy()
y = X.pop('charges')

X.head()
y.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

lr.score(X_test, y_test)

lr.coef_
lr.intercept_


X_train.shape
y_train.shape

X_test.shape
y_test.shape


predictions = pd.DataFrame({'y_true':y_test, 'y_pred':y_pred})
predictions['Error'] = predictions['y_true'] - predictions['y_pred']

predictions['Error'].plot(kind='hist', bins=50)
plt.hist(predictions['Error'], bins=50)

predictions


MSE = mean_squared_error(y_test, y_pred)
MSE

MAE = mean_absolute_error(y_test, y_pred)
MAE








# ELIMINACJA WSTECZNA




X_train_ols = X_train.copy()
X_train_ols = X_train_ols.values
X_train_ols = sm.add_constant(X_train_ols)
X_train_ols

# CHARAKTERYSTYKI STATYSTYCZNE DOTYCZACE MODELU
ols = sm.OLS(endog=y_train, exog=X_train_ols).fit()
predictors = ['const'] + list(X_train.columns)
print(ols.summary(xname=predictors))


# USUWAMY TAM GDZIE P value > 0.4
X_selected = X_train_ols[:,[0,1,2,3,4,5,7,8]]
predictors.remove('region_northwest')

ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print(ols.summary(xname=predictors))


X_selected = X_train_ols[:,[0,1,2,3,5,6,7]]
predictors.remove('sex_male')
ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print(ols.summary(xname=predictors))


X_selected = X_train_ols[:,[0,1,2,3,4,6]]
predictors.remove('region_southeast')
ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print(ols.summary(xname=predictors))

X_selected = X_train_ols[:,[0,1,2,3,4]]
predictors.remove('region_southwest')
ols=sm.OLS(endog=y_train,exog=X_selected).fit()
print(ols.summary(xname=predictors))

ols.pvalues.values

# AUTOMATYZACJA ELIMINACJI WSTECZNEJ


def backward_eliminate(dataframe, p=0.05):
    data = dataframe.copy()
    target = data.pop('charges')
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    print(MAE, lr.score(X_test, y_test))
    
    
    X_train_ols = sm.add_constant(X_train.copy().values)
    ols = sm.OLS(endog=y_train, exog=X_train_ols).fit()
    predictors = ['const'] + list(data.columns)
    print(ols.summary(xname=predictors))
    X_selected = X_train_ols.copy()
    max_p = ols.pvalues.max()
    max_p_col = ols.pvalues.argmax()
    
    while max_p > p:
        X_selected = np.delete(X_selected, max_p_col, axis=1)
        ols = sm.OLS(endog=y_train, exog=X_selected).fit()
        predictors.pop(max_p_col)
        print(ols.summary(xname=predictors))
        max_p = ols.pvalues.max()
        max_p_col = ols.pvalues.argmax()
    return ols.summary(xname=predictors)
    
backward_eliminate(df)


# ZAPISANIE MODELU DO PLIKU model.pickle
ols.save('model.pickle')
