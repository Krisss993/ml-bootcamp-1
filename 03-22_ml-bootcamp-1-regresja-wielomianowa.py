import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.offline as pyo

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import statsmodels.api as sm

sns.set()

np.random.seed(42)
X = np.arange(-10, 10, 0.5)
noise = 80 * np.random.randn(40)
y = -X**3 + 10*X**2 - 2*X + 3 + noise
X = X.reshape(40, 1)

plt.figure(figsize=(8, 6))
plt.title('Regresja wielomianowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.legend()
plt.show()


# REGRESJA LINIOWA
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.plot(X, y_pred, label='cecha x', color='red')
plt.legend()
plt.show()

lr.score(X, y)

# REGRESJA WIELOMIANOWA
lrpoly = LinearRegression()
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

lrpoly.fit(X_poly, y)
y_pred_poly = lrpoly.predict(X_poly)

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.plot(X, y_pred_poly, label='cecha x', color='red')
plt.legend()
plt.show()

lrpoly.score(X_poly, y)

df = pd.DataFrame(X.ravel())
df.rename(columns={0: 'X'}, inplace=True)
df
dfp = pd.DataFrame(X_poly)
dfp


results = pd.DataFrame(data={
    'name': ['regresja liniowa', 'regresja wielomianowa st. 3'],
    'r2_score': [r2_score(y, y_pred), r2_score(y, y_pred_poly)],
    'mae': [mae(y, y_pred), mae(y, y_pred_poly)],
    'mse': [mse(y, y_pred), mse(y, y_pred_poly)],
    'rmse': [np.sqrt(mse(y, y_pred)), np.sqrt(mse(y, y_pred_poly))]
})
results


fig = px.bar(results, x='name', y='r2_score', width=700, title='Regresja wielomianowa - R2_score')
pyo.plot(fig)

fig = px.bar(results, x='name', y='mae', width=700, title='Regresja wielomianowa - mean absolute error')
pyo.plot(fig)

fig = px.bar(results, x='name', y='mse', width=700, title='Regresja wielomianowa - mean squared error')
pyo.plot(fig)   

fig = px.bar(results, x='name', y='rmse', width=700, title='Regresja wielomianowa - root mean squared error')
pyo.plot(fig)





