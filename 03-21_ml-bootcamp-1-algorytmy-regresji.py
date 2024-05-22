import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sns.set()


df = pd.DataFrame({'Dochody':[210,270,290,310,370,400,450,480,510,520],'Wydatki':[140,190,250,270,290,310,340,360,420,390]})
df

plt.figure(figsize=(6,6))
plt.scatter(df['Dochody'], df['Wydatki'])

d_mean = df['Dochody'].mean()

w_mean = df['Wydatki'].mean()


df['licznik'] = df['Wydatki']*(df['Dochody']-d_mean)
df['mianownik'] = (df['Dochody'] - d_mean)**2
df
a = df['licznik'].sum()/df['mianownik'].sum()
b=a*d_mean-w_mean

plt.figure(figsize=(6,6))
plt.scatter(df['Dochody'], df['Wydatki'])
plt.plot(df['Dochody'], a*df['Dochody']+b, color='red')



X = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3000, 3250, 3500, 3750, 4000, 4250])
m = len(X)

print(f'Lata pracy: {X}')
print(f'Wynagrodzenie: {Y}')
print(f'Liczba próbek: {m}')
     
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

ones = np.ones((len(X),1))
ones

X_1 = np.append(X, ones, axis=1)
X_1

np.dot(X_1.T,X_1)
W = np.dot(np.linalg.inv(np.dot(X_1.T,X_1)), np.dot(X_1.T,Y))
W

df = pd.DataFrame({'X':X.ravel(),'Y':Y.ravel()})
df

x_mean = df['X'].mean()
y_mean = df['Y'].mean()

df['licznik'] = df['Y'] * (df['X']-x_mean)
df
df['mianownik'] = (df['X']-x_mean)**2
df
a = df.licznik.sum()/df.mianownik.sum()
b = y_mean - a*x_mean



# METODA SPADKU WZDLOZ GRADIENTU


eta = 0.01
w = np.random.rand(2,1)

intercept = []
coef = []

for i in range(3000):
    gradient = (2/m) * X_1.T.dot(X_1.dot(w) - Y)
    w = w - eta*gradient
    intercept.append(w[0][0])
    coef.append(w[1][0])
print(w)


df=pd.DataFrame({'intercept':intercept,'coef':coef})

df
