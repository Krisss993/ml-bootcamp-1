
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo


np.random.seed(42)



y_true = 100 + 20 * np.random.randn(1000)
y_true
     


y_pred = y_true + 10 * np.random.randn(1000)
y_pred


df = pd.DataFrame({'y_true':y_true, 'y_pred':y_pred})

df['Error'] = df['y_true'] - df['y_pred']

df['Error_SQ'] = df['Error']**2
df

mae = df['Error'].abs().sum()/len(df)
mae

mse = df['Error_SQ'].sum()/len(df)
mse

rmse = mse**(1/2)
rmse

def plot_regression_results(y_true, y_pred): 

    results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    min = results[['y_true', 'y_pred']].min().min()
    max = results[['y_true', 'y_pred']].max().max()

    fig = go.Figure(data=[go.Scatter(x=results['y_true'], y=results['y_pred'], mode='markers'),
                    go.Scatter(x=[min, max], y=[min, max])],
                    layout=go.Layout(showlegend=False, width=800,
                                     xaxis_title='y_true', 
                                     yaxis_title='y_pred',
                                     title='Regresja: y_true vs. y_pred'))
    pyo.plot(fig)

plot_regression_results(y_true, y_pred)


res = df['Error']

fig = px.histogram(df,x=df['Error'], nbins=50)
pyo.plot(fig)

def r2(y_true, y_pred):
    licznik = (sum(y_true-y_pred))**2
    mianownik = (sum(y_true - y_true.mean()))**2
    try:
        r2 = 1 - licznik/mianownik
    except ZeroDivisionError:
        print('0')
    return r2
