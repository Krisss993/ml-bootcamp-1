import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.offline as pyo
import plotly.figure_factory as ff

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
from sklearn import tree as tree_graph

from sklearn.datasets import make_regression, load_breast_cancer
import statsmodels.api as sm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import classification_report






sns.set()
np.set_printoptions(precision=6, suppress=True, edgeitems=10, linewidth=100000, 
                    formatter=dict(float=lambda x: f'{x:.2f}'))

def sigm(x):
    return 1/(1+np.exp(-x))

X = np.arange(-5,5,0.1)
X
y = sigm(X)

plt.figure(figsize=(8,6))
plt.plot(X,y)
plt.show()







raw_data = load_breast_cancer()
raw_data

data = raw_data.data
target = raw_data.target


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')


scaler = StandardScaler()
# WSZELKIE DOPASOWANIA PRZEPROWADZAMY NA ZBIORZE TRENINGOWYM
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




classifier = LogisticRegression()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

classifier.score(X_test, y_test)

y_prob = classifier.predict_proba(X_test)
y_prob

# MACIERZ KONFUZJI
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm)



def plot_confusion_matrix(cm):
    # klasyfikacja binarna
    cm = cm[::-1]
    cm = pd.DataFrame(cm, columns=['pred_0', 'pred_1'], index=['true_1', 'true_0'])

    fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), 
                                      colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=500, height=500, title='Confusion Matrix', font_size=16)
    pyo.plot(fig)

plot_confusion_matrix(cm)

# RAPORT KLASYFIKACJI
print(classification_report(y_test, y_pred))




