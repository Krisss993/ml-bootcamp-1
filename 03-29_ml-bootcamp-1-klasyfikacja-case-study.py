import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
import plotly.express as px
import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import datasets

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets.fashion_mnist import load_data

np.random.seed(42)
sns.set(font_scale=1.3)

raw_digits = datasets.load_digits()
digits = raw_digits.copy()

digits.keys()

images = digits['images']
targets = digits['target']

print(f'images shape: {images.shape}')
print(f'targets shape: {targets.shape}')



plt.figure(figsize=(12, 10))
for index, (image, target) in enumerate(list(zip(images, targets))[:10]):
    plt.subplot(5,2,index+1)
    plt.axis('off')
    plt.imshow(image, cmap='Greys')
    plt.title(f'Label: {target}')
plt.show()

plt.figure(figsize=(18, 13))
for i in range(1, 11):
    plt.subplot(1, 10, i)
    plt.axis('off')
    plt.imshow(images[i-1], cmap='gray_r')
    plt.title(targets[i-1], color='black', fontsize=16)
plt.show()






X_train, X_test, y_train, y_test = train_test_split(images, targets)


print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')



classifier = SVC(gamma=0.001, kernel='linear')
classifier.fit(X_train, y_train)

classifier.score(X_test, y_test)




classifier = SVC(gamma=0.001, kernel='rbf')
classifier.fit(X_train, y_train)

classifier.score(X_test, y_test)


y_pred = classifier.predict(X_test)

classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


fig = ff.create_annotated_heatmap(z=cm, x=['pred_'+str(i) for i in range(10)], y=['pred_'+str(i) for i in range(10)], colorscale='ice', showscale=True, reversescale=True)
pyo.plot(fig)



results = pd.DataFrame(data={'y_pred': y_pred, 'y_test': y_test})
results.head(10)

errors = results[results['y_pred']!=results['y_test']]
error_idx = errors.index

plt.figure(figsize=(12, 10))
for index, error_id in enumerate(error_idx):
    image = X_test[error_id].reshape(8,8)
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap='Greys')
    plt.title(f"True {results.loc[error_id, 'y_test']} Prediction: {results.loc[error_id, 'y_pred']}")




































(X_train, y_train), (X_test, y_test) = load_data()




print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')
print(f'X_train[0] shape: {X_train[0].shape}')









plt.imshow(X_train[0], cmap='gray_r')
plt.axis('off')





class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(18, 13))
for i in range(1, 11):
    plt.subplot(1, 10, i)
    plt.axis('off')
    plt.imshow(X_train[i-1], cmap='gray_r')
    plt.title(class_names[y_train[i-1]], color='black', fontsize=16)
plt.show()





plt.figure(figsize=(12, 10))
for index, (image, target) in enumerate(list(zip(X_train, y_train))[:6]):
    plt.subplot(2, 6, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap='Greys')
    plt.title(f'Label: {class_names[y_train[index]]}')



X_train = X_train / 255.
X_test = X_test / 255.



X_train = X_train.reshape(X_train.shape[0], 28*28)
X_test = X_test.reshape(X_test.shape[0], 28*28)



print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')




classifier = SVC()

classifier.fit(X_train, y_train)

classifier.score(X_test, y_test)

y_pred = classifier.predict(X_test)
classifier.predict_proba(X_test)


res = pd.DataFrame(data={'y_true':y_test, 'y_pred':y_pred})

errors = res[res['y_true']!=res['y_pred']]

errors_idx = errors.index




plt.figure(figsize=(12, 10))
for idx, error_id in enumerate(errors_idx[:10]):
    image = X_test[error_id].reshape(28,28)
    plt.subplot(5,2, idx+1)
    plt.imshow(image, cmap='gray_r')
    plt.title(f'True: {class_names[res.loc[error_id,"y_true"]]}, Pred: {class_names[res.loc[error_id,"y_pred"]]}')
    plt.axis('off')
plt.show()


plt.figure(figsize=(12, 10))
for idx, error_idx in enumerate(errors_idx[:15]):
    image = X_test[error_idx].reshape(28,28)
    plt.subplot(5,3, idx+1)
    plt.imshow(image, cmap='gray_r')
    plt.title(f'True: {class_names[res.loc[error_idx, "y_true"]]}, Pred: {class_names[res.loc[error_idx, "y_pred"]]}')
    plt.axis('off')
plt.show()



cm = confusion_matrix(y_test, y_pred)
cm


def plot_conf_matrix(cm):
    cm = cm[::-1]
    df = pd.DataFrame(cm ,columns=class_names, index=class_names[::-1])

    fig = ff.create_annotated_heatmap(df.values, x=list(df.columns), y=list(df.index), colorscale='ice', showscale=True, reversescale=True)
    pyo.plot(fig)

plot_conf_matrix(cm)


cf = classification_report(y_test, y_pred, output_dict=True)
cf










