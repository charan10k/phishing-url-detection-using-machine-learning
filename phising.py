import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('python/phishing.csv')
label_encoders = {}
for column in df.columns:
    if df[column].dtype == object:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
model = DecisionTreeClassifier()
modelfit = model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
print(metrics.classification_report(ypred, ytest))
accuracy = metrics.accuracy_score(ytest, ypred)
print("\n\nAccuracy Score:", round(accuracy * 100, 2), "%")
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.savefig('confusion_matrix.png')
dot_file = 'tree.dot'
export_graphviz(model, out_file=dot_file, feature_names=X.columns, class_names=['-1', '1'], filled=True, rounded=True)