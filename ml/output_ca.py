import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data_files/dataset.csv')

le = LabelEncoder()
categorical_features = ['Gender', 'Area', 'AreaType', 'HouseType', 'District']
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


def print_report(y_test, y_pred):
    print("Accuracy of the model: ", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Dengue', 'Dengue']))


print_report(y_test, y_pred)

cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5,1.5], ['No Dengue', 'Dengue'])
plt.yticks([0.5,1.5], ['No Dengue', 'Dengue'])
plt.show()
