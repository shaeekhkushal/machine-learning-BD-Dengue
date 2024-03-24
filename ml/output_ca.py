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
mappings = {}
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])
    mappings[feature] = {idx: label for idx, label in enumerate(le.classes_)}

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

gender_dengue = df.groupby('Gender')['Outcome'].mean()
print("Dengue rates by gender:")
for gender, rate in gender_dengue.items():
    print(f"{mappings['Gender'][gender]}: {rate*100:.2f}%")

area_dengue = df.groupby('Area')['Outcome'].mean()
print("Dengue rates by area:")
for area, rate in area_dengue.items():
    print(f"{mappings['Area'][area]}: {rate*100:.2f}%")

# Convert the series to a dataframe
gender_dengue = gender_dengue.reset_index()
gender_dengue.columns = ['Gender', 'DengueRate']
# Replace encoded labels with original labels
gender_dengue['Gender'] = gender_dengue['Gender'].map(mappings['Gender'])
# Save to CSV
gender_dengue.to_csv('gender_dengue.csv', index=False)

# Repeat for area_dengue
area_dengue = area_dengue.reset_index()
area_dengue.columns = ['Area', 'DengueRate']
area_dengue['Area'] = area_dengue['Area'].map(mappings['Area'])
area_dengue.to_csv('area_dengue.csv', index=False)
