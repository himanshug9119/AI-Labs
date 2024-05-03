# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np


data =pd.read_csv("car.data")


data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])


X = data.drop('class', axis=1)
y = data['class']


def func(X, y, test_size, criterion, n_repeats):
    accuracy_scores = []
    f_scores = []

    for _ in range(n_repeats):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        
        clf = DecisionTreeClassifier(criterion=criterion)

        
        clf.fit(X_train, y_train)

        
        y_pred = clf.predict(X_test)

        
        cm = confusion_matrix(y_test, y_pred)
        fscore = f1_score(y_test, y_pred, average='weighted')

        
        accuracy = (cm.diagonal().sum()) / cm.sum()
        accuracy_scores.append(accuracy)
        f_scores.append(fscore)

    
    average_accuracy = np.mean(accuracy_scores)
    average_fscore = np.mean(f_scores)
    return average_accuracy, average_fscore


test_sizes = [0.4, 0.3, 0.2]
criteria = ['entropy', 'gini']
n_repeats = 20


results = {}

for test_size in test_sizes:
    for criterion in criteria:
        key = f'{int((1-test_size)*100)}% training data with {criterion}'
        results[key] = func(X, y, test_size, criterion, n_repeats)


for key, value in results.items():
    print(f"{key}: Average Accuracy: {value[0]}, Average F-score: {value[1]}")


print("\nOverfitting Example:")
print("If a decision tree is trained on a dataset with noise and it learns the noise as if it were a true pattern,")
print("it will make incorrect predictions when presented with new data. This is because it has learned the specific")
print("details of the training data, rather than the underlying patterns that could generalize to new data.")