from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


iris = load_iris()
X = iris.data
Y = iris.target

X_train , X_test , Y_train, Y_test = train_test_split(X,Y, random_state=42, test_size=0.5)

# Build the model_selection

clf = RandomForestClassifier(n_estimators=10)

#Train the classifier
clf.fit(X_train, Y_train)

# Prediction

predicted = clf.predict(X_test)

print(accuracy_score(predicted,Y_test))

with open('/Users/niles/Desktop/Flask-application/machine_learning model woth flask/rf.pkl','wb') as model_pkl:
    pickle.dump(clf , model_pkl)
