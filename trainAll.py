import pandas as pd
data=pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
data=data.dropna(how='any')
data=data.drop(["Person ID"],axis=1)

#Label encoding
category_colums=['Gender','Occupation','BMI Category','Blood Pressure']
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data[category_colums] = data[category_colums].apply(encoder.fit_transform)

X=data.iloc[:,:-1]
y=data.iloc[:,-1]
#array Conver
X=X.to_numpy()

#spilit data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

import warnings
warnings.filterwarnings("ignore")
names = ["K-Nearest Neighbors", "SVM",
         "Decision Tree", "Random Forest",
         "Naive Bayes","ExtraTreesClassifier","VotingClassifier"]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier

classifiers = [
    KNeighborsClassifier(),
    LinearSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    ExtraTreesClassifier(),
    VotingClassifier(estimators=[('DT', DecisionTreeClassifier()), ('rf', RandomForestClassifier()), ('et', ExtraTreesClassifier())], voting='hard')]

clfF=[]
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(name)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('--------------------------------------------------------------')
    clfF.append(clf)

import pickle
pickle.dump(clfF, open("model.pkl", 'wb'))  
pickle.dump(encoder, open("encoder.pkl",'wb'))    
    
    
    
    
    
    
    
    
    
    