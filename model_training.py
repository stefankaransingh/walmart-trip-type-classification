import numpy as np
import pandas as pd


#Model selection and model building libraries


#Data Manipulation
from sklearn.model_selection import train_test_split

#Data Preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#Model Selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV


#Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


#Helper Function
from helper import plot_metric_by_class



if __name__ == '__main__':
    SEED = 5
    PERCENTAGE_OF_TOTAL_DATA_TO_USE = 0.8
    FOLDS = 5
    feature_extracted_data = pd.read_csv('data/feature_extracted_data_method_1_v5.csv')
    X = feature_extracted_data.drop(['TripType','VisitNumber','IsWeekend','Weekday'],1)
    y = feature_extracted_data['TripType'].astype(str)
    labels = list(set(y))

    print("Total Unique labels before split: ",len(labels))

    print("X Shape: " ,X.shape)
    print("y Shape: ",y.shape)
    print("-------------")

    X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=PERCENTAGE_OF_TOTAL_DATA_TO_USE,random_state=SEED,stratify=y)


    print("X Shape: " ,X_train.shape)
    print("y Shape: ",y_train.shape)
    print("-------------")


    X_train,X_val,y_train,y_val= train_test_split(X_train,y_train,train_size=PERCENTAGE_OF_TOTAL_DATA_TO_USE,random_state=SEED,stratify=y_train)

    print("X train Shape: " ,X_train.shape)
    print("y train Shape: ",y_train.shape)
    print("-------------")

    print("Total Unique labels in y train after split: ",len(set(y_train)))


    print("X val Shape: " ,X_val.shape)
    print("y val Shape: ",y_val.shape)
    print("-------------")

    print("Total Unique labels in y val after split: ",len(set(y_val)))


    # models = []
    # models.append(('RF',RandomForestClassifier(class_weight='balanced',n_estimators=200)))
    #
    # names = []
    # outcome = []
    # for name,model in models:
    #     #10-fold Cross Validation
    #     kfold = KFold(n_splits=10,random_state=SEED)
    #     cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring="accuracy")
    #     outcome.append(cv_results)
    #     names.append(name)
    #     msg = "Model Name: %s | Mean Accuracy: %f | SD Accuracy: (%f)" % (name, cv_results.mean(), cv_results.std())
    #     print(msg)

    clf = RandomForestClassifier(class_weight='balanced',n_estimators=200)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_val)

    print(classification_report(y_val, y_pred))

    plot_metric_by_class(list(y_val),list(y_pred),list(set(y_val)),False,'accuracy','rf_accuracy.png')

    plot_metric_by_class(list(y_val),list(y_pred),list(set(y_val)),False,'f1','rf_f1.png')
