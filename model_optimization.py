import numpy as np
import pandas as pd


#Model selection and model building libraries


#Data Manipulation
from sklearn.model_selection import train_test_split


#Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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


if __name__ == '__main__':
    SEED = 5
    PERCENTAGE_OF_TOTAL_DATA_TO_USE = 0.8
    FOLDS = 5
    feature_extracted_data = pd.read_csv('data/feature_extracted_data_method_1_v0.csv')
    X = feature_extracted_data.drop(['TripType','VisitNumber','OnlyReturn','IsWeekend','Weekday'],1)
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


    print("Total Unique labels after split: ",len(set(y_train)))

    #X = pd.get_dummies(X, columns=["Weekday"], prefix=["weekday"])
    # print("X Shape: " ,X.shape)
    # print("y Shape: ",y.shape)

    parameters = {'n_estimators':[20,50,100],'min_samples_leaf':[1,10,25,50],'class_weight':['balanced']}
    rf =  RandomForestClassifier()
    clf = GridSearchCV(rf, parameters,cv=FOLDS,scoring="accuracy")
    clf.fit(X_train,y_train)

    print(clf.best_score_)

    print(clf.best_params_)
