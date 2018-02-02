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

def cross_val_log_loss(clf,X,y,labels,folds,seed):
    results = []
    kfold = KFold(n_splits=folds,random_state=seed)
    for train_index, test_index in kfold.split(X):
        X_train_tmp, X_test_tmp = X[train_index], X[test_index]
        y_train_tmp, y_test_tmp = y[train_index], y[test_index]
        
        clf.fit(X_train_tmp,y_train_tmp)
        y_pred = clf.predict_proba(X_test_tmp)
        result = log_loss(y_test_tmp,y_pred,labels=labels)
        results.append(result)
    results = np.array(results)
    return results.mean(),results.std() 

if __name__ == '__main__':
	SEED = 5
	feature_extracted_data = pd.read_csv('data/feature_extracted_data_method_1.csv')
	X = feature_extracted_data.drop(['TripType','VisitNumber','OnlyReturn','IsWeekend'],1)
	y = feature_extracted_data['TripType'].astype(str)

	X = pd.get_dummies(X, columns=["Weekday"], prefix=["weekday"])
	print("X Shape: " ,X.shape)
	print("y Shape: ",y.shape)

	X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=0.8,random_state=SEED,stratify=y)
	
	models =[]

	models.append(('LR',LogisticRegression(class_weight='balanced')))
	models.append(('KNN',KNeighborsClassifier()))
	models.append(('CART',DecisionTreeClassifier(class_weight='balanced')))
	models.append(('NB',GaussianNB()))
	models.append(('LinearSVC',LinearSVC(class_weight='balanced')))
	models.append(('RF',RandomForestClassifier(class_weight='balanced')))
	names = []
	outcome = []
	for name,model in models:
    		#10-fold Cross Validation
    		kfold = KFold(n_splits=10,random_state=SEED)
    		cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring="accuracy")
    		outcome.append(cv_results)
    		names.append(name)
    		msg = "Model Name: %s | Mean Accuracy: %f | SD Accuracy: (%f)" % (name, cv_results.mean(), cv_results.std())
    		print(msg)
	
	models =[]
	models.append(('KNN',KNeighborsClassifier()))
	models.append(('RF',RandomForestClassifier(class_weight='balanced')))
	labels = list(set(y))
	for name,model in models:
    		mean,std =cross_val_log_loss(model,X_train,y_train,labels,5,SEED)
    		print("Model Name: %s | Mean Log Loss: %f | SD Log Loss: (%f)" % (name, mean, std))
