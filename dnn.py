import numpy as np
import pandas as pd


#Data Manipulation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Model Selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

#For Neural Network Model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

def baseline_model():
    model =Sequential()
    model.add(Dense(5253,input_dim=5253,init='normal',activation='relu'))
    model.add(Dense(38,init='normal',activation='sigmoid'))
    #Compile the model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


if __name__ == '__main__':
    SEED = 5
    PERCENTAGE_OF_TOTAL_DATA_TO_USE = 0.25
    KFOLDS = 3
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

    #encode the class values as integers
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y = encoder.transform(y_train)

    #Convert the inetgers to dummy variables
    dummy_y = np_utils.to_categorical(encoded_y)

    estimator = KerasClassifier(build_fn=baseline_model,epochs=100,batch_size=2000,verbose=0)

    kfold = KFold(n_splits=KFOLDS,shuffle=True,random_state=SEED)
    results = cross_val_score(estimator,np.array(X_train),np.array(dummy_y),cv=kfold)
    print("Accuracy: %.2f%% SD: (%.2f%%)" % (results.mean()*100, results.std()*100))
