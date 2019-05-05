import numpy as np
#import matplotlib.pyplot as plt
#from sklearn import svm
import pandas as pd
#from matplotlib import style
from sklearn.model_selection import train_test_split
#style.use("ggplot")
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix  
import time


#start timer
t0= time.time()

#load dataframe
df = pd.read_csv('C:/Users/Matthias/Desktop/classData.csv', names=['cid', 'id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10', 'id11', 'id12', 'id13', 'id14', 'id15', 'id16', 'id17', 'id18', 'id19', 'id20'])

#set up the training test split
train, test = train_test_split(df, test_size=0.3)

train_X=train.ix[:,2:]
train_y=train.ix[:,1]

test_X=test.ix[:,2:]
test_y=test.ix[:,1]

#clf = svm.SVC(kernel="linear", C= 1.0)
#clf.fit(train_X,train_y)

#build svm classifier
svclassifier = SVC(kernel='linear')  
svclassifier.fit(train_X, train_y)

#make prediction
y_pred = svclassifier.predict(test_X)  

#evaluate
print(confusion_matrix(test_y,y_pred))  
print(classification_report(test_y,y_pred)) 

#stop timer
t1 = time.time() 
print("Time elapsed: ", t1- t0)