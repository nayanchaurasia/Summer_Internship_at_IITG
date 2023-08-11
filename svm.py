import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,f1_score,precision_score,recall_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

###############################################################################################################################
data=pd.read_csv('C:/ecg_all_data.csv')
x=data.iloc[:,0:240]
y=data.iloc[:,240]

#LEVEL ENCODING
label_mapping = {'N': 0, 'L': 1, 'R': 2,'B': 3, 'A': 4, 'a': 5,'J':6,'S':7,
    'V':8,'r':9,'F':10,'e':11,'j':12,'n':13,'E':14,'/':15,'f':16,'Q':17,'?':18}

y_encoded = np.array([label_mapping[value] for value in y])#will generate the number across each data accordig to the class
num_classes = 19
from keras.utils import to_categorical
y_one_hot = to_categorical(y_encoded, num_classes)#will generate the array based on the classes number assigned to it

x_train,x_test,y_train,y_test=train_test_split(x,y_encoded,test_size=0.25,random_state=42)

#%%
model=SVC(decision_function_shape='ovr',C=10)#ovr=one vs rest(all)

model.fit(x_train,y_train)
accuracy=model.score(x_test,y_test)
print(accuracy)
#%%
y_pred=model.predict(x_test)
conf_mat=confusion_matrix(y_test ,y_pred ) 

import seaborn as sn
sn.heatmap(conf_mat,annot=True)
plt.xlabel('predicted values')
plt.ylabel('true values')
print("precision score : ",precision_score(y_test , y_pred ,average='micro'))
print("sensitivity(recall) : ",recall_score(y_test , y_pred ,average='micro'))
print("F1 score: ",f1_score(y_test , y_pred ,average='micro'))

