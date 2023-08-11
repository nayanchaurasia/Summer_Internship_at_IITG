import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import (confusion_matrix,f1_score,precision_score,recall_score)
from sklearn.model_selection import train_test_split

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

x_train,x_test,y_train,y_test=train_test_split(x,y_one_hot,test_size=0.25,random_state=0)

#%%
model=Sequential()
model.add(Dense(units=90,activation='relu',input_dim=240))
dropout=tensorflow.keras.layers.Dropout(0.3)
model.add(Dense(units=60,activation='relu'))
dropout=tensorflow.keras.layers.Dropout(0.2)
model.add(Dense(units=40,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=30,batch_size=100)

#%%
loss,accuracy=model.evaluate(x_test,y_test)
model.summary()

print('test loss: ',loss)
print('test accuracy: ', accuracy)

y_pred=model.predict(x_test)

# Convert the multi-dimensional labels to a 1D array
y_test_flat = np.argmax(y_test, axis=1)
y_pred_flat = np.argmax(y_pred, axis=1)

conf_mat=confusion_matrix(y_test_flat,y_pred_flat)#we use argmax because the conf_max 
#print('confusion matrix is :',conf_mat)# does not take the one hot encoded data
from sklearn.metrics import ConfusionMatrixDisplay
disp=ConfusionMatrixDisplay(conf_mat)
disp.plot()
plt.show()
print("precision score : ",precision_score(y_test_flat, y_pred_flat,average='micro'))
print("sensitivity(recall) : ",recall_score(y_test_flat, y_pred_flat,average='micro'))
print("F1 score: ",f1_score(y_test_flat, y_pred_flat,average='micro'))

