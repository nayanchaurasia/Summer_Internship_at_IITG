import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,f1_score,precision_score,recall_score)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
###############################################################################################################################
data=pd.read_csv('C:/ecg_all_data.csv')
x=data.iloc[:,0:240]
y=data.iloc[:,240]

#LEVEL ENCODING
enc=LabelEncoder()
y_en=enc.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y_en,test_size=0.25,random_state=42)

#%%
rf=RandomForestClassifier(n_estimators=5)
rf.fit(x_train,y_train)

accuracy=rf.score(x_test,y_test)

#%%
decision_trees = rf.estimators_
# Iterate over each decision tree and extract information
for i, tree in enumerate(decision_trees):
    print(f"Decision Tree {i+1}:")
    print(f"Number of nodes: {tree.tree_.node_count}")
    print(f"Max depth: {tree.tree_.max_depth}")

print('test accuracy: ', accuracy)
y_pred=rf.predict(x_test)

conf_mat=confusion_matrix(y_test,y_pred)
#print('confusion matrix is :',conf_mat)# does not take the one hot encoded data
from sklearn.metrics import ConfusionMatrixDisplay
disp=ConfusionMatrixDisplay(conf_mat)
disp.plot()
plt.show()
print("precision score : ",precision_score(y_test, y_pred,average='micro'))
print("sensitivity(recall) : ",recall_score(y_test, y_pred,average='micro'))
print("F1 score: ",f1_score(y_test, y_pred,average='micro'))

