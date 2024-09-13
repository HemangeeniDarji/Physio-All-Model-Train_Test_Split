#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import GridSearchCV


# In[2]:


X=pd.read_csv("X_physio_scaled.csv")


# In[3]:


X.head()


# In[4]:


y=pd.read_csv("y_physio.csv")


# In[5]:


y.head()


# In[6]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=21)


# In[7]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[31]:


classifiers = {
    "Logistic Regression": LogisticRegression(penalty='l2'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(criterion="entropy"),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel='rbf')}
    


# In[32]:


model1=classifiers["Logistic Regression"]
model2=classifiers["Decision Tree"]
model3=classifiers["Random Forest"]
model4=classifiers["K-Nearest Neighbors"]
model5=classifiers["Naive Bayes"]
model6=classifiers["Support Vector Machine"]


# In[33]:


model1.fit(X_train,y_train)


# In[34]:


y_pred_Logistic_1 = model1.predict(X_test)


# In[35]:


accuracy = accuracy_score(y_test, y_pred_Logistic_1)
print("Accuracy:", accuracy)


# In[36]:


y_pred_Logistic_2 = model1.predict(X_train)


# In[37]:


accuracy = accuracy_score(y_train, y_pred_Logistic_2)
print("Accuracy:", accuracy)


# In[38]:


X_test.head()


# In[39]:


y_test.head(10)


# In[40]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[41]:


cm = confusion_matrix(y_test, y_pred_Logistic_1)
print("Confusion matrix is \n",cm)


# In[42]:


report = classification_report(y_test, y_pred_Logistic_1)

print("Classification Report:")
print(report)


# In[21]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# In[22]:


# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_Logistic_1)

# Calculate the AUC
auc = roc_auc_score(y_test, y_pred_Logistic_1)


# In[23]:


plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt


# # ##################################################

# In[25]:


model2.fit(X_train,y_train)


# In[26]:


y_pred_Decision_1 = model2.predict(X_test)


# In[27]:


accuracy = accuracy_score(y_test, y_pred_Decision_1)
print("Accuracy:", accuracy)


# In[28]:


y_pred_Decision_2 = model2.predict(X_train)


# In[29]:


accuracy = accuracy_score(y_train, y_pred_Decision_2)
print("Accuracy:", accuracy)


# In[30]:


# Access the root node of the decision tree
root_node = model2.tree_

# Print the root node
print("Root Node:", root_node)


# In[31]:


cm1 = confusion_matrix(y_test, y_pred_Decision_1)
print("Confusion matrix is \n",cm1)


# In[32]:


report1 = classification_report(y_test,y_pred_Decision_1)

print("Classification Report:")
print(report1)


# In[ ]:





# # #######################################################

# In[33]:


model3.fit(X_train,y_train)


# In[34]:


y_pred_Random_1 = model3.predict(X_test)


# In[35]:


accuracy = accuracy_score(y_test, y_pred_Random_1)
print("Accuracy:", accuracy)


# In[36]:


y_pred_Random_2 = model3.predict(X_train)


# In[37]:


accuracy = accuracy_score(y_train, y_pred_Random_2)
print("Accuracy:", accuracy)


# In[38]:


num_trees = len(model3.estimators_)
print("Number of decision trees in the Random Forest: ", num_trees)


# In[39]:


for i, tree in enumerate(model3.estimators_):
    print("Max depth of tree ", i+1, " is ", tree.max_depth)


# In[40]:


print(model3.get_params())


# # ###################################################

# In[41]:


model4.fit(X_train,y_train)


# In[42]:


y_pred_KNeighbors_1 = model4.predict(X_test)


# In[43]:


accuracy = accuracy_score(y_test, y_pred_KNeighbors_1)
print("Accuracy:", accuracy)


# In[44]:


y_pred_KNeighbors_2 = model4.predict(X_train)


# In[45]:


accuracy = accuracy_score(y_train, y_pred_KNeighbors_2)
print("Accuracy:", accuracy)


# In[46]:


print(model4.get_params())


# In[ ]:





# # ##############################################

# In[ ]:





# In[47]:


model5.fit(X_train,y_train)


# In[48]:


y_pred_GaussianNB_1 = model5.predict(X_test)


# In[49]:


accuracy = accuracy_score(y_test, y_pred_GaussianNB_1)
print("Accuracy:", accuracy)


# In[50]:


y_pred_GaussianNB_2 = model5.predict(X_train)


# In[51]:


accuracy = accuracy_score(y_train, y_pred_GaussianNB_2)
print("Accuracy:", accuracy)


# In[52]:


model5.predict_proba(X_test[:10])


# In[53]:


X_test.head(10)


# # #############################################

# In[59]:


model6.fit(X_train,y_train)


# In[60]:


y_pred_SVC_1 = model6.predict(X_test)


# In[61]:


accuracy = accuracy_score(y_test, y_pred_SVC_1)
print("Accuracy:", accuracy)


# In[62]:


y_pred_SVC_2 = model6.predict(X_train)


# In[63]:


accuracy = accuracy_score(y_train, y_pred_SVC_2)
print("Accuracy:", accuracy)


# In[64]:


print("Value of C parameter:", model6.C)
print("Kernel function used:", model6.kernel)
print("Support vectors:", model6.support_vectors_)
print("Number of support vectors for each class:", model6.n_support_)


# In[65]:


margin_width = 2/np.linalg.norm(model6.coef_)
print("Margin width:", margin_width)


# In[ ]:





# # ###################################################

# In[ ]:




