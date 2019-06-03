
# coding: utf-8

# In[86]:


#The Code of Titanic Sinking Disaster Assignment
#Importing the Python libraries  
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[87]:


#Step 1: Load the Train DataSet 
df=pd.read_csv("E:/Python/Data/titanic/train.csv")
df


# In[88]:


# Removing the columns which are not necessary
df=df.drop(['Name','PassengerId', 'Parch','SibSp','Ticket','Cabin','Embarked','Fare'], axis = 1) 
df.head(7)


# In[89]:


#Step 2: Data Visualization
#Checking if there any null values exist
df.isnull().sum()


# In[90]:


# Step 3: Data Cleaning process
#Fillup the NULL values With MEDIAN Method 
df = df.fillna(df.median().round(2))
df


# In[91]:


#Rechecking  all the columns
df.isnull().sum()


# In[92]:


#Load and clean the Test DataSet
df2=pd.read_csv("E:/Python/Data/titanic/test.csv")
df2


# In[93]:


## Removing the columns which are not necessary
test_data=df2.drop(['Name','PassengerId', 'Parch','SibSp','Ticket','Cabin','Embarked','Fare'], axis = 1) 
test_data.head()


# In[94]:


#Checking the null values
test_data.isnull().sum()


# In[95]:


#Fillup the NULL values With MEDIAN Method
test_data = test_data.fillna(df.median().round(2))
test_data.isnull().sum()


# In[96]:


#We will use Dummy variables to segregate subgroups for "Sex" column 
#and convert categorical values to numerical values 
df = pd.get_dummies(df,columns=['Sex'])
df.head(10)


# In[97]:


#The same method is applying for the TEST Dataset also
test_data = pd.get_dummies(test_data,columns=['Sex'])
test_data.head(10)


# In[98]:


#select the input dataset and output dataset
X = df.drop('Survived', axis=1)
y = df['Survived']


# In[56]:


#Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print(X_train.shape, X_test.shape)
print(y_train.shape,y_test.shape)


# In[99]:


#Import logistic Regression Model
model = LogisticRegression()

#fit the model on train data set
model.fit(X_train, y_train)

#Mean overall accuracy on test set 
model.score(X_test,y_test)


# In[58]:


#intercept and coefficient
model.coef_
model.intercept_


# In[59]:


#Odds Ratio
OR = np.exp(model.coef_)
pd.Series(OR[0],index=X_train.columns)


# In[18]:


#Predicted probablity for test set
probs = model.predict_proba(X_test)
np.round(probs,decimals=2)


# In[60]:


#Predicted class lebel for test set
y_pred = model.predict(X_test)
y_pred


# In[61]:


#Interpret the Model prediction using Confution matrix
mat1 = confusion_matrix(y_test,y_pred)
sns.heatmap(mat1,annot=True,cbar=False,fmt='d')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')


# In[62]:


# Now Final Prediction of Given Test Dataset using logistic Regression model
y_pred = model.predict(test_data)
y_pred


# In[63]:


# We use Decision Tree Algorithm to compare best results
DT= DecisionTreeClassifier()
DT.fit(X_train, y_train)
DT.score(X_test,y_test)


# In[65]:


#We use Random Forest Algorithm also  
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
RF.score(X_test,y_test)


# In[67]:


#Here we can see that Logistic Regression Model Gives the Best Prediction Score
#So, we'll choose Logistic Reg. for our assignment
#Now We create a DataFrame where we store the result as output
df3 = pd.DataFrame(y_pred,columns=['Survived'])
df3.head()


# In[68]:


#Take the "PassengerId" column from test dataset and put our prediction result according to it. 
final_result = df3.join(df2['PassengerId']).iloc[:,::-1]


# In[69]:


final_result


# In[33]:


#Store the result in a Excel file
final_result.to_csv('E:/Python/Data/titanic/final_submission.csv', index=False)

