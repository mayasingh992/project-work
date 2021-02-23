#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Basic Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Data Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Check for Outliers
from scipy.stats import zscore


# In[ ]:


#Import Fraud_Loan_Prediction.csv file
df=pd.read_csv('Fraud_Loan_Prediction.csv')
df


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.dtypes


# In[ ]:


df.info()


# Observation:
# 
# 1. Missing value in many features.
# 2. 614 row with 13 columns in this dataset.
# 3. Loan ID is a unique identifier and can be removed
# 3. All datatype of features correct with the data value of feature.
# 5. Don't seem to have any null values for Property_Area, Loan_Status, Income, Education and Loan ID.

# In[ ]:


df.isnull().sum()


# In[ ]:


# Imputing missing value for gender
# Finding the most appeard value in gender columns

df['Gender'].value_counts()


# In[ ]:


df['Gender'].fillna('Male',inplace=True)


# In[ ]:


# Imputing missing value for Married
# Finding the most appeard value in Married columns

df['Married'].value_counts()


# In[ ]:


df['Married'].fillna('Yes',inplace=True)


# In[ ]:


# Imputing missing value for dependents
# Finding the most appeard value in dependents columns

df['Dependents'].value_counts()


# In[ ]:


df['Dependents'].fillna('0',inplace=True)


# In[ ]:


# Imputing missing value for self_emplyed
# Finding the most appeard value in self_employed columns

df['Self_Employed'].value_counts()


# In[ ]:


df['Self_Employed'].fillna('No',inplace=True)


# In[ ]:


# Imputing missing values for loanAmount
# Strategy -mean

df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)


# In[ ]:


# Imputing missing values for Loan_Amount_Term
# Strategy -mean

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)


# In[ ]:


# Imputing missing values for Credit_History
# Strategy -mean

df['Credit_History'].fillna(df['Credit_History'].mean(),inplace=True)


# # Check missing value

# In[ ]:


df.isnull().sum()


# In[ ]:


sns.heatmap(df.isnull())
plt.show()


# # Summary Statistics of Numerical Variable

# In[ ]:


df.describe()


# Observation:
# 
# 1. Mean is greater than 50% or eual to 50% means all numerical features are right skewd.
# 2. LoanAmount min value 9 and max 700.
# 3. difference between 75% and max very large in ApplicantIncome according to dtandard deviation means oultiers present.

# In[ ]:


#check coorelation
df.corr()


# In[ ]:


sns.heatmap(df.corr(),cmap='rocket',annot=True)


# Observation:
# 
# 1. Loan amount negatively correlated with only credit History.
# 2. credit_History positive correlated with Loan_Amount_Term.
# 3. LoanAmount is higly positive coorelated with AplicantIncome.

# # Data Visualization

# In[ ]:


sns.countplot(x='Loan_Status',data=df,hue='Loan_Status')


# Finding: Present data set approved loan is more than not approved.

# In[ ]:


sns.countplot(x='Gender', data=df,hue='Loan_Status')


# Observation:
# 
# 1. % of loan approved for male candidate is more than female

# In[ ]:


sns.countplot(x='Married', data=df,hue='Loan_Status')


# Finding:
# 
# 1. Apply for loan married person is more than unmarried.
# 2. % of Approvel of loan for married person is more than unmarried person.

# In[ ]:


sns.countplot(x='Dependents', data=df,hue='Loan_Status')


# In[ ]:


sns.countplot(x='Education', data=df,hue='Loan_Status')


# Finding:
# 
# 1. Graduate person apply for loan is more than Not graduate
# 2. % of approvel of loan for graduate more than 50 % according to total apply for loan.

# In[ ]:


sns.countplot(x='Self_Employed', data=df,hue='Self_Employed')


# Finding: in present dataset self empolyed number less than 100.

# In[ ]:


sns.countplot(x='Self_Employed', data=df,hue='Loan_Status')


# Observation:
# 
# 1. Self_Employed person apply for loan very less tha employed person.
# 2. Approvel of loan for employed person more than 50 %

# In[ ]:


sns.countplot(x='Property_Area', data=df)


# Finding:
# 
# 1. Semiurban area appy for loan is maximum in this dataset.
# 2. Apply for loan in rural area minimum in this dataset.
# 3. Difference betrween the apply for loan in different propery area is not more mean data balace according to area of property.

# In[ ]:


sns.countplot(x='Property_Area', data=df,hue='Loan_Status')


# Observation:
# 
# 1. % of approvel loan for urban and semiurban area is better and % of approvelloan in rural area aslo good.

# In[ ]:


sns.catplot(x='Loan_Amount_Term',y='LoanAmount',data=df,kind='bar',aspect=4)
plt.show()


# # Data Visualization of numeric value feature.

# In[ ]:


df.hist(figsize=(12,12))
plt.show()


# Observation:
# 
# 1. Maximum ApplicantIncome less than 10000 more than 10000 seems like outliers
# 2. Maximum CoapplicantIncome less than 10000 seems like outliers.

# In[ ]:


(df['CoapplicantIncome']>10000).value_counts()


# In[ ]:


(df['ApplicantIncome']>10000).value_counts()


# In[ ]:


(df['LoanAmount']>300).value_counts()


# In[ ]:


sns.pairplot(df)


# Check Skewness

# In[ ]:


df.skew()


# In[ ]:


import numpy as np
df.skew()
for col in df.skew().index:
    if col in df.describe().columns:
        if df.skew().loc[col]>0.55:
            df[col]=np.log1p(df[col])
        if df.skew().loc[col]<-0.55:
            df[col]=np.log1p(df[col])


# In[ ]:


#after removing skewness again check
df.skew()


# In[ ]:


df.head()


# In[ ]:


# Drop Loan_ID coumns its not contribute the outcome feature Loan_status
df.drop(columns=['Loan_ID'], axis=1, inplace=True)
df.columns


# # Plooting outliers

# In[ ]:


for i in df.describe().columns:
    sns.boxplot(df[i].dropna())
    plt.show()


# In[ ]:


#seprate the categorical columns and numerical columns
cat_df,num_df=[],[]
for i in df.columns:
    if df[i].dtype=='O':
        cat_df.append(i)
    else:
        num_df.append(i)
print('cat_df >>> \n',cat_df,'\nnum_df >>> \n',num_df)


# Removing Outliers

# In[ ]:


from scipy.stats import zscore
z=np.abs(zscore(df[num_df]))
z


# In[ ]:


#consider threshold = 3
print(np.where(z>3))


# In[ ]:


df=df[(z<3).all(axis=1)]


# In[ ]:


df.shape


# In[ ]:


# Splitting x and y
x=df.drop('Loan_Status',axis=1)
y=df['Loan_Status']
y.replace({'N':0,'Y':1},inplace=True)


# In[ ]:


x=pd.get_dummies(x)
x.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=ss.fit_transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


# In[ ]:


print(x_train.shape,x_test.shape)


# In[ ]:


print(y_train.shape,y_test.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier


# In[ ]:


from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn.model_selection import GridSearchCV,cross_val_score


# In[ ]:


#KNN
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
neighbors={'n_neighbors':range(1,30)}
bknn=GridSearchCV(knn,neighbors)
bknn.fit(x_train,y_train)
bknn.best_params_


# In[ ]:


KNN=KNeighborsClassifier(n_neighbors=11)


# In[ ]:


#Support Vector Classifier
from sklearn.svm import SVC

svc=SVC()
svc_parameters={'kernel':['linear','sigmoid','poly','rbf'],'C':[1,10]}
bsvc=GridSearchCV(svc,svc_parameters)
bsvc.fit(x_train,y_train)
bsvc.best_params_


# In[ ]:


SV=SVC(kernel='linear',C=1)


# In[ ]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

criterion = {'criterion':['gini','entropy']}
dtc=DecisionTreeClassifier(random_state=42)
bdtc=GridSearchCV(dtc,criterion)
bdtc.fit(x_train,y_train)
bdtc.best_params_


# In[ ]:


DTC=DecisionTreeClassifier(criterion='gini',random_state=42)


# In[ ]:


#Random Forest Classifer
from sklearn.ensemble import RandomForestClassifier

parameters = {'n_estimators':range(100,200,300)}
rf=RandomForestClassifier(random_state=42)
brf=GridSearchCV(rf,parameters)
brf.fit(x_train,y_train)
brf.best_params_


# In[ ]:


RFC=RandomForestClassifier(n_estimators=100,random_state=42)


# In[ ]:


#Gradient Boosting Classifier

GBC=GradientBoostingClassifier(n_estimators=250)


# In[ ]:


#Extra Trees Classifier

ETC=ExtraTreesClassifier(n_estimators=250)


# In[ ]:


#AdaBoost Classifier

ABC=AdaBoostClassifier(n_estimators=50)


# In[ ]:


#Bagging Classifier

BC=BaggingClassifier(n_estimators=250)


# In[ ]:


KNN=KNeighborsClassifier(n_neighbors=11)
SV=SVC(kernel='linear',C=1)
LR=LogisticRegression()
DT=DecisionTreeClassifier(criterion='gini',random_state=42)
GNB=GaussianNB()
RFC=RandomForestClassifier(n_estimators=100,random_state=42)
GBC=GradientBoostingClassifier(n_estimators=250)
ETC=ExtraTreesClassifier(n_estimators=250)
ABC=AdaBoostClassifier(n_estimators=50)
BC=BaggingClassifier(n_estimators=250)


# In[ ]:


models=[]
models.append(('KNeighborsClassifier',KNN))
models.append(('SVC',SV))
models.append(('LogisticRegression',LR))
models.append(('DecisionTreeClassifier',DT))
models.append(('GaussianNB',GNB))
models.append(('RandomForestClassifier',RFC))
models.append(('GradientBoostingClassifier',GBC))
models.append(('ExtraTreesClassifier',ETC))
models.append(('AdaBoostClassifier',ABC))
models.append(('BaggingClassifier',BC))


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc


# In[ ]:


Model=[]
score=[]
CVS=[]
rocscore=[]
for name,model in models:
    print('************',name,'**********')
    print('\n')
    Model.append(name)
    model.fit(x_train,y_train)
    print(model)
    pre=model.predict(x_test)
    print('\n')
    AS=accuracy_score(y_test,pre)
    print('Accuracy_score=',AS)
    score.append(AS*100)
    print('\n')
    sc=cross_val_score(model,x,y,cv=10,scoring='accuracy').mean()
    print('Cross_Val_Score=',sc)
    CVS.append(sc*100)
    print('\n')
    false_positive_rate,true_positive_rate,threshold=roc_curve(y_test,pre)
    roc_auc= auc(false_positive_rate,true_positive_rate)
    print('roc_auc_score=',roc_auc)
    rocscore.append(roc_auc*100)
    print('\n')
    print('classification_report\n',classification_report(y_test,pre))
    print('\n')
    cm=confusion_matrix(y_test,pre)
    print(cm)
    print('\n')
    plt.figure(figsize=(10,40))
    plt.subplot(911)
    plt.title(name)
    plt.plot(false_positive_rate,true_positive_rate,label='AUC = %0.2f'% roc_auc)
    plt.plot([0,1],[0,1],'r--')
    plt.legend(loc='lower right')
    plt.ylabel('True positive Rate')
    plt.xlabel('False Positive Rate')
    print('\n\n')


# In[ ]:


result = pd.DataFrame({'Model':Model,'Accuracy_score': score,'cross_val_score':CVS,'Roc_auc_curve':rocscore})
result


# In[ ]:


#save best model

import joblib
from joblib import dump #from joblib import load > to load .pkl file
joblib.dump(SVC,'SVC_Heart _Disease.pkl')


# In[ ]:


['SVC_Heart _Disease.pkl']

