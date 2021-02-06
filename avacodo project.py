#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import pyplot

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import warnings                                                                 
warnings.filterwarnings('ignore') 

# allow plots to appear directly in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


av= pd.read_csv('avocado.ipynb.csv',parse_dates=['Date'])
av


# In[ ]:


av.info()


# In[ ]:


av.dropna(axis = 0, how = 'all', inplace = True)
av


# In[ ]:


av.drop('Unnamed: 0', axis=1,inplace = True)
av


# In[ ]:


av.isnull().sum()


# In[ ]:


# change the data type of type and region
av['type']=av['type'].astype('category')
av['region']=av['region'].astype('category')


# In[ ]:


#check again
av.info()


# In[ ]:


av.isnull().sum()


# In[ ]:


av.columns


# In[ ]:


av.apply(lambda x : len(x.unique()))


# In[ ]:


for col in av.columns:
    if av[col].dtype=="object":
        print("column name is: {} and number of distinct values: {}".format(col,len(av[col].value_counts())))
        print()


# In[ ]:


av["type"].unique()


# In[ ]:


#type columnn has only one value throughout the dataset so it will not help us in any way so lets drop
av.drop(columns=["type"],inplace=True)


# In[ ]:


av.year.unique()


# In[ ]:


av.columns


# In[ ]:


av.shape


# In[ ]:


import seaborn as sns


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


sns.heatmap(av.corr(),vmax = 1, vmin = 0,annot = True)
plt.show()


# # Summary Statistics of Numerical feature

# In[ ]:


av.describe()


# # Observation:
# 
# 1. Min,25%and 50% value of XLarge Bags is 0 and difference between 75% and max very large.
# 2. Acoording to above 1 statment skewness and oultliers present in XLarge Bags.
# 3. In Year feature only two years 2015 and 2016 present.

# In[ ]:


#check correlation
av.corr()


# In[ ]:


datayear = []
for i in av.year.unique():
    datayear.append(av[av.year == i].AveragePrice)
plt.boxplot(datayear)
plt.xticks(range(1,av.year.nunique()+1),av.year.unique())
plt.show()


# In[ ]:


av.AveragePrice.plot(kind = 'hist',bins = 50,figsize = (10,10))
plt.xlabel("Average Prices")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


av_2015=av[av.year==2015]
av_2016=av[av.year==2016]
price_2015=[i for i in av_2015.AveragePrice if i > av_2015.AveragePrice.mean()]
price_2016=[i for i in av_2016.AveragePrice if i > av_2016.AveragePrice.mean()]
print("prices which are higher than average prices in year 2015 :",len(price_2015))
print("prices which are higher than average prices in year 2016 :",len(price_2016))


# In[ ]:


for i in av:
    print(i,'\n',av[i].unique(),'\n\n')


# In[ ]:


import warnings


# In[ ]:


warnings.filterwarnings('ignore')


# In[ ]:


# From dates we can get mothly ,daily,yearly average counts of AveragePrice,type,region which can help in data analysis
av_dates=pd.DataFrame()
av_dates['month']=av['Date'].dt.month_name()
av_dates['year']=av['Date'].dt.year
av_dates['day']=av['Date'].dt.day_name()
av_dates['AveragePrice']=av['AveragePrice']
av_dates['region']=av['region']
av_dates['type']=av['type']
av_dates


# In[ ]:


# averageprice during month of an year
plt.figure(figsize=(12,12))
sns.barplot(x='month',y='AveragePrice',data=av_dates)
plt.show()


# In[ ]:


plt.figure(figsize=(18,10))
sns.lineplot(x="month", y="AveragePrice", hue='type', data=av_dates)
plt.show()


# In[ ]:


# average confirmed during day of week 
sns.barplot(x='day',y='AveragePrice',data=av_dates)
plt.show()


# In[ ]:


# averageprice during  of an year 
sns.barplot(x='year',y='AveragePrice',data=av_dates)
plt.show()


# In[ ]:


#Sorting Average Prices by states
avsorted=av.groupby("region").mean()
avsorted=avsorted.sort_values("AveragePrice",ascending=False)


# In[ ]:


#visualization
plt.figure(figsize=(17,12))
sns.barplot(x=avsorted.index,y=avsorted.AveragePrice)
plt.xticks(rotation= 90)
plt.xlabel('regions')
plt.ylim(0,1.8)
plt.ylabel('Average Prices')
plt.title('Average Prices by regions')


# In[ ]:


#Price is always important
plt.figure(figsize=(9,5))
plt.title("Distribution Price")
ax = sns.distplot(av["AveragePrice"], color = 'r')


# In[ ]:


sns.boxplot(y="type", x="AveragePrice", data=av, palette = 'pink')


# In[ ]:


plt.figure(figsize=(12,20))
sns.set_style('whitegrid')
sns.pointplot(x='AveragePrice',y='region',data=av, hue='year',join=False)
plt.xticks(np.linspace(1,2,5))
plt.ylabel('Region',{'fontsize' : 'large'})
plt.xlabel('AveragePrice',{'fontsize':'large'})
plt.title("Yearly Average Price in Each Region",{'fontsize':20})


# # Predicting Average Price of Avocado

# In[ ]:


import pandas as pd


# In[ ]:


av= pd.read_csv('avocado.ipynb.csv',parse_dates=['Date'])
av


# In[ ]:


av_m=av[[ 'AveragePrice','Total Volume', '4046', '4225', '4770',
       'Total Bags', 'Small Bags', 'Large Bags','XLarge Bags']]
av_m


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler().fit(av_m)
av_s = scaler.transform(av_m)
av_avocado = pd.DataFrame(av_s)
av_avocado.columns = ['AveragePrice','Total Volume', '4046', '4225', '4770',
       'Total Bags', 'Small Bags', 'Large Bags','XLarge Bags']
av_avocado


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


Bags = av[['Small Bags', 'Large Bags']].groupby(av.region).sum()
Bags.plot(kind='line', fontsize = 14,figsize=(14,8))
plt.show()


# In[ ]:


sns.pairplot(av.iloc[:,5:8])


# # Check skewness distribution

# In[ ]:


av.skew()


# In[ ]:


#Distribution plot for all numeric columns including target columns
#In for loop we give describe() function to descriminate numeric columns from categorical columns.
#Because describe() function give the summary of numeric columns
for i in av.iloc[:,0:9]:
    sns.distplot(df[i])
    plt.show()


# In[ ]:


#Lets treat the skewness
import numpy as np
av.skew()
for col in av.skew().index:
    if col in av.describe().columns:
        if av.skew().loc[col]>0.5:
            av[col]=np.sqrt(av[col])
        if av.skew().loc[col]<-0.5:
            av[col]=np.cbrt(av[col])


# In[ ]:


#Again check skewness
av.skew()


# In[ ]:


for i in av.describe().columns:
    sns.boxplot(av[i])
    plt.show()


# In[ ]:


#make copy
av1=av.copy()
av2=av.copy()
av3=av.copy()


# # Removing outliers

# In[ ]:


av1.head()


# In[ ]:


av1.shape


# In[ ]:


from scipy.stats import zscore
z=np.abs(zscore(av1.describe()))
z


# In[ ]:


threshold=3
print(np.where(z>3))


# Finding: No any outliers detect.

# In[ ]:


av1.head()


# # Machine Learning Algorithm for Linear Regression

# In[ ]:


#Splitting x and y for AveragePrice
av1_x=av.iloc[:,1:]
y=av.iloc[:,:1]


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
av1_x['type']=le.fit_transform(av1_x['type'])


# In[ ]:


av1_x['region'].nunique()


# In[ ]:


re=pd.get_dummies(av1['region']) #Dummies for 'region'
av1_x=pd.concat([av1_x,re],axis=1) #Concatenating the dummies with original data
av1_x.drop(columns=['region'],inplace=True) #Dropping 'region'


# In[ ]:


#Scaling the data
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=ss.fit_transform(av1_x)


# In[ ]:


# Model Training and Validation
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# ML Algorithms
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Model Export
import joblib
from joblib import dump #from joblib import load > to load .pkl file


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print('x_train.shape: ',x_train.shape,'x_test.shape: ',x_test.shape,'\ny_train.shape',y_train.shape,'y_test.shape',y_test.shape)


# In[ ]:


#Creating function for Model Training
from sklearn.metrics import mean_squared_error, r2_score
def models(model, x_train, x_test, y_train, y_test,score,rmse):
    #Fit the algorithm on the data
    model.fit(x_train, y_train)
    
    #Predict training set:
    y_pred = model.predict(x_test)
    
    score.append(model.score(x_train,y_train))
    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    print('Score:',model.score(x_train,y_train))
    print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R2 Score:',r2_score(y_test, y_pred))


# In[ ]:


model_name,score,rmse=[],[],[]


# In[ ]:


#Linear Regression
from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
mod='Linear Regression'
print('Model Report for', mod)
models(lreg,x_train,x_test,y_train,y_test,score,rmse)
model_name.append(mod)


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


#Ridge Regression
from sklearn.linear_model import Ridge
parameters={'alpha':[0.01,1,100]}
best=GridSearchCV(Ridge(),parameters)
best.fit(x_train,y_train)
best.best_params_


# In[ ]:


rr=Ridge(alpha=0.01)
mod='Ridge Regression'
print('Model Report for', mod)
models(rr,x_train,x_test,y_train,y_test,score,rmse)
model_name.append(mod)


# In[ ]:


#Lasso Regression
from sklearn.linear_model import Lasso
parameters={'alpha':[0.001,0.01,1]}
best=GridSearchCV(Lasso(),parameters)
best.fit(x_train,y_train)
best.best_params_


# In[ ]:


lr=Lasso(alpha=0.001)
mod='Lasso Regression'
print('Model Report for', mod)
models(lr,x_train,x_test,y_train,y_test,score,rmse)
model_name.append(mod)


# In[ ]:


from sklearn.linear_model import ElasticNet
enr=ElasticNet(alpha=0.01)
mod='Elastic Net'
print('Model Report for', mod)
models(enr,x_train, x_test, y_train, y_test,score,rmse)
model_name.append(mod)


# In[ ]:


#Support Vector Regression
from sklearn.svm import SVR
parameters={'kernel':['linear','poly','rbf']}
best=GridSearchCV(SVR(),parameters)
best.fit(x_train,y_train)
best.best_params_


# In[ ]:


svr=SVR(kernel='rbf')
mod='Support Vector Regression'
print('Model Report for', mod)
models(svr,x_train, x_test, y_train, y_test,score,rmse)
model_name.append(mod)


# In[ ]:


#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
parameters={'max_depth':[8,10,12,15],'min_samples_leaf':[100,150]}
best=GridSearchCV(DecisionTreeRegressor(),parameters)
best.fit(x_train,y_train)
best.best_params_


# In[ ]:


dct=DecisionTreeRegressor(max_depth=10,min_samples_leaf=100)
mod='Decision Tree Regression'
print('Model Report for', mod)
models(dct,x_train, x_test, y_train, y_test,score,rmse)
model_name.append(mod)


# In[ ]:


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
parameters={'n_estimators':[200,300,400],'max_depth':[5,6]}
best=GridSearchCV(RandomForestRegressor(),parameters)
best.fit(x_train,y_train)
best.best_params_


# In[ ]:


rf=RandomForestRegressor(n_estimators=200,max_depth=6)
mod='Random Forest Regression'
print('Model Report for', mod)
models(rf,x_train, x_test, y_train, y_test,score,rmse)
model_name.append(mod)


# In[ ]:


final=pd.DataFrame({'Model Name':model_name,'Score':score,'RMSE':rmse})
final


# conclusions:
# 
# 1. For Linear Regression when output coulmns is AveragePrice Models score findout.
# 2. For AveragePrice Random Forest Regression give best score and poor performance by DTR.
# 3. Random Forest Regression gives 81% score for linear regression.

# In[ ]:


#Export best Model
import joblib
from joblib import dump
joblib.dump(rf,'RFR_Avocado_Dataset.pkl')


# # Machine Learning Algorithm for Classification

# In[ ]:


#Splitting x and y for Region output
av2.head(5)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
av2['type']=le.fit_transform(av2['type'])
av2.head(5)


# In[ ]:


x=av2.drop(columns=['region'])
y=av2['region']


# In[ ]:


2.region.describe()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
y


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=ss.fit_transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split,cross_val_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)
print(x_train.shape,x_test.shape,'\n',y_train.shape,y_test.shape)


# In[ ]:


from sklearn.metrics import classification_report,accuracy_score,roc_auc_score,plot_confusion_matrix,confusion_matrix
from sklearn.model_selection import GridSearchCV


# In[ ]:


#ROC_AUC only handles binary 0,1 values. Using LabelBinarizer to convert y_test and y_pred
from sklearn.preprocessing import LabelBinarizer


# In[ ]:


def multiclass_roc_auc_score(y_test,y_pred):
    lb=LabelBinarizer()
    y_test_new=lb.fit_transform(y_test)
    y_pred_new=lb.fit_transform(y_pred)
    return round(roc_auc_score(y_test_new,y_pred_new)*100,2)


# In[ ]:


#SVC
from sklearn.svm import SVC

svc=SVC()
svc_parameters={'kernel':['linear','sigmoid','poly','rbf'],'C':[1,10]}
bsvc=GridSearchCV(svc,svc_parameters)
bsvc.fit(x_train,y_train)
bsvc.best_params_


# In[ ]:


svc=SVC(kernel='linear',C=10)
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
svc_score=round(accuracy_score(y_test,y_pred)*100,2)
print('svc_score:',svc_score,'%\nmulticlass_roc_auc_score:',multiclass_roc_auc_score(y_test,y_pred),'%')
plt.figure(figsize=(20,12))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)


# In[ ]:


#KNN
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()
neighbors={'n_neighbors':range(1,30)}
bknn=GridSearchCV(knn,neighbors)
bknn.fit(x_train,y_train)
bknn.best_params_


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
knn_score=round(accuracy_score(y_test,y_pred)*100,2)
print('knn_score:',knn_score,'%\nmulticlass_roc_auc_score:',multiclass_roc_auc_score(y_test,y_pred),'%')
plt.figure(figsize=(20,12))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)


# In[ ]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

criterion = {'criterion':['gini','entropy']}
dtc=DecisionTreeClassifier(random_state=42)
bdtc=GridSearchCV(dtc,criterion)
bdtc.fit(x_train,y_train)
bdtc.best_params_


# In[ ]:


dtc=DecisionTreeClassifier(criterion='entropy',random_state=42)
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)
dtc_score=round(accuracy_score(y_test,y_pred)*100,2)
print('dtc_score:',dtc_score,'%\nmulticlass_roc_auc_score:',multiclass_roc_auc_score(y_test,y_pred),'%')
plt.figure(figsize=(20,12))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)


# In[ ]:


#Random Forest Classifer
from sklearn.ensemble import RandomForestClassifier

parameters = {'n_estimators':range(100,200,300)}
rf=RandomForestClassifier(random_state=42)
brf=GridSearchCV(rf,parameters)
brf.fit(x_train,y_train)
brf.best_params_


# In[ ]:


rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
rf_score=round(accuracy_score(y_test,y_pred)*100,2)
print('rf_score:',rf_score,'%\nmulticlass_roc_auc_score:',multiclass_roc_auc_score(y_test,y_pred),'%')
plt.figure(figsize=(20,12))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)


# In[ ]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier(n_estimators=250)
gbc.fit(x_train,y_train)
y_pred=gbc.predict(x_test)
gbc_score=round(accuracy_score(y_test,y_pred)*100,2)
print('gbc_score:',gbc_score,'%\nmulticlass_roc_auc_score:',multiclass_roc_auc_score(y_test,y_pred),'%')
plt.figure(figsize=(20,12))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)


# In[ ]:


#Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

etc=ExtraTreesClassifier(n_estimators=250)
etc.fit(x_train,y_train)
y_pred=etc.predict(x_test)
etc_score=round(accuracy_score(y_test,y_pred)*100,2)
print('etc_score:',etc_score,'%\nmulticlass_roc_auc_score:',multiclass_roc_auc_score(y_test,y_pred),'%')
plt.figure(figsize=(20,12))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)


# In[ ]:


#AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(n_estimators=100)
ada.fit(x_train,y_train)
y_pred=ada.predict(x_test)
ada_score=round(accuracy_score(y_test,y_pred)*100,2)
print('ada_score:',ada_score,'%\nmulticlass_roc_auc_score:',multiclass_roc_auc_score(y_test,y_pred),'%')
plt.figure(figsize=(20,12))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)


# In[ ]:


#Bagging Classifier
from sklearn.ensemble import BaggingClassifier

bc=BaggingClassifier(n_estimators=250)
bc.fit(x_train,y_train)
y_pred=bc.predict(x_test)
bc_score=round(accuracy_score(y_test,y_pred)*100,2)
print('bc_score:',bc_score,'%\nmulticlass_roc_auc_score:',multiclass_roc_auc_score(y_test,y_pred),'%')
plt.figure(figsize=(20,12))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True)


# In[ ]:



models={
    'SVC': svc,
    'KNeighborsClassifier': knn,
    'DecisionTreeClassifier': dtc,
    'RandomForestClassifier': rf,
    'GradientBoostingClassifier': gbc,
    'ExtraTreesClassifier': etc,
    'AdaBoostClassifier': ada,
    'BaggingClassifier': bc,
}


# In[ ]:


models_name,ascore,roc=[],[],[]
for name in models.keys():
    models[name].fit(x_train,y_train)
    models_name.append(name)
    y_pred=models[name].predict(x_test)
    ascore.append(round(accuracy_score(y_test,y_pred)*100,2))
    roc.append(multiclass_roc_auc_score(y_test,y_pred))
final=pd.DataFrame({'Model':models_name,'Accuracy Score':ascore,'ROC_AUC_Score':roc}).sort_values(by='Accuracy Score',ascending=False)
final


# In[ ]:


# Conclusion:

1. For classification when output columns is 'region' different models score findout.
2. For 'region' output best accuracy score achive by ExtraTreesClassifier.
3. ExtraTreesClassifier give 97% accuracy


# In[ ]:


import joblib
from joblib import dump #from joblib import load > to load .pkl file
joblib.dump(etc,'ETC_Avocado.pkl')


# In[ ]:




