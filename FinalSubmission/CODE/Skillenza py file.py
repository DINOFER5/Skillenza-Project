#!/usr/bin/env python
# coding: utf-8

# # Code

# ###### Libraries needed to run this code are imported here

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier,XGBRegressor
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier,VotingClassifier,RandomForestRegressor
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA


# ###### Importing the train and test Datasets

# In[4]:


train=pd.read_csv('C:/PGA10/Projects/Skillenza/TRAINING.csv')
test=pd.read_csv('C:/PGA10/Projects/Skillenza/TEST.csv')


# ###### Saving 'Grade' from train and 'id' from test datasets for later use

# In[5]:


train_grade=train['Grade']
test_id=test['id']


# ###### Dropping 'Grade' from train in order to concatenate the two into a single dataset df

# In[6]:


train=train.drop('Grade',axis=1)

df=pd.concat([train,test],axis=0,ignore_index=True)


# In[7]:


df.head()


# ###### Dropping 'id' column from df

# In[8]:


df=df.drop('id',axis=1)


#  Removing the '$' symbol at the end of the values in 'EXPECTED' column and converting it into int type

# In[9]:


df['EXPECTED']=df['EXPECTED'].str.slice(0, -1, 1)

df['EXPECTED']=df['EXPECTED'].astype(int)


# In[10]:


df.head()


# ###### Testing for missing values and outliers in the data

# In[11]:


df.describe()


# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[47]:


rcParams['figure.figsize']=10,10
df.boxplot()
plt.show()


# ###### Only 'EXPECTED' has outliers and has no missing values so fix it first by log transformation to scale it down

# In[48]:


df.boxplot(column='EXPECTED')


# In[14]:


df['EXPECTED']=np.log1p(df['EXPECTED'])


# In[52]:


df.boxplot(column='EXPECTED')
plt.show()


# ###### 'roof' has repeaded levels due to different case 'no' , 'NO' and 'yes,'YES . 
# ###### Consolidating this:

# In[15]:


df['roof'].value_counts()


# In[16]:


df['roof'][df['roof']=='yes']='YES'
df['roof'][df['roof']=='no']='NO'
df['roof'].value_counts()


# ###### Imputing NaN values of columns with very less missing values with mean:

# In[17]:


df.isnull().sum()


# In[18]:


df['Troom']=df['Troom'].fillna(df['Troom'].mean())
df['Nbedrooms']=df['Nbedrooms'].fillna(df['Nbedrooms'].mean())
df['Nbwashrooms']=df['Nbwashrooms'].fillna(df['Nbwashrooms'].mean())
df['Twashrooms']=df['Twashrooms'].fillna(df['Twashrooms'].mean())
df['Lawn(Area)']=df['Lawn(Area)'].fillna(df['Lawn(Area)'].mean())
df['API']=df['API'].fillna(df['API'].mean())
df.info()


# In[58]:


df.isnull().sum()


# ###### Since 'roof' and 'Roof(Area)' have large number of missing values , we will impute these by building predictive models.

# ###### Fixing multicolinearity before building the models

# In[20]:


plt.figure(figsize=(14,22))
data=df.drop(['roof','Roof(Area)'],axis=1)
data=data.iloc[:,0:11]
sns.heatmap(data.astype(float).corr(),square=True,linecolor='white',annot=True)
plt.show()


# In[21]:


vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
vif["Columns"] = data.columns


# In[22]:


vif.round(2)


# In[23]:


data.head()


# #### Scaling and performing PCA to remove multicolinearity among IDVs

# In[24]:


scd=StandardScaler()
scaled_data=scd.fit_transform(data)


# In[37]:


cov_mat=np.cov(scaled_data.T)


# In[43]:


Eigen_val,Eigen_vec=np.linalg.eig(cov_matmat)


# In[56]:


Eigen_pair=list(zip(np.abs(Eingen_val),Eigen_vec))


# In[60]:


Eigen_pair.sort(key=lambda x: x[0], reverse=True)


# In[61]:


Eigen_pair


# In[91]:


pd.DataFrame(Eigen_val,data.columns).sort_values(0,ascending=False)


# In[62]:


for i in Eigen_pair:
    print(i[0])


# In[63]:


tot = sum(Eigen_val)
var_exp = [(i / tot)*100 for i in sorted(Eigen_val, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# In[77]:


print(var_exp)
cum_var_exp


# In[86]:


rcParams['figure.figsize']=15,10
plt.bar(range(10), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
plt.step(range(10), cum_var_exp, where='mid',
             label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()


# ##### Thus, 7 or 8 principal components would be enough as the eigen values are too low for the others and the explain very little variability of the data

# In[78]:


pca_data = PCA(n_components=8)
principalComponents = pca_data.fit_transform(scaled_data)


# In[79]:


pdf=pd.DataFrame(principalComponents,columns = ['P1', 'P2','P3','P4','P5','P6','P7','P8'])


# In[80]:


pdf.head()


# In[81]:


vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(pdf.values, i) for i in range(pdf.shape[1])]
vif["Column"] = pdf.columns
vif


# ###### Rejoining the dropped columns to new dataframe , pdf

# In[82]:


pdf['roof']=df['roof']

pdf['Roof(Area)']=df['Roof(Area)']


# In[83]:


pdf.head()


# ###### Below, we take observations with non missing values to build a model and use it to predict the values for the entire data.
# ###### We then impute this value into columns with missing values. 
# ###### We first perform this by dropping 'Roof(Area)' because the same observations are missing in both the columns.
# ###### We use a voting classifier built from 5 other models to do this

# In[88]:


nomissing=pdf.drop(['Roof(Area)'],axis=1)
nomissing=nomissing.dropna()

nmX=nomissing.drop('roof',axis=1)
nmY=nomissing['roof']

nmX_train,nmX_test,nmY_train,nmY_test=train_test_split(nmX,nmY,test_size=.25,random_state=42)
lr=LogisticRegression()
xg=XGBClassifier()
svm=SVC()
rf=RandomForestClassifier()
nb=GaussianNB()
vclassifier=VotingClassifier(estimators=[('lr',lr),('xg',xg),('svm',svm),('rf',rf),('nb',nb)],voting='hard')


# In[89]:


mX=pdf.drop(['roof','Roof(Area)'],axis=1)
mY=pdf['roof']


vclassifier.fit(nmX,nmY)
Y_pred=vclassifier.predict(mX)

pdf['roof_impute']=Y_pred

pdf['roof']=pdf['roof'].fillna(pdf['roof_impute'])


# ###### If Roof has a value 'NO' then 'Roof(Area)' will obviosly be 0, we implement this idea here

# In[90]:


pdf['Roof(Area)'][pdf['roof']=='NO']=0


# Label encoding Roof in order to build model to predict 'Roof(Area)'

# In[91]:


labelencoder_X_1=LabelEncoder()
pdf["roof"] = labelencoder_X_1.fit_transform(pdf["roof"])


# In[92]:


pdf=pdf.drop(['roof_impute'],axis=1)

pdf.head()


# In[93]:


complete=pdf.dropna()
cX=complete.drop('Roof(Area)',axis=1)
cY=complete['Roof(Area)']


# ###### We use a XGBRegressor to predict the values of 'Roof(Area)'

# In[94]:


imp=XGBRegressor()
imp.fit(cX,cY)


# In[95]:


ncX=pdf.drop('Roof(Area)',axis=1)

imp_pred=imp.predict(ncX)


# In[96]:


pdf['roof_area_impute']=imp_pred

pdf['Roof(Area)']=pdf['Roof(Area)'].fillna(pdf['roof_area_impute'])

pdf=pdf.drop("roof_area_impute",axis=1)


# #### Test for data consistency and skewness before building our final model

# In[97]:


rcParams['figure.figsize']=10,10
pdf.hist()
plt.show()


# In[98]:


pd.DataFrame(pdf.columns,np.abs(skew(pdf)))


# In[99]:


pdf.info()


# In[100]:


pdf.corr()


# In[101]:


plt.figure(figsize=(14,22))
datax=pdf.iloc[:,0:12]
sns.heatmap(datax.astype(float).corr(),square=True,linecolor='white',annot=True)
plt.show()


# In[99]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(pdf.values, i) for i in range(pdf.shape[1])]
vif["features"] = pdf.columns
vif


# ##### we drop roof as this variability can be explained by 'Roof(Area)'

# In[102]:


pdf=pdf.drop('roof',axis=1)


# In[103]:


pdf.head()


# ##### Scaling Roof along with other components and storing in new dataframe, sdf

# In[105]:


sc=StandardScaler()
scaled=sc.fit_transform(pdf)


# In[106]:


sdf=pd.DataFrame(scaled,index=pdf.index, columns=pdf.columns)


# In[107]:


sdf.head()


# ### Train/Test re-split:

# In[108]:


X_train=sdf.head(7000)
Y_train=train_grade
X_test=sdf.tail(3299)


# # Trying out various models and choosing the best by cross validation

# #### We take tonly our Train data for this purpose

# In[131]:


scores_list=[] #to store score of each model
model_list=[] #to score name of each model


# ### Logistic Regression

# In[150]:


lr=LogisticRegression(solver='lbfgs') #auto selects multinomial if solver is lbfgs


# In[151]:


cv_lr=cross_val_score(lr, X_train,Y_train, cv=10)


# In[152]:


print(cv_lr.mean())


# In[153]:


scores_list.append(cv_lr.mean())
model_list.append('Logistic Regression')


# ### Naive Bayes:

# In[154]:


nb=GaussianNB()


# In[155]:


cv_nb=cross_val_score(nb, X_train,Y_train, cv=10)


# In[156]:


print(cv_nb.mean())


# In[157]:


scores_list.append(cv_nb.mean())
model_list.append('Naive Bayes')


# ### Random Forest

# In[127]:


rf=RandomForestClassifier(n_estimators=20)


# In[128]:


cv_rf=cross_val_score(rf, X_train,Y_train, cv=10)


# In[130]:


print(cv_rf.mean())


# In[132]:


scores_list.append(cv_rf.mean())
model_list.append('Random Forest')


# ### XGBClassifier:

# In[135]:


xgb=XGBClassifier()


# In[136]:


cv_xgb=cross_val_score(xgb, X_train,Y_train, cv=10)


# In[137]:


print(cv_xgb.mean())


# In[138]:


scores_list.append(cv_xgb.mean())
model_list.append('XGB Classifier')


# ### SVM:

# In[141]:


svm=SVC() #using the default 'RBF' kernel


# In[142]:


cv_svm=cross_val_score(svm, X_train,Y_train, cv=10)


# In[143]:


print(cv_svm.mean())


# In[145]:


scores_list.append(cv_svm.mean())
model_list.append('SVM')


# ### Voting Classifier:

# ##### This creates a Voting classifier of previously used classifiers to predict values. It is best to pass odd number of models into this classifiers to help tie breaking moments

# In[160]:


vc=VotingClassifier(estimators=[('lr',lr),('xg',xg),('svm',svm),('rf',rf),('nb',nb)],voting='hard')


# In[161]:


cv_vc=cross_val_score(vc, X_train,Y_train, cv=10)


# In[162]:


print(cv_vc.mean())


# In[164]:


scores_list.append(cv_vc.mean())
model_list.append('Voting Classifier')


# ## Artificail Neural Networks:

# ### TensorFlow backend:

# #### Importing libraries needed to run Keras Classifier for our multiclass problem:

# In[166]:


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils


# ##### The Target variable is to be Label encoded and one hot encoded:

# In[168]:


X=X_train
Y=Y_train


# In[169]:


encod = LabelEncoder()
encod.fit(Y)
encod_Y = encod.transform(Y)


# In[170]:


dum_y = np_utils.to_categorical(encod_Y)


# ##### Creating the base model function to pass to our Keras Classifier:

# In[171]:



def bmodel():
        model = Sequential()
        model.add(Dense(20, input_dim=10, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


# In[172]:


ANN = KerasClassifier(build_fn=bmodel, epochs=200, batch_size=5, verbose=8)


# In[209]:


cv_NN=cross_val_score(ANN, X,encod_Y, cv=10)


# In[214]:


print(cv_NN.mean())


# In[176]:


scores_list.append(cv_NN.mean())
model_list.append('Neural Network')


# In[177]:


scores_list=[i*100 for i in scores_list]


# ### Comparing the various model performances to choose the best:

# In[179]:


Comparision=pd.DataFrame()
Comparision['Model']=model_list
Comparision['Score']=scores_list


# In[180]:


Comparision['Score']=Comparision['Score'].round(3)


# In[188]:


Comparision['Score'][Comparision['Model']=='Neural Network']=


# In[258]:


Comparision.plot.bar(x='Model', y='Score', rot=0)
plt.show()


# ##### Hence we see that our Keras classifier predicts the best

# ### Tuning the Selected ANN model for best accuracy:

# ##### Creating Keras Classifier with a validation split of 33%

# In[234]:


KRAS = KerasClassifier(build_fn=bmodel,validation_split=0.33, epochs=800, batch_size=8)


# ##### Fitting and saving the fit data inside 'tuning' to refer in the next step

# In[235]:


tuning=KRAS.fit(X,encod_Y)


# ##### Let us check the parameters and find out the best epoch value to use in our model

# In[237]:


tuning.history.keys()


# In[238]:



TuneDF=pd.DataFrame()
TuneDF['accuracy']=tuning.history['accuracy']
TuneDF['val_accuracy']=tuning.history['val_accuracy']
TuneDF['loss']=tuning.history['loss']
TuneDF['val_loss']=tuning.history['val_loss']
TuneDF['epoch']=TuneDF.index.tolist() #using the index number as the epoch number as they are the same


# In[259]:


plt.plot(TuneDF['accuracy'])
plt.plot(TuneDF['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Test_cv'], loc='best')
plt.show()


# #### Let's get the epochs where the validation accuracy and train set accuracy are highest

# ##### we do not need loss in the current problem so we ignore loss values

# In[240]:


print('Max Validation Accuracy(epoch number):',
      TuneDF['epoch'][TuneDF['val_accuracy'] == TuneDF['val_accuracy'].max()].tolist(),
      'Score:',
      TuneDF['val_accuracy'].max()
     )
print('')
print("Max Train Accuracy(epoch number):",
      TuneDF['epoch'][TuneDF['accuracy'] == TuneDF['accuracy'].max()].tolist(),
      'Score:',
       TuneDF['accuracy'].max()
     )
      


# ##### The training accuracy is high mainly due to overfitting so let's go with an epoch close to highest validation accuracy

# ###### Hence we now know around which value to give the epoch argument to get close to the best accuracy

# ## Final Model building:

# In[246]:


Final_Keras=KerasClassifier(build_fn=bmodel,epochs=100, batch_size=8)


# In[247]:


Final_Keras.fit(X,encod_Y)


# In[250]:


predictions = Final_Keras.predict(X_test)
Keras_Pred=encod.inverse_transform(predictions)

print(Keras_Pred)


# In[251]:


Output=pd.DataFrame()


# In[252]:


Output['id']=test_id
Output['Grade']=Keras_Pred


# In[253]:


Output.to_csv("C:/PGA10/Projects/Skillenza/OutpuFile.csv",index=False)

