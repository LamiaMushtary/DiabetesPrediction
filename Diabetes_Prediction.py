#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np 


# In[6]:


import seaborn as sn


# In[7]:


import pandas as pd


# In[8]:


import matplotlib.pyplot as plt 


# In[6]:


get_ipython().system('pip install mlxtend')


# In[9]:


from pandas.plotting import scatter_matrix


# In[11]:


get_ipython().system('pip install missingno')


# In[10]:


import missingno as msno


# In[111]:


from mlxtend.plotting import plot_decision_regions


# In[12]:


from pandas.plotting import scatter_matrix


# In[13]:


from sklearn.preprocessing import StandardScaler


# In[14]:


from sklearn.model_selection import train_test_split


# In[20]:


pip install -U scikit-learn scipy matplotlib


# In[23]:


pip install -U scikit-learn scipy matplotlib


# In[15]:


from sklearn.neighbors import KNeighborsClassifier


# In[16]:


from sklearn.metrics import confusion_matrix


# In[17]:


from sklearn import metrics


# In[18]:


from sklearn.metrics import classification_report


# In[19]:


from sklearn.metrics import roc_curve


# In[20]:


from sklearn.metrics import roc_auc_score


# In[21]:


from sklearn.model_selection import GridSearchCV


# In[22]:


import warnings


# In[23]:


warnings.filterwarnings('ignore')


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[41]:


diabetes_df = pd.read_csv(r"C:\Users\lamia.mushtary\Downloads\Data Analysis\Diabetes data analysis\diabetes.csv", encoding= 'unicode_escape')


# In[43]:


diabetes_df.head()


# In[26]:


diabetes_df= pd.read_csv(r"C:\Users\lamia.mushtary\Downloads\Data Analysis\Diabetes data analysis\diabetes.csv", encoding= 'unicode_escape')


# In[27]:


diabetes_df.head()


# In[28]:


diabetes_df.columns


# In[29]:


diabetes_df.info()


# In[30]:


diabetes_df.describe()


# In[31]:


diabetes_df.describe().T


# In[34]:


#checking of any null values or not: 
diabetes_df.isnull()


# In[35]:


#chcking if there is some null value or not
diabetes_df.isnull().sum()


# In[39]:


diabetes_df_copy= diabetes_df.copy(deep= True)
diabetes_df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]=diabetes_df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0,np.nan)
#showing the count of NaN
print (diabetes_df_copy.isnull().sum())


# In[40]:


#plotting data distribution plots
p= diabetes_df.hist(figsize=(20,20))


# In[46]:


#aiming to imput NaN values for the columns in accordance with their distribution
diabetes_df_copy['Glucose'].fillna (diabetes_df_copy['Glucose'].mean(), inplace=True)
diabetes_df_copy['BloodPressure'].fillna (diabetes_df_copy['BloodPressure'].mean(), inplace=True)
diabetes_df_copy['SkinThickness'].fillna (diabetes_df_copy['SkinThickness'].median(), inplace=True)
diabetes_df_copy['Insulin'].fillna (diabetes_df_copy['Insulin'].median(), inplace=True)
diabetes_df_copy['BMI'].fillna (diabetes_df_copy['BMI'].median(), inplace=True)


# In[48]:


#plotting distributions after removing the NaN values
p= diabetes_df_copy.hist(figsize=(20,20))


# In[49]:


#ploting Null Count Analysis Plot
p= msno.bar(diabetes_df) 


# In[56]:


#checking the balace of the data by plotting count of the outcomes by their values
color_wheel= {1: "#0392cf", 2: "#7bc043"}
colors= diabetes_df["Outcome"].map(lambda x:color_wheel.get(x+1))
print(diabetes_df.Outcome.value_counts())
p= diabetes_df.Outcome.value_counts().plot(kind="bar")                                   


# The plot is refering that the number of non-diabetes is almost twice the number of diabetes patients

# In[61]:


#plotting scatter matrix of the uncleaned data
p= scatter_matrix(diabetes_df, figsize=(20,20))


# In[62]:


#plotting the pair plots for the data
p= sn.pairplot(diabetes_df_copy, hue= 'Outcome')


# In[70]:


#correlation between all the features before cleaning
plt.figure(figsize=(12,10))
p= sn.heatmap(diabetes_df.corr(), annot=True, cmap ='RdYlGn')
plt.show()


# In[71]:


#correlation between all the features after cleaning
plt.figure(figsize=(12,10))
p= sn.heatmap(diabetes_df_copy.corr(), annot=True, cmap ='RdYlGn')
plt.show()


# In[72]:


#scaling the data
diabetes_df_copy.head()


# In[78]:


sc_x= StandardScaler()
x= pd.DataFrame(sc_x.fit_transform(diabetes_df_copy.drop(["Outcome"], axis=1),), columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin', 'BMI',	'DiabetesPedigreeFunction',	'Age'])
x.head()


# In[79]:


y= diabetes_df_copy.Outcome


# In[80]:


y


# Splitting the data into Train and Test

# In[81]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=42, stratify=y)


# Model Building

# K-Nearest Neighbor (KNN)

# In[85]:


test_scores= []
train_scores= []

for i in range(1,15):

  knn= KNeighborsClassifier(i)
  knn.fit(x_train,y_train)

  train_scores.append(knn.score(x_train,y_train))
  test_scores.append(knn.score(x_test,y_test))


# In[86]:


train_scores


# In[87]:


test_scores


# In[92]:


max_train_score= max(train_scores)
train_scores_ind= [i for i, v in enumerate(train_scores) if v== max_train_score]
print ('Max train score {}% and k={}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))


# In[93]:


max_test_score= max(test_scores)
test_scores_ind= [i for i, v in enumerate(test_scores) if v== max_test_score]
print ('Max test score {}% and k={}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))


# In[99]:


plt.figure(figsize=(12,5))
p=sn.lineplot(x=range(1,15), y=train_scores, marker='*', label='Train Score')
p=sn.lineplot(x=range(1,15), y=test_scores, marker='o', label='Test Score')


# The best result is captured at k=11 hence 11 is used for the final model 

# In[100]:


knn= KNeighborsClassifier(11)
knn.fit(x_train, y_train)
knn.score(x_test,y_test)


# In[112]:


#plot the decision boundary
value=20000
width=20000
plot_decision_regions(x.values, y.values, clf=knn, legend=2, 
                      filter_feature_values={2:value, 3:value, 4:value, 5:value},
                      filter_feature_ranges={2:width, 3:width, 4:width, 5:width},
                      x_highlight=x_test.values)
plt.title('KNN with Diabetes Data')
plt.show()


# In[116]:


from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
# Fit the KNN model
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train, y_train)
plt.figure(figsize=(10, 6))
plot_decision_regions(X=x_train.values, y=y_train.values, clf=knn, legend=2, X_highlight=x_test.values)
plt.title('KNN with Diabetes Data')
plt.xlabel('Feature 1')  # Replace with actual feature names if known
plt.ylabel('Feature 2')  # Replace with actual feature names if known
plt.show()


# In[116]:


from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
# Fit the KNN model
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train, y_train)
plt.figure(figsize=(10, 6))
plot_decision_regions(X=x_train.values, y=y_train.values, clf=knn, legend=2, X_highlight=x_test.values)
plt.title('KNN with Diabetes Data')
plt.xlabel('Feature 1')  # Replace with actual feature names if known
plt.ylabel('Feature 2')  # Replace with actual feature names if known
plt.show()


# In[116]:


from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
# Fit the KNN model
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train, y_train)
plt.figure(figsize=(10, 6))
plot_decision_regions(X=x_train.values, y=y_train.values, clf=knn, legend=2, X_highlight=x_test.values)
plt.title('KNN with Diabetes Data')
plt.xlabel('Feature 1')  # Replace with actual feature names if known
plt.ylabel('Feature 2')  # Replace with actual feature names if known
plt.show()


# In[117]:


from sklearn.decomposition import PCA


# In[118]:


from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
# Fit the KNN model
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train, y_train)
plt.figure(figsize=(10, 6))
plot_decision_regions(X=x_train.values, y=y_train.values, clf=knn, legend=2, X_highlight=x_test.values)
plt.title('KNN with Diabetes Data')
plt.xlabel('Feature 1')  # Replace with actual feature names if known
plt.ylabel('Feature 2')  # Replace with actual feature names if known
plt.show()


# In[119]:


# Assuming x_train and x_test are your feature matrices
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Fit the KNN model with PCA-reduced data
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train_pca, y_train)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plot_decision_regions(X=x_train_pca, y=y_train, clf=knn, legend=2, X_highlight=x_test_pca)

plt.title('KNN with PCA-Reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[120]:


from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

# Reduce to 2D using PCA
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Convert pandas Series to NumPy arrays
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Fit the KNN model with PCA-reduced data
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train_pca, y_train_np)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plot_decision_regions(X=x_train_pca, y=y_train_np, clf=knn, legend=2, X_highlight=x_test_pca)

plt.title('KNN with PCA-Reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# Confusion Matrix

# In[123]:


pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Fit the KNN model with PCA-reduced data
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train_pca, y_train)

# Predict using PCA-reduced test data
y_pred = knn.predict(x_test_pca)

# Confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sn.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# In[124]:


#Classification Reports
print(classification_report(y_test, y_pred))


# ROC-AUC Curve

# In[127]:


from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Apply PCA transformation (if not already applied)
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Fit the KNN model with PCA-reduced data
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train_pca, y_train)

# Predict probabilities using PCA-reduced test data
y_pred_proba = knn.predict_proba(x_test_pca)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='o', label='KNN (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_proba)))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# Implementing GridSearchCV

# In[129]:


#Incase of classifier like KNN the parameter to be turned is n_neighbors

param_grid={'n_neighbors': np.arange(1,50)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(x,y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))


# In[ ]:




