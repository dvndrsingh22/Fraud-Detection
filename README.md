# Fraud-Detection
From this Dataset i am gonna predict the person is fraud or not

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PowerTransformer


# load a dataset

df = pd.read_csv("creditcard.csv")
df.head()

# last 5 rows of the dataset

df.tail()

# nearly 48 hr data is there

d = 172792/60
d/60

df.shape

df.size

# checking the null values and behaviour of the data

df.info()

df.isnull().sum()

#separating the class on the basis of 0 and 1 if o then it is normal customer nad if it is 1 then it is fraudulant customer
# 0 - for noraml transaction
# 1 - for fraudalent transaction
df['Class'].value_counts()

# naming the 0 and 1
legit = df[df.Class==0]
fraud = df[df.Class ==1
print(legit.shape)
print(fraud.shape)

# checking the description of the Amount wrt ) and 1
legit.Amount.describe()
fraud.Amount.describe()

df.describe()


df.groupby('Class').mean()

legit_sample = legit.sample(n= 492)
legit_sample.shape

df_new = pd.concat([legit_sample,fraud],axis =0)
df_new.shape

# now again checking the first and last 5 rows
df_new.head()
df_new.tail()

df_new['Class'].value_counts()

df_new.groupby('Class').mean()

#spliting the data and stor the data into separate variabel
X = df_new.drop(columns = 'Class',axis =1)
y = df_new['Class']

print(X)
print(y)

# split the data into train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,stratify = y,random_state = 2)

model = LogisticRegression()
model.fit(X_train,y_train)

# Evaluating model
print("training_score:",model.score(X_train,y_train))
print("testing_score:",model.score(X_test,y_test))

# After applying powertransform how the accuracy changes
pt =PowerTransformer()
X_train_pt = pt.fit_transform(X_train)
X_test_pt = pt.fit_transform(X_test)

model = LogisticRegression()
model.fit(X_train_pt,y_train)

print("training_score:",model.score(X_train_pt,y_train))
print("testing_score:",model.score(X_test_pt,y_test))

# With Pipeline
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=20)
pipe = Pipeline((
("it", IterativeImputer()),
("pt",PowerTransformer()),
("lr", LogisticRegression()),
))
pipe.fit(X_train,y_train)

print("Training Accuracy")
print(pipe.score(X_train,y_train))
print("Testing Accuracy")
print(pipe.score(X_test,y_test))

from sklearn.metrics import confusion_matrix,classification_report,recall_score,precision_score,f1_score
predicted = pipe.predict(X_test)
print(confusion_matrix(y_test,predicted))
print(classification_report(y_test,predicted))


#Evaluating models using Cross Validation
from sklearn.model_selection import cross_val_score

scoreslr = cross_val_score(pipe, X_train, y_train, cv=10, scoring='accuracy')
print(scoreslr)

print("Average Accuracy of my model")
print(np.mean(scoreslr))
print("SD of accuracy of the model")
print(np.std(scoreslr,ddof=1))

# 95% Confidence Interval of Accuracy
import scipy.stats as stats

xbar = np.mean(scoreslr)
n=10
s = np.std(scoreslr,ddof=1)
se = s/np.sqrt(n)

stats.t.interval(0.95,df=n-1,loc=xbar,scale=se)

import seaborn as sns
sns.pairplot(X_train,diag_kind='kde')




