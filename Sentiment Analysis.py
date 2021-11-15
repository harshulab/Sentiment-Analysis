#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd
import numpy as np

df = pd.read_csv("amazon_baby.csv")
df.dropna()
df


# In[116]:


#Taking a 30% representative sample
import numpy as np
np.random.seed(34)
df1 = df.sample(frac = 0.1)
df1


# In[117]:


df1['sentiment']=0
for rowind in  range(len(df1)):
    if df1['rating'].iloc[rowind] in [1,2]:
        df1['sentiment'].iloc[rowind] = 0
    else:
        df1['sentiment'].iloc[rowind] = 1
df1


# In[118]:


X = df1.review
y = df1.sentiment
X


# In[119]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=24)
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer()


# In[120]:


# print(X_train)


# In[121]:


# #Vectorizing the text data
# ctmTr=[0]*9176
# for i in range(len(X_train)):
#     ctmTr[i] = cv.fit_transform(X_train.iloc[i])
# # X_test_dtm = cv.transform(X_test)
# for i in ctmTr:
#     print(i)


# In[122]:


# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# lr.fit(ctmTr, y_train)


# In[123]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.5, random_state=4)
from sklearn.feature_extraction.text import TfidfVectorizer
#tfidf vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train.apply(lambda x: np.str_(x)))
X_test_vec = vectorizer.transform(X_test.apply(lambda x: np.str_(x)))
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train_vec, y_train)
lr_score = lr.score(X_test_vec, y_test)
print("Results for Logistic Regression with tfidf")
print("LR Score: ", lr_score)
y_pred_lr = lr.predict(X_test_vec)


#Confusion matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_lr)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
print("TN: ", tn, "FP: ", fp, "FN: ", fn, "TP: ", tp)
#True positive and true negative rates


tpr_lr = round(tp/(tp + fn), 4)
tnr_lr = round(tn/(tn+fp), 4)
print(tpr_lr, tnr_lr)


# In[ ]:




