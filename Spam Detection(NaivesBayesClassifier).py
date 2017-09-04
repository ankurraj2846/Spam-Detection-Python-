
# coding: utf-8

# In[2]:

from sklearn.feature_extraction.text import CountVectorizer


# In[3]:

count_vector = CountVectorizer()


# In[4]:

print(count_vector)   ## Below are the arguments need to input for the CountVectorizer function


# In[5]:

import pandas as pd
df = pd.read_table('/Users/AR/Desktop/NanoDegree/SpamDetection/data.txt', sep = '\t', names =['label', 'SMS'], header = None)


# In[6]:

df.head()


# In[7]:

from sklearn.cross_validation import train_test_split        ## For splitting the training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['SMS'], 
                                                    df['label'],     ## X and Y need to written sequentially in both LHS and RHS
                                                    random_state=1)  

print('Number of rows in the total set: {} '.format(df.shape[0]))   ## We use .format same as %s in C
print('Number of rows in the training set: {}'.format(X_train.shape[0])) ## .shape returns the dimensionality of the dataframe
print('Number of rows in the test set: {}'.format(X_test.shape[0])) ## 0 returns the rows and 1 returns the columns


# In[8]:

from sklearn.naive_bayes import MultinomialNB


# In[9]:

training_data = count_vector.fit_transform(X_train)  ## Fit the testing data and return matrix

testing_data = count_vector.transform(X_test).toarray()  ## Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()

#training_data
#X_train.head()
#X_test.head()
#y_train.head()
#y_test.head()


# In[10]:

#frequency_matrix = pd.DataFrame(testing_data, columns = count_vector.get_feature_names())
#frequency_matrix


# In[11]:

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)


# In[12]:

documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']

test_vector = CountVectorizer()
test_vector.fit(documents)
test_vector.get_feature_names()


# In[13]:

doc_array = test_vector.transform(documents).toarray()
doc_array


# In[14]:

predictions = naive_bayes.predict(testing_data)
predictions


# In[19]:

from sklearn.metrics import  precision_score ,accuracy_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions, average='weighted')))
print('Recall score: ', format(recall_score(y_test, predictions, average='weighted')))
print('F1 score: ', format(f1_score(y_test, predictions, average='weighted')))


# In[ ]:




# In[ ]:



