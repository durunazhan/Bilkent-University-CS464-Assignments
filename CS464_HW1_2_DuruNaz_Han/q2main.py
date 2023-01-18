#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math as m
import matplotlib.pyplot as plt
import numpy as np
import array as arr


# In[3]:


pd.read_csv('bbcsports_train.csv')


# In[4]:


pd.read_csv('bbcsports_val.csv')


# In[5]:


class_vals = pd.read_csv('bbcsports_train.csv')['class_label'].value_counts()
print(class_vals)


# In[6]:


fig, axes = plt.subplots(figsize=(5,5), dpi=100)
plt.bar(class_vals.index, height=class_vals, color ='pink')
plt.xlabel("Classes")
plt.ylabel("No. of instances in each class")
plt.title('Barplot of distribution of each class');


# In[7]:


validation = pd.read_csv('bbcsports_val.csv')['class_label'].value_counts()
print(validation)


# In[8]:


fig, axes = plt.subplots(figsize=(5,5), dpi=100)
plt.bar(validation.index, height=validation, color ='yellow')
plt.xlabel("Classes")
plt.ylabel("No. of instances in each class")
plt.title('Barplot of distribution of each class');


# In[9]:


#We first need to estimate the probability of any particular document belonging to a class yk
#pi is the fraction of each class
pi = {}
train_y_arr = np.array([])
#Set a class index for each document as key
for i in range(0,5):
    pi[i] = 0    
    
train_set = pd.read_csv('bbcsports_train.csv') 

#Get total number of documents
train_x = train_set.shape[0]

#Count the occurence of each class
for i in range(len(train_set)):
    train_y = train_set['class_label'].iloc[i]
    pi[train_y] += 1
    train_y_arr = np.append(train_y_arr,train_y)
    
#Divide the count of each class by total documents 
for key in pi:
    pi[key] /= train_x
    
print("Probability of each class:")
print("\n".join("{}: {}".format(k, v) for k, v in pi.items()))


# In[12]:


#FEATURE EXTRACTION
#dataset_x = train_set.values[:,:-1]
#train_x = dataset_x[:len(train_set.values)]
features= list(train_set.columns)
features.pop() 
len(features)
features


# In[13]:


#likelihood calculation for MLE estimator
theta = {} #likelihood
yk = {} #stores the amount of words in each class
wt = {} #stores the probability of a specific word in each class
for i in range(0,5):
    train_i = train_set[train_set['class_label'] == i] #look at only the documents with label i
    yk[i] = train_i.sum(axis=1).sum() #how many words are there with document label i
    for j in train_i:
        if j != 'class_label': #do this until the last column (each column represents a word)
            train_col = train_i[[j]] #count the amount of word j with documents labeled i 
            wt[j] = float((train_col.sum()/yk[i]))
        else:
            theta[i] = wt
            wt = {}
print(theta)


# In[14]:


#function for the MLE estimator
def MNB_MLE(test):
    data = test.values[:,:-1]
    test_x = data[:len(data)] #the data frame is turned into an array to achieve better runtime for the fucntion
    compare={} #the probabilities for each class will be compared 
    guess_mle=np.array([]) #resulting class for each document will be put here
    for doc in range(len(test_x)): #looking at each document of the given test data set
        compare={}
        for label in np.unique(train_y_arr): #each word will be tested for each label (5 times)
            count = 0
            prob = m.log(pi[label]) #prior probability previously calculated for each class is added to the total probability
            for feature in features: #each word is tested (4163 times)
                if test_x[doc][count] == 0: 
                    prob_w = m.log(1-(theta[label][feature]))
                else:
                    if theta[label][feature]==0:
                         prob_w=test_x[doc][count]*np.nan_to_num(-np.inf) #instead of log(0), -inf is taken
                    else:
                        prob_w= test_x[doc][count]*m.log(theta[label][feature])
                prob+=prob_w
                count=count+1 #looking at the next word in the document
                compare[label]=prob
                
        guess_mle = np.append(guess_mle, max(compare, key=compare.get)) #each class' probability is compared and the largest one is taken
    return(guess_mle)
                        
        


# In[15]:


#accuracy function taken from colab tutorial
def calc_accuracy(ground_truth_labels, predicted_labels): 
	correct = 0
	# Pairs of the ground truth and predicted labels
	for gt_label, pred_label in zip(ground_truth_labels, predicted_labels):
		if gt_label == pred_label: 
			correct += 1
	print(correct/len(ground_truth_labels))


# In[17]:


#extracting the labels from validation set for testing
df = pd.read_csv('bbcsports_val.csv')
dataset_y=df.values[:,-1]
test_y = dataset_y[:len(df.values)]
print(test_y)


# In[18]:


#predicting the validation set via MNB function
prediction = MNB_MLE(df)


# In[19]:


print(prediction)


# In[20]:


#accuracy for the MLE estimator
calc_accuracy(test_y,prediction)


# In[22]:


#confusion matrix for the MLE estimator
confusion = [[0 for i in range(0,5)] for j in range(0,5)]
correct = 0
for gt_label, pred_label in zip(test_y, prediction):
    correct += 1
    confusion[int(gt_label)][int(pred_label)] +=1
print(confusion)


# In[23]:


#likelihood calculation for MAP estimator
theta_map = {} #likelihood
yk_map = {} #stores the amount of words in each class
wt_map = {} #stores the probability of a specific word in each class
alpha = 1 #for this question we chose additive smoothing alpha as 1 but we named it as alpha in case we would want to change it
for i in range(0,5):
    train_i = train_set[train_set['class_label'] == i]
    yk_map[i] = train_i.sum(axis=1).sum() + alpha + (len(train_set.columns) - 1)
    for j in train_i:
        if j != 'class_label':
            train_col = train_i[[j]]
            wt_map[j] = float(((train_col.sum()+alpha)/yk_map[i]))
        else:
            theta_map[i] = wt_map
            wt_map = {}
print(theta_map)


# In[24]:


#we can use the same function we used in MLE for the MAP estimator 
def MNB_MAP(test):
    data = test.values[:,:-1]
    test_x = data[:len(data)] #the data frame is turned into an array to achieve better runtime for the fucntion
    compare={} #the probabilities for each class will be compared 
    guess_map=np.array([]) #resulting class for each document will be put here
    for doc in range(len(test_x)): #looking at each document of the given test data set
        compare={}
        for label in np.unique(train_y_arr): #each word will be tested for each label (5 times)
            count = 0
            prob = m.log(pi[label]) #prior probability previously calculated for each class is added to the total probability
            for feature in features: #each word is tested (4163 times)
                if test_x[doc][count] == 0: 
                    prob_w = m.log(1-(theta_map[label][feature]))
                else:
                    if theta_map[label][feature]==0:
                         prob_w=test_x[doc][count]*np.nan_to_num(-np.inf) #instead of log(0), -inf is taken
                    else:
                        prob_w= test_x[doc][count]*m.log(theta_map[label][feature])
                prob+=prob_w
                count=count+1 #looking at the next word in the document
                compare[label]=prob
                
        guess_map = np.append(guess_map, max(compare, key=compare.get)) #each class' probability is compared and the largest one is taken
    return(guess_map)
                        


# In[25]:


#prediction for the MAP estimator
prediction_MAP = MNB_MAP(df)


# In[26]:


#accuracy for the MAP estimator
calc_accuracy(test_y,prediction_MAP)


# In[27]:


#confusion matrix for the MAP estimator
confusion = [[0 for i in range(0,5)] for j in range(0,5)]
correct = 0
for gt_label, pred_label in zip(test_y, prediction_MAP):
    correct += 1
    confusion[int(gt_label)][int(pred_label)] +=1
print(confusion)


# In[ ]:




