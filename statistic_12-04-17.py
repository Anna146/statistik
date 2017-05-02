
# coding: utf-8

# # Statistic 12-04-2017

# In[1]:
import pickle       # (De-) Serialization
import numpy as np
import pylab as plt # plot
import pandas


# ## Used TestFiles

# In[2]:

#testfiles = [f.split('/')[-1][2:] for f in pickle.load(open('./test_files.pkl','rb'))]

testfiles = [str(x.strip()) for x in open('filelist.txt', 'r')]

# In[3]:

print(testfiles)

# ## Data

# In[4]:
import json
#predicted = pickle.load(open('./predicted.pkl', 'rb'))
predicted = json.load(open('my_pred.json','r'))

#print(predicted[0])
#print(my_pred[5])

# In[5]:

real = pickle.load(open('./real.pkl', 'rb'))[:len(predicted)]


# ## Scoring per File

# In[6]:

db = pickle.load(open('./databuilder.pkl', 'rb'))


# In[7]:

from sklearn import metrics


# For Definition of accuracy, f1-score, precision and recall see http://scikit-learn.org/stable/modules/model_evaluation.html

# In[8]:

res = []
for i in range(len(predicted)):
    r = real[i]
    p = predicted[i]
    l = np.unique(real[0])
    acc = metrics.accuracy_score(y_true=r, y_pred=p)
    f1 = metrics.f1_score(labels=l, y_true=r, y_pred=p, average=None) # average=None -> get score for each label
    prec = metrics.precision_score(labels=l, y_true=r, y_pred=p, average=None)
    rec = metrics.recall_score(labels=l, y_true=r, y_pred=p, average=None)
    
    f1_all = np.ones(len(db.labels))*(-1)
    f1_all[l] = f1
    prec_all = np.ones(len(db.labels))*(-1)
    prec_all[l] = prec
    rec_all = np.ones(len(db.labels))*(-1)
    rec_all[l] = rec
    res.append([acc, np.mean(f1), np.mean(prec), np.mean(rec)])
res = np.array(res)


# - Warning occurs because some labels are not in the document, so it is dividing by zero
# - to get a value per label, these labels which aren't in the document there is a **-1**

# In[9]:

score_labels = lambda x : ['{}_{}'.format(x, l) for l in db.labels]


# In[10]:

cols = ['accuracy', 'mean(f1)', 'mean(prec)', 'mean(recall)']


# In[11]:

pd = pandas.concat((pandas.DataFrame(testfiles, columns=['files']), pandas.DataFrame(res, columns=[cols])), 1)


# In[12]:

#print(pd[:len(testfiles)])



# ## Score About ALL Test Files

# In[18]:

predicted_all_test_files = np.concatenate(predicted)
real_all_test_files = np.concatenate(real)


# ### <font style="color:red">Precision for all test Files<font>

# In[19]:

precision_per_label = metrics.precision_score(y_true=real_all_test_files, y_pred=predicted_all_test_files, average=None)
print(np.mean(precision_per_label))


# ### <font style="color:red">Recall for all test Files<font>

# In[20]:

recall_per_label = metrics.recall_score(y_true=real_all_test_files, y_pred=predicted_all_test_files, average=None)
print(np.mean(recall_per_label))


# ### <font style="color:red">F1 Score for all test Files<font>

# In[21]:

f1_per_label = metrics.f1_score(y_true=real_all_test_files, y_pred=predicted_all_test_files, average=None)
print(np.mean(f1_per_label))



# ### Confusion Matrix

# In[22]:

cc = metrics.confusion_matrix(real_all_test_files, predicted_all_test_files)
cc = cc / np.sum(cc, 0)
plt.figure(figsize=(15, 10))

plt.imshow(cc, cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.colorbar()
plt.xticks(np.arange(len(db.labels)), db.labels, rotation=90)
plt.yticks(np.arange(len(db.labels)), db.labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()


# In[23]:

plt.figure(figsize=(15,5))
plt.bar(range(len(precision_per_label)), precision_per_label)
plt.ylim(0,1)
plt.xticks(np.arange(len(db.labels)), db.labels, rotation=90)
plt.title('Precision Per Label')


# In[24]:

pandas.DataFrame(np.transpose([db.labels, precision_per_label]), columns=['Label', 'Precision'])


# In[25]:

plt.figure(figsize=(15,5))
plt.bar(range(len(recall_per_label)), recall_per_label)
plt.ylim(0,1)
plt.xticks(np.arange(len(db.labels)), db.labels, rotation=90)
plt.title('Recall Per Label')


# In[26]:

pandas.DataFrame(np.transpose([db.labels, recall_per_label]), columns=['Label', 'Recall'])


# In[27]:

plt.figure(figsize=(15,5))
plt.bar(range(len(f1_per_label)), f1_per_label)
plt.ylim(0,1)
plt.xticks(np.arange(len(db.labels)), db.labels, rotation=90)
plt.title('F1 Per Label')


# In[28]:

pandas.DataFrame(np.transpose([db.labels, f1_per_label]), columns=['Label', 'F1'])

