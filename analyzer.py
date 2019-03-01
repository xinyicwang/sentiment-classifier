#!/usr/bin/env python
# coding: utf-8

# ## Twitter Sentiment Analyzer
# 
# 

# In[1]:


import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")


# Categorize data
cols = ['sentiment','id','date','query_string','user','text']

# Load the dataset
df = pd.read_csv("./trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None, names=cols, encoding="ISO-8859-1")
df.head()


# ### Data preprocessing
# 
# 

# In[2]:


import re
from bs4 import BeautifulSoup

# Tokenization
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

# Function for data cleaning
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
        
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    
    # Remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    
    return (" ".join(words)).strip()


# In[4]:


# Testing
testing = df.text[:100]
test_result = []

for t in testing:
    test_result.append(tweet_cleaner(t))   


# In[5]:


test_result


# In[6]:


nums = [0,400000,800000,1200000,1600000]

# Parsing the first group of tweets
print("Cleaning and parsing the tweets...\n")

clean_tweet_texts = []
for i in range(nums[0],nums[1]):
    if( (i+1)%10000 == 0 ):
        print("Tweets %d of %d has been processed" % (i+1, nums[1]))                                                                      
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))


# In[8]:


# Parsing the second group of tweets
print("Cleaning and parsing the tweets...\n")

clean_tweet_texts = []
for i in range(nums[1],nums[2]):
    if( (i+1)%10000 == 0 ):
        print("Tweets %d of %d has been processed" % (i+1, nums[2]))                                                                      
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))


# In[9]:


# Parsing the third group of tweets
print("Cleaning and parsing the tweets...\n")

clean_tweet_texts = []
for i in range(nums[2],nums[3]):
    if( (i+1)%10000 == 0 ):
        print("Tweets %d of %d has been processed" % (i+1, nums[3]))                                                                      
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))


# In[10]:


# Parsing the fourth group of tweets
print("Cleaning and parsing the tweets...\n")

clean_tweet_texts = []
for i in range(nums[3],nums[4]):
    if( (i+1)%10000 == 0 ):
        print("Tweets %d of %d has been processed" % (i+1, nums[4]))                                                                      
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))


# In[11]:


# Saving cleaned data as .csv
clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = df.sentiment
clean_df.head()

clean_df.to_csv('clean_tweet.csv',encoding='utf-8')

csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.head()


# ### Classifier

# In[12]:


# Update the data cleaning function by adding sentiment analysis
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

part1 = r'@[A-Za-z0-9_]+'
part2 = r'https?://[^ ]+'
combined_pat = r'|'.join((part1, part2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner_updated(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
        
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)   
    # Tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    
    return (" ".join(words)).strip()


# In[13]:


df = pd.read_csv("./trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None,usecols=[0,5],
                 names=['sentiment','text'], encoding="ISO-8859-1")
# Map data by sentiment
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
df.head()


# In[14]:


# Cleaning the tweets
print("Cleaning the tweets...\n") 
clean_tweet_texts = []
for i in range(0,len(df)):
    if( (i+1)%100000 == 0 ):
        print("Tweets %d of %d has been processed" % (i+1, len(df)))                                                                     
    clean_tweet_texts.append(tweet_cleaner_updated(df['text'][i]))


# In[16]:


# Save the file as .csv
clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = df.sentiment
clean_df.to_csv('clean_tweet.csv')

csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.head()


# In[17]:


# Remove null entries during cleaning
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
my_df.info()


# ### Model

# In[18]:


# Split the data
x = my_df.text
y = my_df.target

# Cross validation
from sklearn.model_selection import train_test_split
SEED = 2000

x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, 
                                                                                  random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, 
                                                              test_size=.5, random_state=SEED)

print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".
      format(len(x_train),(len(x_train[y_train == 0]) / (len(x_train)*1.))*100,
       (len(x_train[y_train == 1]) / (len(x_train)*1.))*100))

print("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".
      format(len(x_validation),(len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,
       (len(x_validation[y_validation == 1]) / (len(x_validation)*1.))*100)) 

print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".
      format(len(x_test),(len(x_test[y_test == 0]) / (len(x_test)*1.))*100,
       (len(x_test[y_test == 1]) / (len(x_test)*1.))*100)) 


# ### Results

# In[19]:


# Baseline (for comparison)
from textblob import TextBlob as tb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

tbresult = [tb(i).sentiment.polarity for i in x_validation]
tbpred = [0 if n < 0 else 1 for n in tbresult]

conmat = np.array(confusion_matrix(y_validation, tbpred, labels=[1,0]))

confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
                         columns=['predicted_positive','predicted_negative'])

print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_validation, tbpred)*100)) 
print("-"*80) 
print("Confusion Matrix\n") 
print(confusion) 
print("-"*80) 
print("Classification Report\n") 
print(classification_report(y_validation, tbpred)) 


# In[22]:


# Feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from time import time

# Function to train on a different number of features iteratively
def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    if len(x_test[y_test == 0]) / (len(x_test)*1.) > 0.5:
        null_accuracy = len(x_test[y_test == 0]) / (len(x_test)*1.)
    else:
        null_accuracy = 1. - (len(x_test[y_test == 0]) / (len(x_test)*1.))
    t0 = time()
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print("Null accuracy: {0:.2f}%".format(null_accuracy*100)) 
    print("Accuracy score: {0:.2f}%".format(accuracy*100)) 
    if accuracy > null_accuracy:
        print("Model is {0:.2f}% more accurate than null accuracy".format((accuracy-null_accuracy)*100)) 
    elif accuracy == null_accuracy:
        print("Model has the same accuracy with the null accuracy") 
    else:
        print("Model is {0:.2f}% less accurate than null accuracy".format((null_accuracy-accuracy)*100)) 
    print("Train and test time: {0:.2f}s".format(train_test_time)) 
    print("-"*80) 
    return accuracy, train_test_time


# Function to check the accuracy of logistic regression
cvec = CountVectorizer()
lr = LogisticRegression()
n_features = np.arange(10000,100001,10000)

def nfeature_accuracy_checker(vectorizer=cvec, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=lr):
    result = []
    print(classifier)
    print("\n") 
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Validation result for {} features".format(n)) 
        nfeature_accuracy,tt_time = accuracy_summary(checker_pipeline, x_train, y_train, x_validation, y_validation)
        result.append((n,nfeature_accuracy,tt_time))
    return result


# #### Unigram

# In[23]:


print("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n") 
feature_result_wosw = nfeature_accuracy_checker(stop_words='english')


# In[24]:


print("RESULT FOR UNIGRAM WITH STOP WORDS\n") 
feature_result_ug = nfeature_accuracy_checker()


# In[31]:


# Implement stop words
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()
cvec.fit(my_df.text)

neg_doc_matrix = cvec.transform(my_df[my_df.target == 0].text)
pos_doc_matrix = cvec.transform(my_df[my_df.target == 1].text)
neg_tf = np.sum(neg_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()


# In[32]:


# Slice term frequencies
term_freq_df.columns = ['negative', 'positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
term_freq_df.sort_values(by='total', ascending=False).iloc[:10]

term_freq_df.to_csv('term_freq_df.csv',encoding='utf-8')


# In[33]:


# Open as .csv file
csv = 'term_freq_df.csv'
term_freq_df = pd.read_csv(csv,index_col=0)
term_freq_df.sort_values(by='total', ascending=False).iloc[:10]


# In[35]:


# Compare accuracy under different conditions
my_stop_words = frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index))

print("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n") 
feature_result_wocsw = nfeature_accuracy_checker(stop_words=my_stop_words)

print("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n") 
feature_result_wosw = nfeature_accuracy_checker(stop_words='english')

print("RESULT FOR UNIGRAM WITH STOP WORDS\n") 
feature_result_ug = nfeature_accuracy_checker()


# In[36]:


# Visualize results of the comparison
nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wocsw = pd.DataFrame(feature_result_wocsw,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='with stop words')
plt.plot(nfeatures_plot_ug_wocsw.nfeatures, nfeatures_plot_ug_wocsw.validation_accuracy,label='without custom stop words')
plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy,label='without stop words')
plt.title("Without stop words VS With stop words (Unigram): Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()


# #### Bigram

# In[29]:


print("RESULT FOR BIGRAM WITH STOP WORDS\n") 
feature_result_bg = nfeature_accuracy_checker(ngram_range=(1, 2))


# #### Trigram

# In[37]:


print("RESULT FOR TRIGRAM WITH STOP WORDS\n") 
feature_result_tg = nfeature_accuracy_checker(ngram_range=(1, 3))


# In[38]:


nfeatures_plot_tg = pd.DataFrame(feature_result_tg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_bg = pd.DataFrame(feature_result_bg,columns=['nfeatures','validation_accuracy','train_test_time'])
nfeatures_plot_ug = pd.DataFrame(feature_result_ug,columns=['nfeatures','validation_accuracy','train_test_time'])

plt.figure(figsize=(8,6))
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy,label='trigram')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy,label='bigram')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='unigram')
plt.title("N-gram(1~3) test result : Accuracy")
plt.xlabel("Number of features")
plt.ylabel("Validation set accuracy")
plt.legend()

