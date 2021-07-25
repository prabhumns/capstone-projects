#!/usr/bin/env python
# coding: utf-8

# In[1]:


CURR_DIR = "/tf/Capstone Project/Project-2"


# In[2]:


import pandas as pd
import numpy as np
import unicodedata
import pandas as pd
import numpy as np
import nltk
import unicodedata
import contractions
import os
from datetime import datetime
from num2words import num2words
print(os.getcwd())
os.chdir(CURR_DIR)
print(os.getcwd())


# In[3]:


def shape_df(df):
    print(f"Number of observations: {df.shape[0]}")
    print(f"Number of variables: {df.shape[1]}")
    print(f"Number of duplicates: {df.duplicated().sum()}")
    print(f"Are there any missing values {df.isnull().values.any()}")
    print("-----")
    print(df.dtypes.sort_values(ascending=True))
    print("------")
    print("Datatypes' proportion:")
    print(df.dtypes.value_counts(ascending=True))


# In[4]:


def null_val(df):
    detect_null_val = df.isnull().values.any()
    if detect_null_val:
        null_abs = df.isnull().sum()
        null_pc = df.isnull().sum() / df.isnull().shape[0] *100
        null_concat = pd.concat([null_abs,null_pc], axis=1).round(2)
        null_concat.columns = ['Absolute', 'Percent']
        return null_concat.sort_values(by="Absolute", ascending=False)
    else:
        print("There are no missing values.")


# In[5]:


def corrs(x):
    mask = np.triu(x.corr(), 1)
    plt.figure(figsize=(19, 9))
    return sns.heatmap(x.corr(), annot=True, vmax=1, vmin=-1, square=True, cmap='BrBG', mask=mask);


# In[6]:


def unique_counts(df, thresh = 15):
    for column in df.columns:
        if df[column].nunique() < thresh:
            print(df.groupby([column], dropna = False).size(), '\n\n')


# In[7]:


df_raw = pd.read_csv('./Data/prods.csv')


# In[8]:


df_raw.head()


# In[9]:


df_raw.describe()


# In[10]:


df_raw['department_id'].nunique()


# In[11]:


df_raw['aisle_id'].nunique()


# In[12]:


shape_df(df_raw)


# In[13]:


null_val(df_raw)


# In[14]:


unique_counts(df_raw, thresh = 200)


# In[15]:


df = df_raw.copy() #Working on a copy of dataframe


# ### Some text cleaning functions used later

# In[16]:


#Removing HTML Tags
from bs4 import BeautifulSoup

def remove_html_tags(text):
    print('Removing HTML Tags, the text can be as big as entire wepage')
    return BeautifulSoup(text, 'html.parser').get_text()


#Removing Accented characters
import unicodedata

def remove_accented_chars(text):
    print("Removing accented characters, which convert résumé to resumve")
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text


# "Well this was fun! See you at 7:30, What do you think!!? #$@@9318@ 🙂🙂🙂" ==> 'Well this was fun See you at  What do you think  '


import re

def remove_special_characters(text, remove_digits=False):
    print("Removing special characters like smileys from the text")
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text


#Removing stopwords
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
new_stop_words = []
all_stop_words = stop_words.union(new_stop_words)

not_stopwords = {'no', 'not'}
final_stop_words = set(
    [word for word in all_stop_words if word not in not_stopwords]
)
def remove_stop_words(text):
    new_text = []
    for word in text.split():
        if word not in stop_words:
            new_text.append(word)
    return " ".join(new_text)
            
#Removing symbols (Without apostrophe)
symbols = "!\"#$%&()*+-./:;<=>?,@[\]^_`{|}~\n"
def remove_punctuation(text):
    for i in symbols:
        text = np.char.replace(text, i, ' ')
    return str(text)

#Convers popin' to poping
def convert_to_ing_words(text):
    new_text = []
    for word in text.split():
        if word[-3:] == "in'":
            new_text.append(word[:-3] + "ing")
        else:
            new_text.append(word)
    return " ".join(new_text)

#Conver Numeric values
def convert_numbers_to_words(text):
    new_text = []
    for word in text.split():
        if (word.isnumeric()):
            new_text.extend(num2words(word).split())
        else:
            new_text.append(word)
    return " ".join(new_text)
            
    
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def pos_tag_wordnet(tagged_tokens):
    tag_map = {'j': wordnet.ADJ, 'v': wordnet.VERB, 'n': wordnet.NOUN, 'r': wordnet.ADV}
    new_tagged_tokens = [(word, tag_map.get(tag[0].lower(), wordnet.NOUN))
                            for word, tag in tagged_tokens]
    return new_tagged_tokens

def wordnet_lemmatize_text(text):
    wnl = WordNetLemmatizer()
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(text))
    wordnet_tokens = pos_tag_wordnet(tagged_tokens)
    lemmatized_text = ' '.join(wnl.lemmatize(word, tag) for word, tag in wordnet_tokens)
    return lemmatized_text


# In[17]:


#Some product names with special and accented characters
df.iloc[[205, 23, 149]]


# In[18]:


#existing key-value pairs in contractions library
contractions.slang_dict


# In[19]:


#Few added contraction key-value pairs to fix the dialect of the names
contractions.add("lil'", 'little')
contractions.add("n'", 'and')
contractions.add("'n", 'and')
contractions.add("pop'n", "poping")
contractions.add("pop'ettes", "popettes")
contractions.add("chick'n", "chicken")


# In[20]:


def basic_cleaning(input_text):
    text = str(input_text)
    
#   Removing accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
#     Converting to lowercase
    text = text.lower().strip()
    
    #Doing stop words twice once before expanding once, after expanding
    text = remove_stop_words(text)
    
    #Fixing contractions
    text = contractions.fix(text)
    
    #Remove punctuation
    text = remove_punctuation(text)
    text = convert_numbers_to_words(text)
    
    #Replacing and character
    text = text.replace("&", 'and')
    text = text.replace("\\", '')
    text = text.replace("%", " percent")
    
    text = remove_punctuation(text)
    text = convert_to_ing_words(text)
    
    #Second stop words call
    text = remove_stop_words(text)    
    
    text = remove_punctuation(text)
    
    return text


# In[21]:


#product name will be the name from CSV
#changed_text will be after basic text processing
#Lemmatized_text column is after applying lemmatization on the changed text
df['changed_text'] = df['product_name'].apply(basic_cleaning)
df['lemmatized_text'] = df['changed_text'].apply(wordnet_lemmatize_text)


# In[22]:


#df[['changed_text', 'lemmatized_text']].to_csv('temp-'+ str(datetime.now()) + '.csv')


# In[23]:


#Finding the tf-idf vector values
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(df['lemmatized_text'].to_numpy())
tv_matrix = tv_matrix.toarray()

vocab = tv.get_feature_names()
vectors = pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)
total_df = df.join(vectors)


# In[24]:


vectors.shape


# In[25]:


#Calculating the cosine similarity matrix
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tv_matrix)

similarity_df = pd.DataFrame(similarity_matrix)
total_df = total_df.join(similarity_df)


# In[26]:


#Taking the top 5 matched products for each product

def get_similar_articles(x):
    return ", ".join([str(i) for i in df['product_id'].loc[x.argsort()[-6:-1]]])

total_df['top_5_similar_products'] = [get_similar_articles(x) for x in similarity_matrix]


# In[27]:


total_df[['product_id', 'top_5_similar_products']]


# In[28]:


def get_names_of_products(column_name, product_ids_as_string):
    all_names = []
    
    for id_ in product_ids_as_string.split(', '):
        similar_name = df.loc[df.product_id == int(id_), column_name].values.item()
        all_names.append(str(similar_name))
        
    return ", ".join(all_names)

total_df['similar_product_names'] = total_df['top_5_similar_products'].apply(lambda x: get_names_of_products('product_name', x))
total_df['similar_dept_ids'] = total_df['top_5_similar_products'].apply(lambda x: get_names_of_products('department_id', x))


# In[29]:


total_df[['similar_product_names', 'product_id', 'similar_dept_ids']]


# In[30]:


total_df['suggested_products'] = total_df['similar_product_names']
total_df['suggested_pro_ids'] = total_df['top_5_similar_products']
total_df['suggested_dept_ids'] = total_df['similar_dept_ids']
total_df[
    ['product_id', 
     'product_name', 
     'aisle_id', 
     'department_id', 
     'suggested_products', 
     'suggested_pro_ids', 
     'suggested_dept_ids']].to_csv('output.csv')

