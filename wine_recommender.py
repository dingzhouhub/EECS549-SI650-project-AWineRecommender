import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud, STOPWORDS
from IPython.display import Image
#from IPython.display import Image
from os import path
from sklearn.cluster import KMeans

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text 
from sklearn.metrics.pairwise import linear_kernel

plt.style.use('fivethirtyeight')

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_val_score

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.tokenize import word_tokenize, regexp_tokenize
import nltk
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
from unicodedata import normalize

import string
import spacy
from spacy.lang.en import English

#load file
df_wine = pd.read_csv('wine-reviews/winemag-data-130k-v2.csv')

df_wine = df_wine.drop_duplicates(subset='title',keep=False)
## Do some initial cleaning on the dataset: Drop the repeated index and twitter handle columns; check data types
df_wine.drop(["Unnamed: 0", "taster_twitter_handle"], axis=1, inplace=True)
## Make GSM one set of characters so that it doesn't get split in later text analysis
df_wine.loc[df_wine.variety == 'G-S-M','variety'] = 'GSM'
# Extract the year where it exists from the title column
df_wine['year'] = df_wine['title'].str.extract('(\d\d\d\d)')
#print("%.1f%% of the wines have a year associated with them." %((1-(df_wine['year'].isnull().sum()/len(df_wine['year'])))*100))
df_wine = df_wine.dropna(subset=['country','description','points','price','province','title','variety','year'])

sample = df_wine.sample(n = 40000, random_state = 0)
sample = sample.reset_index()
sample = sample.drop(['index'], axis = 1)
df_wine=sample

one_hot = pd.get_dummies(df_wine['country'])
KmeanMatrix = pd.concat([one_hot, df_wine[['price','points','year']]], axis=1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(KmeanMatrix)
KmeanMatrix = scaler.transform(KmeanMatrix)

kmeans = KMeans(n_clusters = 20, n_init = 5, n_jobs = -1)
kmeans.fit(KmeanMatrix)

wine=df_wine
wine['clusterK'] = kmeans.labels_

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    #mytokens = [ word for word in mytokens if word not in punctuations ]
    #mytokens = [ word for word in mytokens if word not in stopwords.words() ]

    # return preprocessed list of tokens
    return mytokens

tf = TfidfVectorizer(stop_words=stop_words,tokenizer=spacy_tokenizer)
tf_matrix = tf.fit_transform(wine['description'])

taster = wine.taster_name.unique().tolist()
taster = [taster for taster in taster if str(taster) != 'nan']
user_taster = pd.DataFrame(np.zeros((len(wine),len(taster))),columns=taster)
for i in taster:
    user_taster[i] = (wine[wine['taster_name']==i]['points'] < wine[wine['taster_name']==i]['points'].mean()).astype(int)*2-1
user_taster[~(user_taster==user_taster)]=0

wine = pd.concat([wine,user_taster],axis=1)

def getChoice(key):
    choices = wine['title']    
    
    #gets the top five closest matches ands asks the user to pick one
    matches = (process.extract(key, choices, scorer = fuzz.ratio, limit = 5))
    print("\n Here are the 5 closest matches...Pick one for the recommender to use. \n")
    print("\n".join(map(str, matches)))
    
    global pick_one
    pick_one = input("Enter wine here: ")
    nameAdd = (process.extract(pick_one, choices, scorer = fuzz.ratio, limit = 1))[0][0]
#     print("\n Okay, here are 10 other wines that are like", pick_one.title())
    return nameAdd
#     return pick_one

username = input('What is your user name? ')
if username not in wine.columns:
    wine[username]=0

flag = True
while(flag):
    wineChoice = input('What desired wine would you like to search for? ')
    name = getChoice(wineChoice)
    wine.loc[wine['title'] == name,username] = 1
    
    choice = input('Do you want to continue input the desired wine? Input y/n: ')
    flag = choice == 'y'

choice = input("Are there any wine you do not want? Input y/n: ")
flag = choice == 'y'    
  
while(flag):
    wineChoice = input('What undesired wine would you like to search for? ')
    name = getChoice(wineChoice)
    wine.loc[wine['title'] == name,username] = -1
    
    choice = input('Do you want to continue input the undesired wine? input y/n: ')
    flag = choice == 'y'

cl = np.zeros(20)
for line in range(len(wine)):
    cl[wine.loc[line,'clusterK']] += wine.loc[line, username] 
    
wineLike = wine[wine['clusterK']==max(cl)]
tf = TfidfVectorizer(stop_words=stop_words,tokenizer=spacy_tokenizer)
tf_matrix = tf.fit_transform(wineLike['description'])
cosine_sim = linear_kernel(tf_matrix, tf_matrix)
cosine = pd.DataFrame(cosine_sim)

from nltk.tokenize import word_tokenize, regexp_tokenize
import nltk
from nltk.corpus import stopwords
# Assign the stop words to a variable
stop_words = set(stopwords.words('english'))

all_air_text = wine[wine[username] == 1] # Extract all columns that were her choices

air_text = ''.join(string for string in all_air_text['description']) # Join the text in all of her choices into one string

# air_text = air_text.replace(r"'",'').replace(r'(','').replace(r')','') # Replace unnecessary characters

# pattern = r'\w+|\d+' # Write a pattern for matching both text or digits 

# regex = regexp_tokenize(air_text, pattern) # Apply pattern to air_text variable

# unique_regex = list(set(regex)) # List of unique words in the full combined description
# unique_regex = [i for i in unique_regex if i not in stop_words] # Remove stop words from the unique regex
air_text = pd.Series(air_text)
seri= wineLike['description']
seri = seri.append(air_text)
seri = seri.reset_index()
seri = seri.rename(columns={0:'description'})

def tfidf_recommendation(df_K):
    
    """
    Takes in a data frame with wine descriptions, passes the descriptions into a TFIDF function, determines
    the similarity of a given observation/description to the rest of the inputs using cosine similarity, 
    and return a data frame with the top similarity scores for each of our wines.
    """
    
    # Extract the description column from the entered data
    
#     all_descriptions = df_K[['description']]
    
    # Initialize a TFIDF Vectorizer model to work with the text data
    
    tf = TfidfVectorizer(analyzer='word',
                     min_df=0,
                     stop_words='english')

    # Use the initiated TFIDF model to transform the data in descriptions
    
    tfidf_matrix = tf.fit_transform(seri['description'])
#     tfidf_like = tf.fit_transform(pd.Series(air_text))
    # Compute the cosine similarities between the items in the newly transformed TFIDF matrix
    cosine_similarities = cosine_similarity(tfidf_matrix,tfidf_matrix)
    indice = cosine_similarities[-1,:].argsort()[-12:-2]
 
    return seri.iloc[indice,0].values

rec = sample.loc[tfidf_recommendation(wineLike).tolist(),['description', 'title']]
print(rec)
