import nltk
import re
import matplotlib.pyplot as plt
nltk.download('stopwords')
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def text_cleaner(text,num):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    # newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num==0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens=newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:  #removing short word
            long_words.append(i)
    return (" ".join(long_words)).strip()


def clean_data(data):
    cleaned_summary = []
    for t in data['Summary']:
        cleaned_summary.append(text_cleaner(t,1))
    cleaned_summary = []
    for t in data['Summary']:
        cleaned_summary.append(text_cleaner(t,1))
    data.replace('', np.nan, inplace=True)
    data.dropna(axis=0,inplace=True)
    return data


def word_count(data):
    text_word_count = []
    summary_word_count = []

    # populate the lists with sentence lengths
    for i in data['cleaned_text']:
        text_word_count.append(len(i.split()))

    for i in data['cleaned_summary']:
        summary_word_count.append(len(i.split()))

    length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})

    length_df.hist(bins = 30)
    plt.show()
    

def create_training_data(data, max_text_len, max_summary_len):
    cleaned_text =np.array(data['cleaned_text'])
    cleaned_summary=np.array(data['cleaned_summary'])

    short_text=[]
    short_summary=[]

    for i in range(len(cleaned_text)):
        if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])
    return pd.DataFrame({'text':short_text,'summary':short_summary})