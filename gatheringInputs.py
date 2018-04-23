import pandas as pd
import numpy as np
import re
import nltk

# Reading tweets from file
InputFile = "Input/file201701_201812.csv"
sd = pd.read_csv(InputFile)
data = sd[sd.columns[5]].values
reviewsArray = np.array(data)


#Preprocessing
EMOJI_NAME_REGEX = re.compile('<Emoji:.*>')
HASHTAGS_REGEX = re.compile('#\w*')
MENTIONS_REGEX = re.compile('@[^\s]+')
LINK_REGEX = re.compile('https?://[^\s]+')
Link_SECOND_REGEX = re.compile('(\w*\.)+com/[^\s]+')
EXTRA_SPACES_REGEX = re.compile('\s{2,}')




pattern = r''' (?x)             

            \w+'\w+  

            |\w+

            |\.

            '''
reviewsArray_copy = []

def preprocess_hashtags(tweet):
    return HASHTAGS_REGEX.sub('', tweet)    #Remove Hastags

def preprocess_Emojis(tweet):
    return EMOJI_NAME_REGEX.sub('', tweet)    #Remove Hastags

def preprocess_mentions(tweet):
    return MENTIONS_REGEX.sub('', tweet)    #Remove Mentions

def remove_extra_spaces(tweet):
    return EXTRA_SPACES_REGEX.sub(' ', tweet).strip()   #Remove Extra spaces

def remove_hyperlinks(tweet):
    return LINK_REGEX.sub('', tweet)        #Remoce Hyperlinks

def remove_secondHyperlinks(tweet):
    return Link_SECOND_REGEX.sub('', tweet)

preprocessing_pipeline = [
    preprocess_hashtags,
    preprocess_mentions,
    remove_hyperlinks,
    preprocess_Emojis,
    remove_secondHyperlinks,
    remove_extra_spaces
]

def preprocess_tweet(tweet, pipeline):
    for pipe in pipeline:
        tweet = pipe(tweet)
    return tweet

def alpha_filter(w):
    # pattern to match word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(w)):
        return True
    else:
        return False




def main():
    for i, raw_tweet in enumerate(reviewsArray):
        reviewsArray[i] = preprocess_tweet(raw_tweet, preprocessing_pipeline)
        reviewsArray_copy.append(nltk.regexp_tokenize(reviewsArray[i], pattern))
        reviewsArray_copy[i] = [eachToken for eachToken in reviewsArray_copy[i] if not alpha_filter(eachToken)]

main()





