import pandas as pd
import numpy as np
import re
import nltk
import os



# Reading tweets from file
AmazonQ1 = "Data/output_got_amazon_Qrt1.csv"
AmazonQ2 = "Data/output_got_amazon_Qrt2.csv"
AmazonQ3 = "Data/output_got_amazon_Qrt3.csv"
AmazonQ4 = "Data/output_got_amazon_Qrt4.csv"

WalmartQ1 = "Data/output_got_Walmart_Qrt1.csv"
WalmartQ2 = "Data/output_got_Walmart_Qrt2.csv"
WalmartQ3 = "Data/output_got_Walmart_Qrt3.csv"
WalmartQ4 = "Data/output_got_Walmart_Qrt4.csv"

EbayQ1 = "Data/eBay-1st-Quarter.csv"
EbayQ2 = "Data/eBay-2nd-Quarter.csv"
EbayQ3 = "Data/eBay-3rd-Quarter.csv"
EbayQ4 = "Data/eBay-4th-Quarter.csv"

TargetQ1 = "Data/Target-1st-Quarter.csv"
TargetQ2 = "Data/Target-2nd-Quarter.csv"
TargetQ3 = "Data/Target-3rd-Quarter.csv"
TargetQ4 = "Data/Target-4th-Quarter.csv"

BestBuyQ1 = "Data/output_got_bestbuy_Qrt1.csv"
BestBuyQ2 = "Data/output_got_bestbuy_Qrt2.csv"
BestBuyQ3 = "Data/output_got_bestbuy_Qrt3.csv"
BestBuyQ4 = "Data/output_got_bestbuy_Qrt4.csv"

# quartKeys = {0:'aq1',1:'aq2',2:'aq3',3:'aq4',
#         4:'wq1',5:'wq2',6:'wq3',7:'wq4'}

# quartKeys = {0:'eb1',1:'eb2',2:'eb3',3:'eb4',
#         4:'ta1',5:'ta2',6:'ta3',7:'ta4'}

quartKeys = {0:'bb1',1:'bb2',2:'bb3',3:'bb4'}

# files = [AmazonQ1,AmazonQ2,AmazonQ3,AmazonQ4,WalmartQ1,WalmartQ2,WalmartQ3,WalmartQ4]
files = [BestBuyQ1,BestBuyQ2,BestBuyQ3,BestBuyQ4]
# files = [EbayQ1,EbayQ2,EbayQ3,EbayQ4,TargetQ1,TargetQ2,TargetQ3,TargetQ4]


words = set(nltk.corpus.words.words())

#Preprocessing
EMOJI_NAME_REGEX = re.compile('<Emoji:.*>')
HASHTAGS_REGEX = re.compile('#\w*')
MENTIONS_REGEX = re.compile('@[^\s]+')
# LINK_REGEX = re.compile('https?://[^\s]+')

LINK_REGEX = re.compile('https?://(.*/[\s]?)?(.*_)?[^\s]+')
# Link_SECOND_REGEX = re.compile('(\w*\.)+com/[^\s]+')
EXTRA_SPACES_REGEX = re.compile('\s{2,}')




pattern = r''' (?x)             

            \w+'\w+  

            |\w+

            |\.

            '''
# reviewsArray_copy = []

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

# def remove_secondHyperlinks(tweet):
#     return Link_SECOND_REGEX.sub('', tweet)

preprocessing_pipeline = [
    preprocess_hashtags,
    preprocess_mentions,
    remove_hyperlinks,
    preprocess_Emojis,
    # remove_secondHyperlinks,
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

def isNan(tweet):
    return tweet != tweet

arrayDict = {}
tokenizedArrayDict = {}
arrayTodelete = []
def populateDictionaries(reviewsArray,key):
    reviewsArray_copy = []
    # reviewsArray = np.array(reviewsArray)
    for i, raw_tweet in enumerate(reviewsArray):
        if isNan(raw_tweet):
            arrayTodelete.append(i)
            continue
        reviewsArray[i] = preprocess_tweet(raw_tweet, preprocessing_pipeline)
    reviewsArray = np.delete(reviewsArray, arrayTodelete)
    for i, raw_tweet in enumerate(reviewsArray):
        reviewsArray_copy.append(nltk.regexp_tokenize(reviewsArray[i], pattern))
        reviewsArray_copy[i] = [eachToken for eachToken in reviewsArray_copy[i] if eachToken.lower() in words and not alpha_filter(eachToken) and eachToken != "via"]
    # print(len(np.delete(reviewsArray,arrayTodelete)))

    # print(len(reviewsArray),len(reviewsArray_copy))
    arrayDict[key] = reviewsArray
    tokenizedArrayDict[key] = reviewsArray_copy

# alpha_filter(eachToken)



def main():
    for i in range(len(files)):
        sd = pd.read_csv(files[i])
        data = sd[sd.columns[4]].values
        reviewsArray = np.array(data)
        populateDictionaries(reviewsArray,quartKeys[i])
        # output = ""
        # name = 'Output/' + quartKeys[i] + '.txt'
        # file = open(name, 'w', encoding="utf-8")
        # temp = tokenizedArrayDict[quartKeys[i]]
        # for each in temp:
        #     sentence = " ".join(each)
        #     output += sentence+'\n'
        # file.write(output)
        # file.close()


    # print(arrayDict,tokenizedArrayDict)
# main()








