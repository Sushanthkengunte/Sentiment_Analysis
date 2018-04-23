
from nltk.corpus import sentence_polarity
import random
import nltk
import re
sentences = sentence_polarity.sents()
documents = [(sent, cat) for cat in sentence_polarity.categories() for sent in sentence_polarity.sents(categories=cat)]
random.shuffle(documents)

stopwords = nltk.corpus.stopwords.words('english')

all_words_list = [word for (sent,cat) in documents for word in sent]

def alpha_filter(w):
    # pattern to match word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(w)):
        return True
    else:
        return False

all_words_list = [ eachToken for eachToken in all_words_list if not alpha_filter(eachToken)]
all_words_list = [ eachToken for eachToken in all_words_list if eachToken not in stopwords]

all_words = nltk.FreqDist(all_words_list)

word_items = all_words.most_common(2000)
word_features = [word for (word, freq) in word_items]