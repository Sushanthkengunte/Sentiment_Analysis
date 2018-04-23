import nltk
import gatheringInputs as Processing
import featureWords as fw
from sklearn.svm import LinearSVC

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

documents = fw.documents
featuresets = [(document_features(d,fw.word_features), c) for (d,c) in documents]

amazonDocument = Processing.reviewsArray
reviewFeatureSets = [(document_features(each,fw.word_features))for each in amazonDocument]


train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
# classifier = nltk.classify.DecisionTreeClassifier.train(train_set)
# classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
print (nltk.classify.accuracy(classifier, test_set))
# decision tree, SVM,

positive = list()
negative = list()
predictions = list()

for i,each in enumerate(reviewFeatureSets):
    temp = classifier.classify(each)
    predictions.append((Processing.reviews[i],temp))


for each in predictions:
    if(each[1] == 'pos'):
        positive.append(each[0])
    else:
        negative.append(each[0])



file = open('Output/positive_SVM.txt', 'w')
output = " =>POSITIVE\n".join(positive)
file.write(output)
file.close()

file = open('Output/negative_SVM.txt', 'w')
output = " =>NEGATIVE\n".join(negative)
file.write(output)
file.close()
