import nltk
import gatheringInputs as Processing
import featureWords as fw
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt;
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np


def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

documents = fw.documents
featuresets = [(document_features(d,fw.word_features), c) for (d,c) in documents]

# amazonDocument = Processing.reviewsArray_copy
# reviewFeatureSets = [(document_features(each,fw.word_features))for each in amazonDocument]


train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
# classifier = nltk.classify.DecisionTreeClassifier.train(train_set)
# classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(train_set)
print (nltk.classify.accuracy(classifier, test_set))
# decision tree, SVM,

# positive = list()
# negative = list()
predictions = list()
graphDict = {}

def main():
    count = 0
    Processing.main()
    for key in Processing.tokenizedArrayDict.keys():

        tokenizedDocuments = Processing.tokenizedArrayDict[key]
        reviewFeatureSets = [(document_features(each, fw.word_features)) for each in tokenizedDocuments]
        for i, one in enumerate(reviewFeatureSets):
            temp = classifier.classify(one)
            reviewsArray = Processing.arrayDict[key]
            predictions.append((reviewsArray[i], temp))
        positive = list()
        negative = list()
        for predict in predictions:
            if (predict[1] == 'pos'):
                positive.append(predict[0])
            else:
                negative.append(predict[0])
        posName = 'Output/positive_' + Processing.quartKeys[count] + '.txt'
        negName = 'Output/negative_' + Processing.quartKeys[count] + '.txt'
        file = open(posName, 'w', encoding="utf-8")
        print(Processing.quartKeys[count], 'Positive Count',len(positive))

        output = " =>POSITIVE\n".join(positive)
        # print(output)
        file.write(output)
        file.close()

        output = ""
        file = open(negName, 'w', encoding="utf-8")
        negativeOutput = [each for each in negative if each]
        print(Processing.quartKeys[count], 'Negative Count', len(negative))
        output = " =>NEGATIVE\n".join(negativeOutput)
        file.write(output)
        file.close()
        graphDict[Processing.quartKeys[count]] = (len(positive),len(negative))
        count+= 1


main()
positiveForGraph = []
neagativeForGraph = []
# objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
# y_pos = np.arange(len(objects))
objects = []
for eacgQuarterPositive in graphDict.keys():
    objects.append(eacgQuarterPositive)
    quart = graphDict[eacgQuarterPositive]
    positiveForGraph.append(quart[0])
    neagativeForGraph.append(quart[1])

y_pos = np.arange(len(objects))

plt.bar(y_pos, positiveForGraph, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('positive')
plt.title('Quarter wise positive')

plt.show()

plt.bar(y_pos, neagativeForGraph, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Negative')
plt.title('Quarter wise negative')

plt.show()
