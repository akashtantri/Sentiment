'''import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB'''
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

fp1 = open('trainset.csv','r')
fp2 = open('traindata1.csv','r')
#fp3 = open('mix.csv','r')


reader1 = csv.reader(fp1,delimiter=',')
reader2 = csv.reader(fp2,delimiter=',')
#reader3 = csv.reader(fp3,delimiter=',')


name1 = []
val1 = []

name2 = []
val2 = []

#name3 = []
#val3 = []

#examples = ['not so bad']

for x1 in reader1:
	name1.append(x1[1]);
	val1.append(x1[0]);

for x2 in reader2:
	name2.append(x2[1]);
	val2.append(x2[0]);

'''for x3 in reader3:
	name3.append(x3[1]);
	val3.append(x3[0]);
'''
'''
pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())])
		
pipeline.fit(name1,val1)
pipeline.fit(name2,val2)
#pipeline.fit(name3,val3)

def classify(exp):
	predictions = pipeline.predict(exp)

	if '1' in predictions:
		return 1
		#print ("Positive")
	else:
		return 0
		#print ("Negative")
		
#classify(examples)
'''
'''
with open("battery.txt", "r") as ins:
    arraylines = []
    for line in ins:
        arraylines.append(line)
for item in arraylines:
	x=[]
	x.append(item)
	classify(x)
'''
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(name1)
train_lab = vectorizer.transform(val1)

vectorizer1 = TfidfVectorizer()
test_set = vectorizer1.fit_transform(name2)
#test_lab = vectorizer.transform(val)

#test_vectors = vectorizer.transform(examples)

classifier_rbf = svm.SVC(kernel='linear',C=1,gamma=1)
classifier_rbf.fit(train_vectors,val1)

#prediction_rbf = classifier_rbf.predict(test_vectors)
#print(classifier_rbf.decision_function(test_vectors))

prediction_rcf = classifier_rbf.score(test_set,val2)


#print prediction_rbf

print prediction_rcf