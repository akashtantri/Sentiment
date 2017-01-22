import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report


fp = open('traindata1.csv','r')
reader = csv.reader(fp,delimiter=',')

name = []
val = []
examples = ['front camera is bad']
	
for x in reader:
	name.append(x[1]);
	val.append(x[0]);
	

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(name)
train_lab = vectorizer.transform(val)
test_vectors = vectorizer.transform(examples)

classifier_rbf = svm.SVC()
classifier_rbf.fit(train_vectors,val)
prediction_rbf = classifier_rbf.predict(test_vectors)

print prediction_rbf