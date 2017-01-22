import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

fp = open('traindata.csv','r')

reader = csv.reader(fp,delimiter=',')

name = []
val = []
examples = ['Simply great. Camera quality and focusing too good. Front camera also good.Speed and performance better.Screen wise and clarity of screen awesome.Handy mobile.Light weight and slimBest features : 3500 MAh, Removable Battery, 4G, Flash for camera and 2GB DDR3 RAM']
exp2 = ['Moto E3 problem When the battery reaches 15% and i connect it to charging it does not get charged .Ive to reboot itthen it gets charged .'];
for x in reader:
	name.append(x[2]);
	val.append(x[1]);
	

pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())])
pipeline.fit(name, val)
predictions = pipeline.predict(exp2)

if '1' in predictions:
	print ("Positive")
else:
	print ("Negative")