import json,sys,re,math
from random import random
import numpy as np
from pprint import pprint
from sklearn import svm
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,LinearSVC,SVR
from sklearn.linear_model import *

from nltk.tokenize import wordpunct_tokenize
from nltk.corpus   import stopwords
from nltk.corpus   import words

stopwords = [ 
		w for w in stopwords.words('english')
				if  w not in ['who','what','when','where','how','is']
			]
def word_tokenize_filter(string):
	return [ w for w in wordpunct_tokenize(string) if not re.match(r'^[\'\"\:\;\.\,\!]$',w) ]
#eng_words = set([ w.lower() for w in words.words('en') ])
def prep_words(training_data,target,clsf,n):
	counter = CountVectorizer(
			tokenizer=word_tokenize_filter,
#			stop_words=stopwords,
			binary=True,
			dtype=np.byte,
			ngram_range = (1,1),
			min_df = 1
		)
	model = Pipeline([
							('vect',counter),
							('clsf',clsf)
						 ])
	training_data  = [  d['question_text'] for d in training_data ]
	#training_data  = input_data[1:5000]
	model.fit(training_data,target)
	words = counter.get_feature_names()
	weights = np.abs(clsf.coef_)
	important = zip(weights,words)
	important.sort()
	print [ w for _,w in important[-n:] ]

def prep_topics(training_data,target,clsf,n):
	counter = DictVectorizer()
	model = Pipeline([
							('vect',counter),
							('clsf',clsf)
						 ])
	training_count = int(sys.stdin.next())
	training_data  = [ { t['name']:1 for t in d['topics']} for d in training_data ]
	#training_data  = input_data[1:5000]
	model.fit(training_data,target)
	words = counter.get_feature_names()
	weights = clsf.coef_.toarray()[0]
	important = zip(abs(weights),words)
	important.sort()
	print [ w for _,w in important[-n:] ]

if __name__=="__main__":
	training_count = int(sys.stdin.next())
	training_data  = [ json.loads(sys.stdin.next()) for _ in xrange(training_count) ]
	target         = [ math.log(math.log(obj['__ans__']+1)+1) for obj  in training_data ]
	#prep_topics(training_data,target,SVR(kernel='linear'),50)
	#prep_words(training_data,target,Ridge(),200)
	training_data.sort(key=lambda x:x['__ans__'])
	for i in training_data:

		print "%0.3f %s"%(math.log(math.log(i['__ans__']+0.9)+0.9),i['question_text'].encode('utf-8'))

