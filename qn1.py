import sys,json,re,math,itertools
import numpy as np
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,HashingVectorizer
from sklearn.feature_extraction import DictVectorizer,FeatureHasher
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.naive_bayes import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import *
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus   import words
from nltk.corpus   import stopwords,gazetteers,names
from nltk.collocations import *
from nltk.metrics import BigramAssocMeasures
from nltk.probability import *
from sklearn.ensemble import RandomForestClassifier
eng_words = set([ w.lower() for w in words.words('en') ])
qn_words = set(['who','what','what',
	'when','where','how',
	'is','should','do',
	'if','would','should'])
stopwords = [ w for w in stopwords.words('english') if  w not in qn_words ]
places = set([ w.lower() for w in gazetteers.words() ])
names  = set([ w.lower() for w in names.words() ])


class Extractor:
	def __init__(self,fun,per_instance=True):
		self.extractor = fun
		self.per_instance = True
	def fit(self,X,Y):
		pass
	def transform(self,X):
		if self.per_instance:
			return [ self.extractor(x) for x in X ] 
		else: return self.extractor(X)
	def fit_transform(self,X,_):
		return self.transform(X)

class ToArray:
	def __init__(self):
		pass
	def fit(self,X,Y):
		pass
	def transform(self,X):
		return X.toarray() 
	def fit_transform(self,X,_):
		return self.transform(X)

qn_type_words = [ set(l) for l in [
	['who'],
	['what','why','how'],
	['which'],
	['when'],
	['where'],
	['is','do','can','did','was'],
	['should','could'],
	['would','will']
	]]

def find_collocations(text,length):
	finder = BigramCollocationFinder.from_words(text,window_size=5)
	finder.apply_freq_filter(int(math.log(length)/2))
	return finder.nbest(BigramAssocMeasures.chi_sq,50)

allcaps = re.compile(r'^[A-Z]+$')
startscap = re.compile(r'^[A-Z]')
def formatting_features(question):
	question = question.strip()
	tokens = [ w for w in wordpunct_tokenize(question) if not re.match(r'[\'\"\.\?\!\,\/\\\(\)\`]',w) ]
	qn_mark   = 1 if "?" in question else -1 
	start_cap = 1 if re.match(r'^[A-Z]',question) else -1
	if tokens:
		qn_type = [ sum(1.0 for w in tokens if w in qws)
				for qws in qn_type_words ]
		nm_pres = sum(1.0 for w in tokens if w.lower() in names)
		pl_pres = sum(1.0 for w in tokens if w.lower() in places)
	else:
		qn_type = [0.0]*len(qn_type_words)
		nm_pres = 0.0
		pl_pres = 0.0
	total_words = len(tokens)
	correct_form_count = sum(1.0 for w in tokens
			if (w.lower() in eng_words and not allcaps.match(w))
			or startscap.match(w))
	correct_form_ratio = correct_form_count/float(total_words+1e-10)

	result = [
			nm_pres,pl_pres,
			qn_mark,start_cap,
			correct_form_ratio,
			math.log(total_words+1)
			] + qn_type
	return result

training_count = int(sys.stdin.next())
training_data  = [ json.loads(sys.stdin.next()) for _ in xrange(training_count) ]
target         = [ obj['__ans__'] for obj  in training_data ]

ans_sentences  = ( w.lower()
		for instance in training_data if instance['__ans__']
		for w in wordpunct_tokenize(instance['question_text'])
		if len(w) > 3 and w.lower() not in stopwords)
nans_sentences  = ( w.lower()
		for instance in training_data if not instance['__ans__']
		for w in wordpunct_tokenize(instance['question_text'])
		if len(w) > 3 and w.lower() not in stopwords)

collocated = find_collocations(ans_sentences,len(training_data)) + find_collocations(nans_sentences,len(training_data))

formatting = Pipeline([
	('other',  Extractor(formatting_features)),
	('scaler', StandardScaler())
])

question = Pipeline([
	('extract', Extractor(lambda x: x['question_text'].lower())),
	('hasher',  Extractor(lambda x: {
		(f,s): 	(1 if f in x else 0)+
				(1 if s in x else 0)
		for f,s in collocated if f in x or s in x})),
	('counter',DictVectorizer()),
])

def topicname(x):
	topic_names = [t['name'] for t in x['topics']]
	topic_names.sort()
	res = {tt:1 for tt in topic_names}
	return res
topics = Pipeline([
	('extract',Extractor(topicname)),
	('counter',FeatureHasher(n_features=2**12+1)),
	#('counter',DictVectorizer()),
	('cluster',MiniBatchKMeans(n_clusters=8))
])

topic_question = Pipeline([
	('content',FeatureUnion([
		('question', question),
		('topics',   topics),
	])),
])


formatting = Pipeline([
	('extract', Extractor(lambda x: x['question_text'])),
	('formatting',formatting)
])

others = Pipeline([
	('extract', Extractor(lambda x: [
		1.0 if x['anonymous'] else 0,
	])),
	#('scaler',  StandardScaler())
])

followers = Pipeline([
	('extract',Extractor(lambda x: [
		math.log(sum(t['followers'] for t in x['topics'])+1)
	])),
	('scaler' ,StandardScaler())
])
model = Pipeline([
	('union',FeatureUnion([
		('content', topic_question),
		('formatting',formatting),
		('followers',followers),
		('others',others)
		])),
	('classify',SGDClassifier(loss='hinge',alpha=1e-3,n_iter=1000))
#	('classify',SVC())
#	('classify',MultinomialNB())
	#('classify',RandomForestClassifier(n_estimators=10))
	])


model.fit(training_data,target)

test_count = int(sys.stdin.next())
test_data  = [ json.loads(sys.stdin.next()) for _ in xrange(test_count) ]
for i,j in zip(model.predict(test_data).tolist(),test_data):
	print json.dumps({ 
		'__ans__':i,'question_key':j['question_key']
		})
