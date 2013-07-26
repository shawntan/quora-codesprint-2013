import sys,json,re,math
import numpy as np
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,HashingVectorizer
from sklearn.feature_extraction import DictVectorizer,FeatureHasher
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import *
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus   import words
from nltk.corpus   import stopwords,gazetteers,names
from sklearn.feature_selection import *
eng_words = set([ w.lower() for w in words.words('en') ])
qn_words = set(['who','what','what',
				'when','where','how',
				'is','should','do',
				'if','would','should'])
stopwords = [ w for w in stopwords.words('english') if  w not in qn_words ]
places = set([ w.lower() for w in gazetteers.words() ])
names  = set([ w.lower() for w in names.words() ])


class Extractor:
	def __init__(self,fun):
		self.extractor = fun
	def fit(self,X,Y):
		pass
	def transform(self,X):
		return [ self.extractor(x) for x in X ] 
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
	['who','which','when','where'],
	['what','why','how'],
	['is','do','can','did','was'],
#	['should','could','would','will']
]]

def formatting_features(obj):
	question = obj['question_text'].strip()
	tokens   = [ w for w in wordpunct_tokenize(question) if not re.match(r'[\'\"\.\?\!\,\/\\\(\)\`]',w) ]
	top_toks = set([ w.lower() for t in obj['topics'] 
						for w in wordpunct_tokenize(t['name']) ])
	qn_toks  = set(tokens)
	qn_tok_words = len(top_toks & qn_toks)

	qn_mark   = 1 if "?" in question else -1 
	start_cap = 1 if re.match(r'^[A-Z]',question) else -1
	if tokens:
		qn_type = [ sum(1.0 for w in tokens if w in qws)
						for qws in qn_type_words ]
		nm_pres = sum(1.0 for w in tokens if w.lower() in names
							and re.match(r'^[A-Z]',w))
		pl_pres = sum(1.0 for w in tokens if w.lower() in places
							and re.match(r'^[A-Z]',w))
	else:
		qn_type = [-1.0]*len(qn_type_words)
		nm_pres = -1.0
		pl_pres = -1.0
	total_words = len(tokens)
	#dict_words  = sum(1 for w in tokens if w.lower() in eng_words)
	correct_form_count = sum(1.0 for w in tokens
			if (w.lower() in eng_words and not re.match(r'^[A-Z]+$',w))
			or re.match(r'^[A-Z]',w)
		)
	correct_form_ratio = correct_form_count/float(total_words+1e-10)
	token_word_ratio   = qn_tok_words/float(total_words+1e-10)
#	name_ratio        = (nm_pres + pl_pres)/float(total_words+1e-10)
	result = [
				nm_pres,pl_pres,
				qn_mark,start_cap,
				correct_form_ratio,
				token_word_ratio,
				qn_tok_words,
				correct_form_count,
				math.log(total_words+1)
			] + qn_type
	return result

word_counter = CountVectorizer(
		tokenizer=wordpunct_tokenize,
		stop_words=stopwords,
		binary=True,
		ngram_range=(1,2),
	#	dtype=np.float32
	)

counter = HashingVectorizer(
#		vocabulary=vocabulary,
		tokenizer=wordpunct_tokenize,
		stop_words=stopwords,
#		binary=True,
		n_features = 2**10+1,
		ngram_range=(1,2),
#		dtype=np.float32
	)

formatting = Pipeline([
	('other',  Extractor(formatting_features)),
	('scaler', StandardScaler())
])

def word_scorer(x):
	res = {}
	tokens = wordpunct_tokenize(x)
	for i,w in enumerate(tokens):
		w = w.lower()
		if w not in stopwords and len(w) > 3:
			res[w] = math.exp(-i/len(tokens)) + 1
	return res


question = Pipeline([
	('extract', Extractor(lambda x: x['question_text'])),
	#('counter', word_counter),
	('word_s', Extractor(word_scorer)),
	('counter',DictVectorizer()),
	('f_sel',   SelectKBest(
		score_func=lambda X,Y:f_regression(X,Y,center=False),k=100)),
#	('cluster',MiniBatchKMeans(n_clusters=8))
])
topics = Pipeline([
	('extract',Extractor(lambda x: {
		t['name']:1 for t in x['topics']
	})),
	('counter', FeatureHasher(n_features=2**16+1, dtype=np.float32)),
	('cluster',MiniBatchKMeans(n_clusters=60))
	#('cluster',MiniBatchKMeans(n_clusters=8))
])

topic_question = Pipeline([
	('content',FeatureUnion([
		('question', question),
		('topics',   topics)
	])),
])
others = Pipeline([
	('extract', Extractor(lambda x: [
		float(1 if x['anonymous'] else 0),
	])),
	('scaler',  StandardScaler())
])


ctopic = Pipeline([
	('extract',Extractor(lambda x:
		{ x['context_topic']['name']:1 }
		if x['context_topic'] else { 'none':1})),
	('counter',FeatureHasher(n_features=2**8+1, dtype=np.float)),
	('f_sel',  SelectKBest(
		score_func=lambda X,Y:f_regression(X,Y,center=False),
		k=180)),
])

followers = Pipeline([
	('extract',Extractor(lambda x: [
		math.log(sum(t['followers'] for t in x['topics'])+0.001)
	])),
	('scaler' ,StandardScaler())
])
model = Pipeline([
	('union',FeatureUnion([
		('content', topic_question),
		('ctopic',  ctopic),
		('formatting',formatting),
		('followers',followers),
		('others',others)
	])),
#	('toarray',ToArray()),
#	('dim_red',PCA(n_components=2)),
#	('regress',DecisionTreeRegressor())
#	('regress',KNeighborsRegressor())
#	('regress',SVR())
#	('regress',Ridge())
	('regress',RidgeCV(alphas=[ 0.1**(-i) for i in range(10)]))
#	('regress',SGDRegressor(alpha=1e-3,n_iter=1500))

])



training_count = int(sys.stdin.next())
training_data  = [ json.loads(sys.stdin.next()) for _ in xrange(training_count) ]
target         = [ math.log(obj['__ans__']+0.9) for obj  in training_data ]

model.fit(training_data,target)
#sys.stderr.write(' '.join(vocabulary)+"\n")
#sys.stderr.write("%s\n"%counter.transform([' '.join(vocabulary)]))

test_count = int(sys.stdin.next())
test_data  = [ json.loads(sys.stdin.next()) for _ in xrange(test_count) ]

for i,j in zip(model.predict(test_data).tolist(),test_data):
	print json.dumps({ 
		'__ans__':math.exp(i)-0.9,'question_key':j['question_key']
	})

