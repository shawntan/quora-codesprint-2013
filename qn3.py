"""
57.892	58.62
"""
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
#	['who','which','when','where'],
#	['what','why','how'],
	['is','do','can','did','was'],
	['i'],
#	['deleted'],
#	['should','could','would','will']
]]

def formatting_features(obj):
	question = obj['question_text'].strip()
	topics   = [ t['name'] for t in obj['topics'] ]
	tokens   = [ w for w in wordpunct_tokenize(question) if not re.match(r'[\'\"\.\?\!\,\/\\\(\)\`]',w) ]
	punct    = [ p for p in wordpunct_tokenize(question) if re.match(r'[\'\"\.\?\!\,\/\\\(\)\`]',p) ]
	top_toks = set([ w.lower() for t in obj['topics'] 
						for w in wordpunct_tokenize(t['name']) ])
	qn_toks  = set(tokens)
	qn_topic_words = len(top_toks & qn_toks)

	qn_mark   = 1 if "?" in question else -1 
	start_cap = 1 if re.match(r'^[A-Z]',question) else -1
	if tokens:
		qn_type = [ 1 if sum(1.0 for w in tokens if w in qws) else 0
						for qws in qn_type_words ]
		nm_pres = sum(1.0 for w in tokens if w.lower() in names
							and re.match(r'^[A-Z]',w))
		pl_pres = sum(1.0 for w in tokens if w.lower() in places
							and re.match(r'^[A-Z]',w))
	else:
		qn_type = [-1.0]*len(qn_type_words)
		nm_pres = -1.0
		pl_pres = -1.0

#	qn_somewhere =  1 if sum(qn_type) and (re.match(r'\?$',question)
#						or re.match(r'\?\s*[A-Z]',question)) else -1

	total_words = len(tokens)
	#dict_words  = sum(1 for w in tokens if w.lower() in eng_words)
	correct_form_count = sum(1.0 for w in tokens
			if (w.lower() in eng_words and not re.match(r'^[A-Z]+$',w))
			or re.match(r'^[A-Z]',w)
		)
	question_form = 1 if '?' in punct and sum(1 for w in tokens if w in qn_words) else -1
	#correct_form_ratio = correct_form_count/float(total_words+1)
	topic_word_ratio  = qn_topic_words/float(total_words+1)
	name_ratio        = (nm_pres + pl_pres)/float(total_words+1)
	punctuation_ratio = len(punct)/float(total_words+1)
	result = [
				1 if nm_pres else 0,
				1 if pl_pres else 0,
				qn_mark,
				start_cap,
		#		qn_somewhere,
		#		correct_form_ratio,
		#		len(punct),
				math.log(len(topics)+1),
		#		name_ratio,
				topic_word_ratio,
				1 if qn_topic_words else 0,
				correct_form_count,
				math.log(total_words+1)
			] + qn_type
	return result
def get_model(**args):
	word_counter = CountVectorizer(
			tokenizer=wordpunct_tokenize,
			stop_words=stopwords,
		#	binary=True,
			ngram_range=(1,1),
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
		('scaler', StandardScaler()),
	])

	def word_scorer(x):
		res = {}
		tokens = wordpunct_tokenize(x)
		for i,w in enumerate(tokens):
			w = w.lower()
			if w not in stopwords and len(w) > 3:
				res[w] = res.get(w,0) + 1/(i+1)  #math.exp(-i*len(tokens)) + 1
		return res


	question = Pipeline([
		('extract', Extractor(lambda x: x['question_text'])),
	#	('counter', word_counter),
		('word_s', Extractor(word_scorer)),
		('counter',DictVectorizer()),
		('f_sel',   SelectKBest(score_func=lambda X,Y:f_regression(X,Y,center=False),k=args['question_K'])),#80
	#	('cluster',MiniBatchKMeans(n_clusters=8))
	])
	topics = Pipeline([
		('extract',Extractor(lambda x: {
			t['name']:1 for t in x['topics']
		})),
	#	('counter', FeatureHasher(n_features=2**16+1, dtype=np.float32)),
		('counter',DictVectorizer()),
		('f_sel',   SelectKBest(score_func=lambda X,Y:f_regression(X,Y,center=False),k=args['topics_K'])),#180
	#	('cluster', MiniBatchKMeans(n_clusters=55))
	#	('cluster',MiniBatchKMeans(n_clusters=8))
	])

	ctopic = Pipeline([
		('extract',Extractor(lambda x:
			{ x['context_topic']['name']:1 }
			if x['context_topic'] else {})),
		('counter',FeatureHasher(n_features=2**10+1, dtype=np.float)),
		#('counter',DictVectorizer()),
		('f_sel',   SelectKBest(score_func=lambda X,Y:f_regression(X,Y,center=False),k=args['ctopics_K'])),#45
	])

	topic_question = Pipeline([
		('content',FeatureUnion([
			('question', question),
			('topics',   topics)
		])),
	])
	others = Pipeline([
		('extract', Extractor(lambda x: [
			1 if x['anonymous'] else 0,
			1 if x['promoted_to'] else 0,
			1 if x['num_answers'] else 0,
			#math.log(x['num_answers']+1),
			math.log(x['promoted_to']+1),
			math.log(x['promoted_to']+1) - math.log(sum(t['followers'] for t in x['topics'])+1),
			x['promoted_to']/float(sum(t['followers'] for t in x['topics'])+1),
			math.log(x['num_answers']+1) - math.log(sum(t['followers'] for t in x['topics'])+1),
			x['num_answers']/float(sum(t['followers'] for t in x['topics'])+1),
		])),
		('scaler',  StandardScaler())
	])


	followers = Pipeline([
		('extract',Extractor(lambda x: [
			math.log(sum(t['followers'] for t in x['topics'])+args['smoother'])
		])),
	#	('scaler' ,StandardScaler())
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
		('regress',RidgeCV(alphas=[ 0.1**(-i) for i in range(5)]))
	#	('regress',SGDRegressor(alpha=1e-3,n_iter=1500))

	])
	return model


if __name__ == '__main__':
	training_count = int(sys.stdin.next())
	training_data  = [ json.loads(sys.stdin.next()) for _ in xrange(training_count) ]
	target         = [ math.log(obj['__ans__']+0.9) for obj  in training_data ]

	model = get_model(**{'question_K': 70, 'ctopics_K': 40, 'topics_K': 210, 'smoother': 1})
	model.fit(training_data,target)
	#sys.stderr.write(' '.join(vocabulary)+"\n")
	#sys.stderr.write("%s\n"%counter.transform([' '.join(vocabulary)]))

	test_count = int(sys.stdin.next())
	test_data  = [ json.loads(sys.stdin.next()) for _ in xrange(test_count) ]

	for i,j in zip(model.predict(test_data).tolist(),test_data):
		print json.dumps({ 
			'__ans__':math.exp(i)-0.9,'question_key':j['question_key']
		})


