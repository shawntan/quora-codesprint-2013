import sys,json,math
from itertools import *
from qn1       import get_model
if __name__ == '__main__':
	training_count = int(sys.stdin.next())
	training_data  = [ json.loads(sys.stdin.next()) for _ in xrange(training_count) ]
	target         = [ obj['__ans__'] for obj  in training_data ]
	n_folds    = 10
	fold_size  = len(training_data)/n_folds
	fold_train = [ training_data[:f*fold_size]+training_data[(f+1)*fold_size:] for f in range(n_folds) ]
	fold_test  = [ training_data[f*fold_size:(f+1)*fold_size] for f in range(n_folds) ]

	fold_target_train = [ target[:f*fold_size]+target[(f+1)*fold_size:] for f in range(n_folds) ]
	fold_target_test  = [ target[f*fold_size:(f+1)*fold_size] for f in range(n_folds) ]


	test_count = int(sys.stdin.next())
	test_data  = [ json.loads(sys.stdin.next()) for _ in xrange(test_count) ]
	answers    = [ json.loads(line)['__ans__'] for line in open(sys.argv[1],'r') ]
	hyper_params = [
    	('max_n_grams', [1,2,3]),
        ('class_alpha', [ 10**(-i)   for i in range(4) ]),
        ('class_iters', [ 100*i      for i in range(10,20)]),
		('smoother',    [ 10**(-i)   for i in range(4) ]),
		('question_K',  [ 10*i       for i in reversed(range(1,8)) ]),
		('topics_K',    [ 10*i       for i in range(20,30) ]),
		('ctopics_K',   [ 10*i       for i in range(3,8) ]),
	]
	param_vals = [p for _,p in hyper_params]
	max_acc = 0.0
	max_param = None
	for i in product(*param_vals):
		param = dict( pair for pair in izip((n for n,_ in hyper_params),i) )
		total_acc = 0
		print "New parameters:",param
		for j,(train,test,target_train,target_test) in enumerate(izip(fold_train,fold_test,fold_target_train,fold_target_test)):
			print "\tFold no. %d"%j
			model = get_model(**param)
			model.fit(train,target_train)
			total_count = 0
			match_count = 0
			for i,j in zip(model.predict(test).tolist(),target_test):
				total_count += 1
				match_count += 1 if i==j else 0
			total_acc += 100*match_count/float(total_count)
		acc = total_acc/float(n_folds)
		if max_acc < acc:
			max_acc,max_param = acc,param
			print "Accuracy: %0.3f"%(acc)
			print "Suggested parameters:", param
	print "Suggested parameters:", max_param 
