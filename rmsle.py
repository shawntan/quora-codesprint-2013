import sys,json,math
total_count = 0
total_log_error_sq = 0
I = [ json.loads(i) for i in open(sys.argv[1],'r') ]
J = [ json.loads(i) for i in open(sys.argv[2],'r') ]
I.sort(key=lambda x:x.get('question_key'))
J.sort(key=lambda x:x.get('question_key'))
for i,j in zip(I,J):
	assert(i['question_key'] == j['question_key'])
	total_log_error_sq += ( math.log(i['__ans__']+1) - math.log(j['__ans__']+1))**2
	total_count += 1

msle = ( 0.5 / math.sqrt( total_log_error_sq/total_count) ) * 100
print "Accuracy: %0.3f%% "%msle
