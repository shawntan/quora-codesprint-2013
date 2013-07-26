import sys,json


total_count = 0
correct_count = 0
for i,j in zip(open(sys.argv[1],'r'),open(sys.argv[2],'r')):
	i = json.loads(i)
	j = json.loads(j)
	assert(i['question_key'] == j['question_key'])
	if i['__ans__'] == j['__ans__']: correct_count += 1
	total_count += 1

print "Accuracy: %0.3f"%(100*float(correct_count)/total_count)
