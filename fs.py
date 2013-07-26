topic_fd  = FreqDist()
topic_pos = FreqDist()
topic_neg = FreqDist()

for instance in training_data:
	topic_names = [w.lower() for w in wordpunct_tokenize(instance['question_text']) if len(w) > 3 and w.lower() not in stopwords]
	topic_names.sort()
	for t in itertools.combinations(topic_names,2):
		if instance['__ans__']: topic_pos.inc(t)
		else: topic_neg.inc(t)
		topic_fd.inc(t)
topic_scores = {}
pos_topic_count = topic_pos.N()
neg_topic_count = topic_neg.N()
total_topic_count = pos_topic_count + neg_topic_count
for topic, freq in topic_fd.iteritems():
	if freq < 5: continue
	pos_score = BigramAssocMeasures.chi_sq(
			topic_pos[topic],(freq, pos_topic_count), total_topic_count)
	neg_score = BigramAssocMeasures.chi_sq(
			topic_neg[topic],(freq, neg_topic_count), total_topic_count)
	topic_scores[topic] = pos_score + neg_score
bestwords = set( w for w,s in
					sorted(
						topic_scores.iteritems(),
						key=lambda (w,s): s,
						reverse=True
					)[:50])

