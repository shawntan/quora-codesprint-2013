"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""
import sys,json,math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from nltk.tokenize import wordpunct_tokenize
eps = 1
training_count = int(sys.stdin.next())
training_data  = [ json.loads(sys.stdin.next()) for _ in xrange(training_count) ]

do = lambda x: sum(t['followers'] for t in x['topics'])+1
#do = lambda x: len(wordpunct_tokenize(x['question_text']))


X = np.array([do(i) for i in training_data ])
mu, sigma = np.mean(X), np.std(X)
print mu,sigma
fig = plt.figure()
ax = fig.add_subplot(111)

# the histogram of the data
n, bins, patches = ax.hist(X, facecolor='green', alpha=0.75,normed=True)

# hist uses np.histogram under the hood to create 'n' and 'bins'.
# np.histogram returns the bin edges, so there will be 50 probability
# density values in n, 51 bin edges in bins and 50 patches.  To get
# everything lined up, we'll compute the bin centers
bincenters = 0.5*(bins[1:]+bins[:-1])
# add a 'best fit' line for the normal PDF
y = mlab.normpdf( bincenters, mu, sigma)
l = ax.plot(bincenters, y, 'r--', linewidth=1)

ax.set_xlabel('No. of Followers')
ax.set_ylabel('Frequency')
#ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
#ax.set_xlim(40, 160)
#ax.set_ylim(0, 0.03)
ax.grid(True)

plt.show()
