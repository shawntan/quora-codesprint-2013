import sys,json,math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import stats
from nltk.tokenize import wordpunct_tokenize
training_count = int(sys.stdin.next())
training_data  = [ json.loads(sys.stdin.next()) for _ in xrange(training_count) ]

do = lambda x: sum(t['followers'] for t in x['topics']) 

X = np.array([do(i) for i in training_data ])
Y = np.array([i['__ans__']+1 for i in training_data])

A = np.vstack([X,np.ones(len(training_data))]).T

m,c = np.linalg.lstsq(A,Y)[0]
pearR = np.corrcoef(X,Y)[1,0]

plt.xlabel('Total no. of Followers')
plt.ylabel('Interest')
m,c,r,p_value,std_err = stats.linregress(X,Y)
plt.scatter(X,Y)
plt.plot(X,X*m + c,color='red',label="$r = %6.2e$"%(r))
plt.legend(loc=3)
plt.show()


X = np.log(X+0.9)
Y = np.log(Y+0.9)

A = np.vstack([X,np.ones(len(training_data))]).T

m,c,r,p_value,std_err = stats.linregress(X,Y)
plt.xlabel('Total no. of Followers (log)')
plt.ylabel('Interest (log)')
plt.scatter(X,Y)
plt.plot(X,X*m + c,color='red',label="$r = %6.2e$"%(r))
plt.legend(loc=3)
plt.show()
