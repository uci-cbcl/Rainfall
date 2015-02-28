"""
Created on Sun Feb 22 20:17:14 2015

@author: Elmira
"""

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from multiprocessing import pool
import multiprocessing


def run_loop(X,T, y,par_num):
	if par_num == 1:
		# Fit regression model
		svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
		y_rbf = svr_rbf.fit(X, y).predict(T)
		rbf_output_file = open('/cbcl/forouzme/HW/ML_project/results/rbf_output.txt')
		rbf_output_file.write('ID,Prediction\n')
		for i,p in enumerate(y_rbf):
			line = str(i)+ ',' + str(p)+'\n'
			rbf_output_file.write(line)
		rbf_output_file.close()
		print "hah_ one done"
	elif par_num == 2:
		# Fit regression model
		svr_lin = SVR(kernel='linear', C=1e3)
		y_lin = svr_lin.fit(X, y).predict(X)
		lin_output_file = open('/cbcl/forouzme/HW/ML_project/results/lin_output.txt')
		lin_output_file.write('ID,Prediction\n')
		for i,p in enumerate(y_lin):
			line = str(i)+ ',' + str(p)+'\n'
			lin_output_file.write(line)
		lin_output_file.close()
		print "hah_ two done"	
	else:	
		# Fit regression model
		svr_poly = SVR(kernel='poly', C=1e3, degree=2)
		y_poly = svr_poly.fit(X, y).predict(X)
		poly_output_file = open('/cbcl/forouzme/HW/ML_project/results/poly_output.txt')
		poly_output_file.write('ID,Prediction\n')
		for i,p in enumerate(y_poly):
			line = str(i)+ ',' + str(p)+'\n'
			poly_output_file.write(line)
		poly_output_file.close()
		print "hah_ three done"
def main():
	###############################################################################
	# Generate sample data
	X1 = np.loadtxt('/cbcl/forouzme/HW/ML_project/data/kaggle.X1.train.txt', delimiter=',')
	X2 = np.loadtxt('/cbcl/forouzme/HW/ML_project/data/kaggle.X2.train.txt', delimiter=',')    
	X = np.concatenate((X1,X2), axis=1)
	y = np.loadtxt('/cbcl/forouzme/HW/ML_project/data/kaggle.Y.train.txt', delimiter=',')
	T1 = np.loadtxt('/cbcl/forouzme/HW/ML_project/data/kaggle.X1.test.txt', delimiter=',')
	T2 = np.loadtxt('/cbcl/forouzme/HW/ML_project/data/kaggle.X2.test.txt', delimiter=',')
	T = np.concatenate((T1,T2), axis=1)	
	p = multiprocessing.Process(target=run_loop, args=(X,T,y,[1,2,3]))
	p.start()
	


"""
	# look at the results
	plt.scatter(X, y, c='k', label='data')
	plt.hold('on')
	plt.plot(X, y_rbf, c='g', label='RBF model')
	plt.plot(X, y_lin, c='r', label='Linear model')
	plt.plot(X, y_poly, c='b', label='Polynomial model')
	plt.xlabel('data')
	plt.ylabel('target')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()
"""
if __name__== "__main__":
	main()
