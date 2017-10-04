import numpy as np 
import matplotlib.pyplot as plt

def FOeuler(FOLDE,x0,t0,tf,dt=1e-5):
	"""First Order Euler Engine

	arguments
	---------
	FOLDE: defined function name of first order equation
	x0: initial pos - scalar
	t0: initial time - scalar
	tf: final t - scalar
	**dt: time step value - scalar

	returns
	-------
	T: time values - numpy array
	X: position values - numpy array
	"""

	# Create numpy array of time values
	T = np.arange(t0,tf+dt,dt)

	# Create empty numpy array of postion values
	X = np.zeros(len(T))

	# Add initial position to position array
	X[0] = x0

	# first order engine
	for i in range(len(T[:-1])):
		# filling in Y with euler engine
		X[i+1] = X[i] + FOLDE(T[i])*dt

	return T,X

def SOeuler(SOLDE,x0,v0,t0,tf,dt=1e-5):
	"""Second Order Euler Engine

	arguements
	----------
	SOLDE: defined function for second order equation
	x0: inital position - scalar
	v0: initial velocity - scalar
	t0: initial time - scalar
	tf: final time - scalar
	**dt: time step - scalar

	returns
	-------
	T: time values - numpy array
	X: position values - numpy array
	"""

	# define X, Y, and Y' arrays
	T = np.arange(t0,tf+dt,dt)
	X = np.zeros(len(T))
	V = np.zeros(len(T))

	# initalize Y and Y' arrays
	X[0],V[0] = x0,v0

	#SO engine
	for i in range(len(T[:-1])):
		X[i+1] = X[i] + V[i]*dt
		V[i+1] = V[i] + SOLDE(T[i],X[i],V[i])*dt

	return T,X

def FOLDE(t):
	"""function of y'
	say y-prime is cos(x)
	we know y should then be sin(x)

	arguments
	---------
	t: scalar or array

	returns
	-------
	x: scalar or array
	"""
	x = np.cos(t)
	return x

def SOLDE(x,y,yp):
	"""function of y''
	lets take Agenda 9 problem a.i
	y'' - 6y' + 9y = 0
	y(-1) = 1
	y'(-1) = 7


	arguments
	---------
	x: scalar
	y: position - scalar
	yp: velocity - scalar

	returns
	-------
	a: acceleration - scalar
	"""
	a = 6*yp - 9*y
	return a

def show(size=(15,7.5),fs=20):
	""" uses the matplotlib library for plotting
	compares numpy's sin function with euler engine results

	arguments
	---------
	** size: figure size - tuple
	** fs: fontsize - scalar

	returns
	-------
	matplotlib figur with 4 subplots
	"""

	# calls FOeuler to define X and Y arrays for euler engine
	X,Y = FOeuler(FOLDE,0,0,4*np.pi)

	# analytical FOLDE array
	Y1 = np.sin(X)

	# absolute error array between FOeuler and analytical
	Err1 = Y1-Y

	# results of SOeuler
	X3,Y3 = SOeuler(SOLDE,1,7,-1,1)

	# analytical array of SOLDE
	Y4 = (4 + 3*X3) * np.exp(3*X3 + 3)

	# absolute error array between SOeuler and analytical
	Err2 = Y4-Y3

	# defines a matplotlib figure with defined figuresize
	fig = plt.figure(figsize=size)

	# defines subplot 1 (on a 2 x 2 set) (upper left)
	ax1 = plt.subplot(221)
	# sets title for axis 1
	ax1.set_title('FO Euler and np.sin()', fontsize=fs+2)
	# defines x and y labels
	ax1.set_xlabel('radians', fontsize=fs)
	ax1.set_ylabel('sin(x)', fontsize=fs)
	# plots X,Y arrays (FO euler) and gives it a linestyle, color, and label
	ax1.plot(X,Y,'-', color='b', label='FO Euler')
	# plots X,Y2 array for numpy's sin function
	ax1.plot(X,Y1,'-', color='r', label='analytical')
	# makes a legend with defined labels in the 'best' location
	ax1.legend(loc='best')
	# set the limits along the x and y axies
	ax1.set_xlim([min(X),max(X)])
	ax1.set_ylim([min(Y)-.1,max(Y)+.1])

	# defines a second subplot (upper right)
	ax2 = plt.subplot(222)
	ax2.set_title('Absolute Error for FOLDE', fontsize=fs+2)
	ax2.set_xlabel('radians',fontsize=fs)
	ax2.set_ylabel('ERR',fontsize=fs)
	# plots the absolute err between FOeuler and analytical function
	ax2.plot(X,Err1, color='m', label='error')
	ax2.legend(loc='best')
	ax2.set_xlim([min(X),max(X)])
	ax2.set_ylim([min(Err1),max(Err1)])

	# defines subplot (lower left)
	ax3 = plt.subplot(223)
	ax3.set_title('Agenda problem 9 a i',fontsize=fs+2)
	ax3.set_xlabel('time [sec]',fontsize=fs)
	ax3.set_ylabel('position [m]', fontsize=fs)
	# plot SOeuler results
	ax3.plot(X3,Y3, color='b', label='SO Euler')
	# plot analytical function
	ax3.plot(X3,Y4, color='r', label='analytical')
	ax3.legend(loc='best')
	ax3.set_xlim([min(X3),max(X3)])
	#ax3.set_ylim([(min(Y3))])

	# subplot (lower right)
	ax4 = plt.subplot(224)
	ax4.set_title('Absolute Error for SOLDE', fontsize=fs+2)
	ax4.set_xlabel('time [s]', fontsize=fs)
	ax4.set_ylabel('ERR', fontsize=fs)
	# plot absolute error between the SOeuler and analytical function
	ax4.plot(X3,Err2, color='m', label='error')
	ax4.legend(loc='best')
	ax4.set_xlim([min(X3),max(X3)])

	plt.tight_layout()
	plt.show()

	# save fig
	fig.savefig('engine_test.png')