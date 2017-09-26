# import libraries needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import sympy as sy
matrix  =   sy.Matrix
diff    =   sy.diff

# define symbols
x,y,z   =   sy.symbols('x y z')
a,b,c   =   sy.symbols('a b c')

# define CC basis vectors
I       =   sy.eye(3)
xhat    =   I.row(0)
yhat    =   I.row(1)
zhat    =   I.row(2)

"""
vectors: 1D sympy matricies
scalars: sympy symbols
scalar functions: functions of scalars
"""

def dot(v1,v2):
    """ find dot product between two vectors

    args
    ----
    v1: vector 1
    v2: vector 2

    returns
    -------
    scalar
    """
    return v1.dot(v2)

def cross(v1,v2):
    """ find cross product between two vectors

    args
    ----
    v1: vector 1
    v2: vector 2

    returns
    -------
    vector
    """
    return v1.cross(v2).T

def gradient(s):
    """ find gradient of scalar function

    args
    ----
    s:  scalar function

    returns
    -------
    vector
    """
    return xhat*diff(s,x) + yhat*diff(s,y) + zhat*diff(s,z)

def curl(v):
    """ find curl of vector

    args
    ----
    v:  vector

    returns
    -------
    vector
    """
    x1  =   diff(v[2],y) - diff(v[1],z)
    y1  =   diff(v[0],z) - diff(v[2],x)
    z1  =   diff(v[1],x) - diff(v[0],y)
    return x1*xhat + y1*yhat + z1*zhat

def divergence(v):
    """ find divergence of vector

    args
    ----
    v:  vector

    returns
    -------
    scalar
    """
    return diff(v[0],x) + diff(v[1],y) + diff(v[2],z)

def laplacian(s):
    """ find laplacian of scalar function

    args
    ----
    s:  scalar function

    returns
    -------
    scalar
    """
    return divergence(gradient(s))

# scalar functions
s           =   {}
s['phi']    =   x**2 + y**2 + z**2
s['chi']    =   sy.ln(x*y/z)
s['psi']    =   sy.exp(x**2 + y**2)
s['f']      =   x * sy.cos(y * sy.sin(z))
s['varphi'] =   z / sy.tan(x**2 + y**2)
s['xi']     =   z / ( (x-a)**2 - (y-b)**2 + (z-c)**2 )

# vector fields
v           =   {}
v['v1']     =   x*y*xhat + y*z*yhat + x*z*zhat
v['v2']     =   sy.ln(x)*xhat + sy.ln(y)*yhat + sy.ln(z)*zhat
v['v3']     =   y*z*xhat + x*z*yhat + x*y*zhat
v['v4']     =   x**2*xhat + y**2*yhat + z**2*zhat
v['v5']     =   (x-y)**2*xhat + (y-z)**2*yhat + (z-x)**2*zhat
v['v6']     =   xhat/x + yhat/y + zhat/z

# results
d           =   {}

# gradients of scalar functions (a)
for key in s:
    key1    =   'grad_'+key
    d[key1] =   gradient(s[key])

# divergences of vector fields (b)
for key in v:
    key1    =   'div_'+key
    d[key1] =   divergence(v[key])

# curls of vector fields (c)
for key in v:
    key1    =   'curl_'+key
    d[key1] =   curl(v[key])

# curls of gradients of scalar functions (d)
for key in s:
    key1    =   'cg_'+key
    key2    =   'grad_'+key
    grad    =   d[key2]
    d[key1] =   curl(grad)

# laplacian of scalar fields (e)
for key in s:
    key1    =   'lap_'+key
    d[key1] =   laplacian(s[key])

# plot scalar field (contour and surface)
def plot_scalar(s,zconst=1,dx=.1,xRange=5,col=cm.coolwarm,savename=None):
    """ plot a scalar function on the x-y plane

    args
    ----
    s:          sympy scalar function
    zconst:     ** constant z, default = 1
    dx:         ** grid spacing, default = .1
    xRange:     ** max X and Y, default = 10
    col:        ** color map, default = cm.coolwarm
    savename:   ** path and filen name to save figure, default = None

    returns
    -------
    none: saves fig if savename != True
    """
    # close any open figures
    if s==0:
        print("s = 0, not making plot.")
        return
    plt.close('all')

    # make XY grid
    XR      =   np.arange(-xRange,xRange+dx,dx)
    YR      =   np.array(XR, copy=True)
    X,Y     =   np.meshgrid(XR,YR)

    # convert sybolic function to python function
    f       =   sy.lambdify((x,y,z),s)

    # calculate surface height/contour value
    Z       =   f(X,Y,zconst)

    # construct figure
    fig     =   plt.figure(figsize=(15,15))
    ax      =   fig.add_subplot(111, projection='3d')
    ax.set_title("z = %s" % zconst, fontsize=20)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    surf    =   ax.plot_surface(X,Y,Z, cmap=col, alpha=.8)
    # contx   =   ax.contourf(X,Y,Z, zdir='x', offset=-sRange, cmap=col)
    # conty   =   ax.contourf(X,Y,Z, zdir='y', offset=sRange, cmap=col)
    contz   =   ax.contourf(X,Y,Z, zdir='z', offset=-abs(np.min(Z)), cmap=col)
    fig.colorbar(surf)
    ax.set_aspect(1)

    if savename != None:
        fig.savefig(savename+'.png')
        plt.close()
    else:
        plt.show()

# plot vector field
def plot_field(v,zconst=1,dx=.5,xRange=5,savename=None):
    """ plot a vector field

    args
    ----
    v:          vector field
    zconst:     ** constant z, default = 1
    dx:         ** grid spacing, default = .5
    xRange:     ** half the grid span, default = 5
    savename:   ** path and file name to save figure, default = None

    returns
    -------
    none: saves fig if savename != True
    """
    if v[0]==v[1]==v[2]==0:
        print(v + ' = 0, not making plot')
        return
    plt.close('all')

    # X       =   np.arange(-xRange,xRange+dx,dx)
    X       =   np.linspace(-xRange,xRange,20)
    Y       =   np.array(X, copy=True)
    X,Y     =   np.meshgrid(X,Y)

    f       =   sy.lambdify((x,y,z),v)

    Z       =   f(X,Y,zconst)
    U       =   Z[0,0]
    V       =   Z[0,1]

    fig     =   plt.figure(figsize=(10,10))
    plt.title('z = %s' % zconst, fontsize=20)
    plt.xlabel('x',fontsize=15)
    plt.ylabel('y',fontsize=15)
    Q       =   plt.quiver(X,Y,U,V,units='width')

    if savename != None:
        fig.savefig(savename+'.png')
        plt.close()
    else:
        plt.show()

# quickly generate all plots in a new directory
def quick_plot(dirname='Agenda1Plots'):
    dirname     =   str(dirname)
    import os

    if not os.path.exists(dirname):
        try:
            os.stat(dirname)
        except:
            os.mkdir(dirname)

    if not os.path.exists(dirname+'/surface'):
        try:
            os.stat(dirname+'/surface')
        except:
            os.mkdir(dirname+'/surface')

    if not os.path.exists(dirname+'/field'):
        try:
            os.stat(dirname+'/field')
        except:
            os.mkdir(dirname+'/field')

    for key in s:
        try:
            plot_scalar(s[key], savename=dirname+'/surface/'+key)
        except:
            print("error in plotting a scalar surface")
        try:
            plot_field(d['grad_'+key], savename=dirname+'/field/grad_'+key)
        except:
            print("error in plotting a gradient field")
        try:
            plot_scalar(d['lap_'+key], savename=dirname+'/surface/lap_'+key)
        except:
            print("error in plotting a laplaciean surface")

    for key in v:
        try:
            plot_field(v[key], savename=dirname+'/field/'+key)
        except:
            print("error in plotting a vector field.")
        try:
            plot_div(d['div_'+key], savename=dirname+'/surface/'+key)
        except:
            print("error in plotting a divergence surface.")
        try:
            plot_field(d['curl_'+key], savename=dirname+'/field/curl_'+key)
        except:
            print("error in plotting a curl field.")
