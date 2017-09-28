import sympy as sy
from sympy.solvers import pdsolve
import numpy as np
from numpy import pi,sin,cos,exp,sinh
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as ani
import matplotlib.animation as manimation
import pdb
import os
import PDE_analytic as pa

#===============================================================================
""" auxillary functions """
#===============================================================================

def directory_checker(dirname):
    """ check if a directory exists, creates it if it doesn't"""
    dirname =   str(dirname)
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
        except:
            os.stat(dirname)

def set_mpl_params():
    mpl.rcParams['xtick.labelsize']=15
    mpl.rcParams['ytick.labelsize']=15

    mpl.rcParams['legend.loc']='best'
    mpl.rcParams['legend.numpoints']=1
    mpl.rcParams['legend.fontsize']=15

    mpl.rcParams['lines.linewidth']=3
    mpl.rcParams['lines.markersize']=20

    mpl.rcParams['font.size']=10

    mpl.rcParams['figure.figsize']=(15,15)
    mpl.rcParams['figure.titlesize']=27
    mpl.rcParams['figure.titleweight']='bold'

    mpl.rcParams['axes.titlesize']=25
    mpl.rcParams['axes.titleweight']='bold'
    mpl.rcParams['axes.linewidth']=1
    mpl.rcParams['axes.labelsize']=20
    mpl.rcParams['axes.labelpad']=10
set_mpl_params()

#===============================================================================
""" general parameters """
#===============================================================================

a,b =   1, 1

X   =   np.linspace(0,a,100)
Y   =   np.linspace(0,b,100)
X,Y =   np.meshgrid(X,Y)

Nmax    =   100
N       =   np.arange(1,Nmax+1)
N       =   N*2 - 1

#===============================================================================
""" square drum """
directory_checker('square_drum/')
dirname     =   'square_drum/single_frequency/'
directory_checker(dirname)
#===============================================================================

def Z_tmn(t,m,n,omega):
    """ square drum surface values at time t """

    # amplitude factor outside of sum
    A   =   ep*(2/pi)**6

    # wavenumbers and frequency
    km      =   m*pi/a
    kn      =   n*pi/b

    return A/(m*n) * sin(km*X) * sin(kn*Y) * cos(omega*t)

def square_drum(N_lowest=10,plots=True,movies=True):

    """anugular frequencies"""
    def omega_mn(m,n):
        return np.sqrt( (pi*v)**2 * ( (m/a)**2 + (n/b)**2 ) )

    """select the N lowest frequencies and their m,n values"""
    def find_m_n_omega():
        Irange  =   int(N_lowest/2)
        # start with some m,n integers 1 - 5
        M,N     =   np.arange(1,Irange),np.arange(1,Irange)
        # select only odd values
        M,N     =   2*M-1,2*N-1
        # make empty matrix of omega values
        O       =   np.zeros( (len(M),len(N)) )
        # fill in omega values
        for i,m in enumerate(M):
            for j,n in enumerate(N):
                O[i,j]  =   omega_mn(m,n)
        # flatten omega matrix to 1D
        O1      =   O.flatten()
        # select indecies of N lowest frequencies
        O1      =   O1.argsort()[:N_lowest]
        # find indecies in relation to m,n
        O1      =   divmod(O1,Irange-1)
        Mi,Ni   =   O1[0],O1[1]
        # pdb.set_trace()
        # 1D arrays of m,n combinations that make N lowest frequencies
        M       =   np.array([ M[ Mi[i] ] for i in range(N_lowest) ])
        N       =   np.array([ N[ Ni[i] ] for i in range(N_lowest) ])

        # create 1D array of N lowest frequency values
        O       =   np.array([ O[Mi[i],Ni[i]] for i in range(N_lowest) ])
        return M,N,O

    M,N,Omega   =   find_m_n_omega()

    # plot single surface of single: t,m,m
    def plot_frequency(i,t=0):

        plt.close('all')
        Zt      =   Z_tmn(t,M[i],N[i],Omega[i])
        fig     =   plt.figure()
        ax      =   fig.gca(projection='3d')
        ax.set_title('t = %s, $\omega$ = %.2f, m = %s, n = %s, a = %s, b = %s, v = %s, $\epsilon$ = %s' % (t,Omega[i],M[i],N[i],a,b,v,ep))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect(1)
        surf    =   ax.plot_surface(X,Y,Zt,cmap=cm.viridis, alpha=.8)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        Zmin,Zmax   =   np.min(Zt),np.max(Zt)
        ax.set_zlim(.9*Zmin, 1.1*Zmax)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        filename = 'm_%s_n_%s.png' % (M[i],N[i])
        fig.savefig(dirname + filename)
        plt.close('all')

    if plots:
        for i in range(N_lowest):
            plot_frequency(i)

    def mk_movie():
        directory_checker('square_drum/')

        FFMpegWriter    =   manimation.writers['ffmpeg']
        metadata        =   dict(title='Square Drum', artist='Matplotlib')
        writer          =   FFMpegWriter(fps=10, metadata=metadata)

        fig             =   plt.figure()
        ax              =   fig.gca(projection='3d')

        with writer.saving(fig, "square_drum/Square_drum_animation.mp4", 100):
            for i in range(N_lowest):
                fig.clear()
                ax              =   fig.gca(projection='3d')

                period          =   2*pi/Omega[i]
                T               =   np.linspace(0,period,10)
                Z0              =   Z_tmn(0,M[i],N[i],Omega[i])
                Zmax            =   np.max(Z0)

                surf            =   ax.plot_surface(X,Y,Z0, cmap=cm.seismic)
                fig.colorbar(surf, shrink=0.5, aspect=5)

                for t in T:
                    Z   =   Z_tmn(t,M[i],N[i],Omega[i])
                    ax.clear()
                    ax.set_title('t = %.2f, $\omega$ = %.2f, m = %s, n = %s, a = %s, b = %s, v = %s, $\epsilon$ = %s' % (t,Omega[i],M[i],N[i],a,b,v,ep))
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlim(-1.1*Zmax, 1.1*Zmax)
                    ax.zaxis.set_major_locator(LinearLocator(10))
                    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                    ax.plot_surface(X,Y,Z, cmap=cm.seismic, vmin=-Zmax, vmax=Zmax)
                    writer.grab_frame()

    if movies: mk_movie()

    return

#===============================================================================
""" temperature plate """
dirname1    =   'temperature_plate/'
directory_checker(dirname)

v,ep    =   5, .2
Theta0  =   350
Theta1  =   200
Theta3  =   400

x       =   sy.symbols('x',positive=True,real=True)
x0      =   sy.Rational(a,2)
sig     =   sy.Rational(a,4)
f1      =   Theta3/sy.sqrt(2*pi*sig**2) * sy.exp(-(x-x0)**2 / (2*sig**2) )
f2      =   Theta3*sy.cos( pi*(x-a/2) )
#===============================================================================

def Bn1(n,theta):
    kn  =   n*pi/a
    return 2*theta*(1-(-1)**n) / ( n*pi*sinh(kn*b) )

def Theta_n1(n,Y=Y,theta=Theta0):
    kn  =   n*pi/a
    return Bn1(n,theta) * sin(kn*X) * sinh(kn*Y)

def Bn3(n,f):
    kn  =   n*pi/a
    f1  =   f * sy.sin(kn*x)
    f2  =   sy.integrate(f1,(x,0,a))
    bn  =   2/(a*sinh(kn*b)) * f2
    return bn.evalf()

def Theta_n3(n,f):
    k   =   n*pi/a
    B   =   float(Bn3(n,f))
    Z   =   B * sin(k*X) * sinh(k*(b-Y))
    return Z

def SS1():
    """ Theta(0,y) = Theta(a,y) = Theta(x,0) = 0, Theta(x,b) = Theta0 """

    Z   =   np.zeros( (100,100) )
    for n in N:
        Z   +=  Theta_n1(n)

    plt.close('all')
    fig     =   plt.figure()
    ax      =   fig.gca(projection='3d')
    ax.set_title('N$_{\mathrm{max}}$ = %s, $\Theta(0,y) = \Theta(a,y) = \Theta(x,0) = 0, \Theta(x,b) = \Theta_0$' % Nmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect(1)
    surf    =   ax.plot_surface(X,Y,Z, cmap=cm.inferno)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig.savefig(dirname1+'SS1.png')
    plt.close()

    return Z

def SS2(Z1):
    """ Theta(0,y) = Theta(a,y) = 0, Theta(x,0) = Theta1, Theta(x,b) = Theta0 """

    Z   =   np.zeros( (100,100) )
    for n in N:
        Z   +=  Theta_n1(n,Y=b-Y,theta=Theta1)

    plt.close('all')
    fig     =   plt.figure()
    ax      =   fig.gca(projection='3d')
    ax.set_title('N$_{\mathrm{max}}$ = %s, $\Theta(0,y) = \Theta(a,y) = 0, \Theta(x,0) = \Theta_1, \Theta(x,b) = \Theta_0$' % Nmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect(1)
    surf    =   ax.plot_surface(X,Y,Z+Z1, cmap=cm.inferno)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig.savefig(dirname1+'SS2.png')
    plt.close()

    return Z + Z1

def SS3(f=f2):
    """ Theta(0,y) = Theta(a,y) = Theta(x,b) = 0, Theta(x,0) = f(x) """

    Z   =   np.zeros( (100,100) )
    for n in N:
        Z  +=  Theta_n3(n,f2)

    plt.close('all')
    fig     =   plt.figure()
    ax      =   fig.gca(projection='3d')
    ax.set_title('N$_{\mathrm{max}}$ = %s, $\Theta(0,y) = \Theta(a,y) = 0, \Theta(x,b) = 0, \Theta(x,0) = f(x) )$' % Nmax )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    surf    =   ax.plot_surface(X,Y,Z, cmap=cm.inferno)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig.savefig(dirname1+'SS3.png')
    plt.close()

    return Z

def SS4(Z1,Z3):
    """ Theta(0,y) = Theta(a,y) = 0, Theta(x,0) = f(x), Theta(x,b) = Theta_0"""

    plt.close('all')
    fig     =   plt.figure()
    ax      =   fig.gca(projection='3d')
    ax.set_title('N$_{\mathrm{max}}$ = %s, $\Theta(0,y) = \Theta(a,y) = 0, \Theta(x,0) = f(x), \Theta(x,b) = \Theta_0$' % Nmax )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    surf    =   ax.plot_surface(X,Y,Z1+Z3, cmap=cm.inferno)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig.savefig(dirname1+'SS4.png')
    plt.close()

    return Z1+Z3

#===============================================================================
""" 2D diffusion """
directory_checker('diffusion')
dirname2    =   'diffusion/time_slice/'
directory_checker(dirname2)
#===============================================================================

def Theta_t(t,eta=1e-3):

    def Theta_nt(n,t):
        n1  =   2*n-1
        kn  =   n1/(2*b)
        return (-1)**n * 4/(n1*pi) * cos(kn*pi*(b-Y)) * exp(-kn**2 * pi**2 * eta*t)

    total   =   0
    for n in np.arange(1,2*Nmax+1):
        total   +=  Theta_nt(n,t)

    return 350 + 150*total

def plot_diff(t):

    plt.close('all')
    Z       =   Theta_t(t)
    Zmax    =   np.max(Z)

    fig     =   plt.figure()
    ax      =   fig.gca(projection='3d')
    ax.set_title('t = %.2f' % t)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlim(0, 1.1*Zmax)
    ax.view_init(elev=10,azim=20)
    surf    =   ax.plot_surface(X,Y,Z, cmap=cm.inferno)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()
    fig.savefig(dirname2+'t_%.2f.png' % t)
    plt.close()

def mk_diff_movie():
    plt.close('all')

    FFMpegWriter    =   manimation.writers['ffmpeg']
    metadata        =   dict(title='Heat Diffusion in 2D Plate', artist='Matplotlib')
    writer          =   FFMpegWriter(fps=15, metadata=metadata)

    fig             =   plt.figure()
    ax              =   fig.gca(projection='3d')

    with writer.saving(fig, "diffusion/diffusion_animation.mp4", 100):
        T               =   np.linspace(0,100,300)
        Z0              =   Theta_t(0)
        Zmax            =   np.max(Z0)

        surf            =   ax.plot_surface(X,Y,Z0, cmap=cm.inferno, )
        fig.colorbar(surf, shrink=0.5, aspect=5)

        for t in T:
            Z   =   Theta_t(t)
            ax.clear()
            ax.set_title('t = %.0f' % t)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlim(200, 1.1*Zmax)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.plot_surface(X,Y,Z, cmap=cm.inferno, vmin=200, vmax=Zmax)
            ax.view_init(elev=10,azim=20)
            writer.grab_frame()
    plt.close('all')

#===============================================================================
""" Agenda 13 """
directory_checker('Agenda13/')
directory_checker('Agenda13/npy/')
directory_checker('Agenda13/increasing_N')
# directory_checker('Agenda13/increasing_z')

a,b,c   =   1,1.5,1
X       =   np.linspace(0,a,100)
Y       =   np.linspace(0,b,100)
Z       =   np.linspace(0,c,100)

Nmax    =   6
N       =   np.arange(1,Nmax)
N       =   2*N - 1

Yx,Zx   =   np.meshgrid(Y,Z)
Xy,Zy   =   np.meshgrid(X,Z)
Xz,Yz   =   np.meshgrid(X,Y)

y       =   sy.symbols('y',real=True,positive=True)
f1      =   1000 * (x - a/2)**2 - (y - b/2)**2
f2      =   - f1
#===============================================================================

def fX(n,x):
    return sy.sin(n * sy.pi * x / a)

def fY(n,y):
    return sy.sin(n * sy.pi * y / b)

def fZ(n,z):
    k   =   sy.sqrt( (n*sy.pi/a)**2 + (n*sy.pi/b)**2 )
    return sy.sinh(k*z)

def A_n(f,n):
    i1  =   f * fX(n,x)
    i2  =   sy.integrate(i1,(x,0,a)) * fY(n,y)
    i3  =   sy.integrate(i2,(y,0,b))
    ans =   4 * i3 / (a * b * fZ(n,c) )
    return np.float(ans.evalf())

def phi_1n(n,x,y,z):
    kx  =   n*pi/a
    ky  =   n*pi/b
    k   =   np.sqrt( kx**2 + ky**2 )
    x1  =   sin(kx*x)
    y1  =   sin(ky*y)
    z1  =   sinh(k*(c-z))
    ans =   x1 * y1 * z1
    return ans

def phi_2n(n,x,y,z):
    kx  =   n*pi/a
    ky  =   n*pi/b
    k   =   np.sqrt( kx**2 + ky**2 )
    x1  =   sin(kx*x)
    y1  =   sin(ky*y)
    z1  =   sinh(k*z)
    return x1 * y1 * z1

def phi_n(A1,A2,n,x,y,z):
    one =   A1 * phi_1n(n,x,y,z)
    two =   A2 * phi_2n(n,x,y,z)
    ans =   one * two
    return ans

def calculate_PHI_bisection():
    # calculate PHI for plotting and movie
    PHI_x   =   np.zeros( (100,100) )
    PHI_y   =   np.zeros_like(PHI_x)
    PHI_z   =   np.zeros_like(PHI_x)

    Yx,Zx   =   np.meshgrid(Y,Z)
    Xy,Zy   =   np.meshgrid(X,Z)
    Xz,Yz   =   np.meshgrid(X,Y)

    for n in N:
        A1      =   A_n(f1,n)
        A2      =   A_n(f2,n)
        PHI_x   +=  phi_n(A1, A2, n, a/2, Yx,  Zx )
        PHI_y   +=  phi_n(A1, A2, n, Xy,  b/2, Zy )
        PHI_z   +=  phi_n(A1, A2, n, Xz,  Yz,  c/2)

    np.save('Agenda13/npy/PHI_x.npy',PHI_x)
    np.save('Agenda13/npy/PHI_y.npy',PHI_y)
    np.save('Agenda13/npy/PHI_z.npy',PHI_z)

def calculate_PHI_increasing_N():

    PHI_nxy =   np.zeros( (Nmax,100,100) )

    for i,n in enumerate(N):
        print(n)
        A1              =   A_n(f1,n)
        A2              =   A_n(f2,n)
        PHI_nxy[i,:,:]  =   phi_n(A1,A2,n,Xz,Yz,c/2)

    np.save('Agenda13/npy/PHI_nxy.npy',PHI_nxy)

def calculate_PHI_increasing_z():

    PHI_zxy =   np.zeros( (100,100,100) )

    for i,z in enumerate(Z):
        print(z)
        for n in N:
            A1  =   A_n(f1,n)
            A2  =   A_n(f2,n)
            phi =   phi_n(A1,A2,n,Xz,Yz,z)
            PHI_zxy[i,:,:]  +=  phi

    np.save('Agenda13/npy/PHI_zxy.npy',PHI_zxy)

def plot_phi_bisection(color=cm.Blues):

    plt.close('all')

    PHI_x   =   np.load('Agenda13/npy/PHI_x.npy')
    PHI_y   =   np.load('Agenda13/npy/PHI_y.npy')
    PHI_z   =   np.load('Agenda13/npy/PHI_z.npy')

    fig =   plt.figure(figsize=(30,15))

    ax1 =   fig.add_subplot(131)
    ax1.set_title('$\phi(x = a/2,y,z)$')
    ax1.set_xlabel('Y')
    ax1.set_ylabel('Z')
    ax1.set_aspect(1)
    cf1 =   ax1.contourf(Yx,Zx,PHI_x, 100, cmap=color)

    ax2 =   fig.add_subplot(132)
    ax2.set_title('$\phi(x,b/2,z)$')
    ax2.set_xlabel('Z')
    ax2.set_ylabel('Y')
    ax2.set_aspect(1)
    cf2 =   ax2.contourf(Zy,Xy,PHI_y, 100, cmap=color)

    ax2 =   fig.add_subplot(133)
    ax2.set_title('$\phi(x,y,c/2)$')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect(1)
    cf3 =   ax2.contourf(Xz,Yz,PHI_z, 100, cmap=color)

    fig.colorbar(cf1, cmap=color)

    plt.tight_layout()

    fig.savefig('Agenda13/bisection.png')

    plt.close()

def plot_phi_increasing_N(color=cm.Blues):

    PHI     =   np.load('Agenda13/npy/PHI_nxy.npy')

    plt.close('all')

    for i,n in enumerate(N):

        fig =   plt.figure()
        ax  =   fig.add_subplot(111)
        ax.set_title('n = %s: $\phi(x,y,c/2)$, $f_1(x,y) = (x-a/2)^2 - (y-b/2)^2$, $f_2(x,y) = -f_1(x,y)$' % n)
        ax.set_xlim(0,a)
        ax.set_ylim(0,b)
        cf  =   ax.contourf(Xz,Yz,PHI[i,:,:], 100, cmap=color)
        fig.colorbar(cf,cmap=color)

        fig.savefig('Agenda13/increasing_N/N_%s.png' % n)
        plt.close()

def movie_phi_increasing_z(color=cm.Blues):

    PHI     =   np.load('Agenda13/npy/PHI_zxy.npy')
    Zmax    =   np.max(PHI)
    Zmin    =   np.min(PHI)

    plt.close('all')

    FFMpegWriter    =   manimation.writers['ffmpeg']
    metadata        =   dict(title='$\phi(x,y,z)$, $f_1(x,y) = (x-a/2)^2 - (y-b/2)^2$, $f_2(x,y) = -f_1(x,y)$', artist='Matplotlib')
    writer          =   FFMpegWriter(fps=10, metadata=metadata)

    fig =   plt.figure()
    ax  =   fig.gca()

    with writer.saving(fig, "Agenda13/increasing_z_animation.mp4", 100):

        cf  =   ax.contourf(X,Y,PHI[0,:,:], 100, cmap=color, vmin=Zmin, vmax=Zmax )
        fig.colorbar(cf)

        for i,z in enumerate(Z):
            ax.clear()
            ax.set_title('z = %.2f: $\phi(x,y,z)$, $f_1(x,y) = (x-a/2)^2 - (y-b/2)^2$, $f_2(x,y) = -f_1(x,y)$' % z)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            cf  =   ax.contourf(Xz,Yz,PHI[i,:,:], 100, cmap=color, vmin=Zmin, vmax=Zmax)
            writer.grab_frame()

    plt.close('all')

#===============================================================================
""" run all """
#===============================================================================

def run():
    square_drum()

    Z1  =   SS1()
    Z2  =   SS2(Z1)
    Z3  =   SS3()
    Z4  =   SS4(Z1,Z3)

    for t in np.linspace(0,1000,10):
        plot_diff(t)

    mk_diff_movie()

    calculate_PHI_bisection()
    plot_phi_bisection()

    calculate_PHI_increasing_N()
    plot_phi_increasing_N()

    calculate_PHI_increasing_z()
    movie_phi_increasing_z()
























    return
