import sympy as sy
import numpy as np
import pdb
from sympy import besselj,bessely,besseli,besselk
import scipy.special as ss
import matplotlib.pyplot as plt
from library.misc import directory_checker, find_lowest
from library.plotting import set_mpl_params
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani
import matplotlib.animation as manimation
from matplotlib import cm
from sympy.solvers import solve
from sympy import init_printing
import os
import math

init_printing()
set_mpl_params()
parent  =   "T13_BesselFunctions/"
directory_checker(parent)
pr  =   sy.pprint

#===============================================================================
""" Bessel Series Representation """
#===============================================================================

def Jm_pos(m,x,eps=1e-15):
    """ BF: Eq 29 """

    def J_n(n):
        return (-1)**n * (x/2)**(2*n + m) / ( sy.factorial(n) * sy.gamma(n+m+1) )

    n   =   1
    Ji  =   J_n(0)
    Jf  =   Ji + J_n(n)

    while abs(Jf - Ji) > eps:
        n   +=  1
        Ji  =   Jf
        Jf  =   Ji + J_n(n)

    return Jf

def Jm_neg(m,x,eps=1e-15):
    """ BF: Eq 30 """

    def J_n(n):
        return (-1)**n * (x/2)**(2*n - m) / ( sy.factorial(n) * sy.gamma(n-m+1) )

    n   =   1
    Ji  =   J_n(0)
    Jf  =   Ji + J_n(n)

    while abs(Jf - Ji) > eps:
        n   += 1
        Ji  =   Jf
        Jf  =   Ji + J_n(n)

    return Jf

def Jm_series(m,x,eps=1e-15):

    # if m >= 0:
    #     return Jm_pos(m,x,eps)
    # else:
    #     return Jm_neg(abs(m),x,eps)

    J   =   Jm_pos(abs(m),x,eps=eps)
    assert divmod(m,1)[1] == 0, "m is integer"
    if m >= 0:
        return J
    else:
        return (-1)**abs(m) * J

def Nm_series(m,x,eps=1e-10):
    """ BF: Eq 33 """
    m       =   abs(m)
    Jp      =   Jm_series(m,x,eps=eps)

    if divmod(m,1)[1] == 0:
        Jn      =   (-1)**m * Jp
        alpha   =   (m+eps)*sy.pi
    else:
        Jn      =   Jm_series(-m,x,eps=eps)
        alpha   =   m*sy.pi

    return -(sy.cos(alpha) * Jp - Jn) / sy.sin(alpha)

#===============================================================================
""" Bessel Integral Representation """
#===============================================================================

def Jm_integral(m,x):
    def f(theta):
        return sy.cos( x * sy.sin(theta) - m*theta )

    integral    =   mpmath.quad(f,[0,2*np.pi])
    return integral/(2*sy.pi).evalf()

#===============================================================================
""" How to calculate bessel functions the quickest? We have:
1) Jm_series(m,x)
2) Jm_integral(m,x)
3) mpmath.besselj(m,x)
4) scipy.special.jv(m,x)

try using "timeit" in the commandline to see which one is the fastest

sources:
http://docs.sympy.org/0.7.1/modules/mpmath/functions/bessel.html
https://docs.scipy.org/doc/scipy/reference/special.html
"""

#===============================================================================
""" Figures 1 & 2 """
#===============================================================================

def plot_fig1():
    plt.close('all')

    X   =   np.linspace(0,10,100)
    fig =   plt.figure()

    ax1 =   fig.add_subplot(121)
    ax1.set_xlabel('X')
    ax1.set_ylabel('J$_m$(x)')
    for m in range(5):
        ax1.plot(X,ss.jv(m,X), label="m = %s" % m)
    ax1.legend()
    ax1.set_xlim(0,np.max(X))

    ax2 =   fig.add_subplot(122)
    ax2.set_xlabel('X')
    ax2.set_ylabel('N$_m$(x)')
    for m in range(5):
        ax2.plot(X,ss.yv(m,X), label="m = %s" % m)
    ax2.legend()
    ax2.set_xlim([0,np.max(X)])
    ax2.set_ylim([-1,.7])

    plt.tight_layout()

    fig.savefig(parent + "Jm_Ym.png")
    plt.close()
    print("plot saved to /T13_BesselFunctions/Jm_Ym.png")

def plot_fig2():
    plt.close('all')

    X   =   np.linspace(0,5,100)
    fig =   plt.figure()

    ax1 =   fig.add_subplot(121)
    ax1.set_xlabel('X')
    ax1.set_ylabel('I$_m$(x)')
    for m in range(5):
        ax1.plot(X,ss.iv(m,X), label="m = %s" % m)
    ax1.legend()
    ax1.set_xlim(0,np.max(X))
    ax1.set_ylim([0,10])

    ax2 =   fig.add_subplot(122)
    ax2.set_xlabel('X')
    ax2.set_ylabel('K$_m$(x)')
    for m in range(5):
        ax2.plot(X,ss.kv(m,X), label="m = %s" % m)
    ax2.legend()
    ax2.set_xlim([0,np.max(X)])
    ax2.set_ylim([0,10])

    plt.tight_layout()

    fig.savefig(parent + "Im_Km.png")
    plt.close()
    print("plot saved to /T13_BesselFunctions/Im_Km.png")

#===============================================================================
""" Problem 2 """
# geometry constants
a           =   1
v           =   1
nmax        =   10
N_lowest    =   10
d           =   .1

# grid
R       =   np.linspace(0,a,100)
PHI     =   np.linspace(0,2*np.pi,100)
R,PHI   =   np.meshgrid(R,PHI)

# convert grid to CC
X,Y     =   R*np.cos(PHI),R*np.sin(PHI)
#===============================================================================

def zeta_Matrix():
    zeta    =   np.zeros( (nmax,nmax) )
    for i in range(nmax):
        zeta[i,:]   =   ss.jn_zeros(i,nmax)
    return zeta
zeta    =   zeta_Matrix()
omega   =   (v/a) * zeta

M,N,Omega   =   find_lowest(omega,N_lowest)

def Z_t(Amn,m,n,t):
    w   =   omega[m,n]
    Z   =   Amn * ss.jv(m,w*R/v) * np.cos(m*PHI) * np.cos(w*t + m*np.pi)
    return np.array( Z , dtype=np.float)

# exercise 51
def exercise_51():
    print("\nExercise 51")
    for i in range(N_lowest):
        print("omega[%s,%s] = %s" % (M[i],N[i],Omega[i]) )

def exercise_52(color=cm.seismic):
    print("\nExercise 52")

    home    =   parent + 'exercise_52/'
    directory_checker(parent)
    directory_checker(home)

    set_mpl_params()

    def plot_mn(i):

        m   =   M[i]
        n   =   N[i]
        w   =   Omega[i]
        Z   =   Z_t(1,0,i,0)

        fig     =   plt.figure()
        ax      =   plt.gca(projection='3d')
        ax.set_title('$\omega_{%s%s} = %.2f$' % (m,n,w))
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([0])
        ax.set_zlim([-1,1])
        surf    =   ax.plot_surface(X,Y,Z, cmap=color)
        # fig.colorbar(surf, cmap=color, shrink=.8)

        name    =   home + 'm_%s_n_%s.png' % (m,n)
        fig.savefig(name)
        plt.close()
        print("saved figure to %s" % name)

    for i in range(6):
        plot_mn(i)

def problem_2(modes=5,fpm=50,color=cm.seismic):
    """
    args
    ----
    modes:  The number of modes that will be in the movies
    fpm:    frames per mode
    color:  the color scheme for the images and movies
    """

    home    =   parent + 'Problem2/'
    directory_checker(home)

    def solve_boundary():

        """ set up expression """
        # indecies
        m,n =   sy.symbols('m n',integer=True)
        # constants
        phim,a,d,v  =   sy.symbols('phi_m a d v', real=True)
        # variables
        rho,phi,t,omega,x   =   sy.symbols('rho phi t omega x',real=True,positive=True)
        # Functions
        # A   =   sy.Function('A')(m,n)
        A   =   sy.symbols('A',real=True)
        J   =   besselj(m,omega*rho/v)
        Z   =   A * J * sy.cos(m*phi) * sy.cos(omega*t + phim)

        """ solve first initial condition Zdot(rho,phi,t=0) = 0 """
        # find the left hand side - time derivative of Z
        lhs =   sy.diff( Z , t )
        # set t = 0
        lhs =   lhs.subs(t,0)
        # set Z1 = 0
        eq1 =   sy.Eq( lhs , 0 )
        # all factors are save except for phi_m; solve for phi_m
        phim1   =   m*sy.pi # sympy.solve can be used, but it returns the trival case
        # substitute new value of phim into original Z
        Z   =   Z.subs( phim , phim1 )

        """ solve the implied condition that d/drho Z(rho=a,phi,t) = 0 """
        # set left hand side to d/drho Z
        lhs =   sy.diff( Z , rho )
        # set rho = a
        lhs =   lhs.subs( rho , a )
        """ J' has to vanish, which means that the two terms shown in the output
        must equal 0 which implies m = 0 """
        # set m = 0
        Z   =   Z.subs( m , 0 )

        """ plug in condition Z(rho,phi,t=0) = d(1-rho/a) """
        lhs =   Z.subs( t , 0 )
        rhs =   d * (1 - rho/a)
        """ we need to get rid of rho dependance, so exploit orthoganality of Jm(x)
        http://www.math.usm.edu/lambers/mat415/lecture15.pdf
        """
        omega1  =   sy.symbols('omega_1',positive=True, real=True)
        lhs     *=  rho * besselj(0,omega1*rho/v)
        rhs     *=  rho * besselj(0,omega1*rho/v)
        # integrate both sides WRT rho
        lhs     =   sy.integrate( lhs , (rho,0,a) )
        rhs     =   sy.integrate( rhs , (rho,0,a) )
        """ looking at the output, we know that omega = omega1
        and that the first term will vanish when the root is plugged in."""
        eq      =   sy.Eq( lhs , rhs )
        eq      =   eq.subs( omega1 , omega )
        # replace a omega / v with zeta_0 (the nth root of J0)
        zeta0   =   sy.symbols('zeta_0', positive=True)
        eq      =   eq.subs( a * omega / v , zeta0)
        A1      =   solve( eq , A )[0]
        An      =   sy.lambdify( (a,d,v,omega,zeta0) , A1 , 'sympy')
        # pdb.set_trace()
        return An

    def mk_array_single():
        print("\nstarting Z_ntxy creation...")
        # calculate coefficient function
        An  =   solve_boundary()
        def A_n(n):
            return An(a,d,v,omega[0,n],zeta[0,n]).evalf()

        # make empty drum head height array w/ indecies (mode, time, x, y)
        Z_ntxy  =   np.zeros( ( modes , fpm , 100 , 100 ) )
        T       =   np.zeros( ( modes , fpm ) )

        for n in range(modes):

            print("\nmode: %s/%s..." % (n+1,modes) )
            omega_n =   omega[0,n]
            T_n     =   np.linspace(0 , 2*np.pi/omega_n , fpm)
            T[n,:]  =   T_n
            A       =   A_n(n)

            for ti,time in enumerate(T_n):
                print("time index: %s/%s... " % (ti+1,fpm) )
                Z_ntxy[n,ti,:,:]    =   Z_t(A,0,n,time)

        print("\nsaving arrays to %s" % home)
        np.save(home + 'Z_ntxy.npy', Z_ntxy)
        np.save(home + 'T_ntxy.npy', T)

    def mk_array_multiple():
        print("\nstarting Z_txy creation...")
        # calculate coefficient function
        An  =   solve_boundary()
        def A_n(n):
            return An(a,d,v,omega[0,n],zeta[0,n]).evalf()

        omega0  =   omega[0,:modes]
        T0      =   2*np.pi / omega0
        Tmin    =   np.min(T0)
        Tmax    =   np.max(T0)
        factor  =   math.ceil( Tmax/Tmin)
        print("factor: %s" % factor)
        # pdb.set_trace()

        T       =   np.linspace(0, factor*Tmin, factor*fpm)
        Z_txy   =   np.zeros( (factor*fpm,100,100) )

        for it,time in enumerate(T):
            print("\ntime index: %s/%s..." % (it+1,factor*fpm) )
            for n in range(modes):
                print("mode: %s/%s..." % (n+1,modes) )
                A   =   A_n(n)
                Z_txy[it,:,:]   +=  Z_t(A,0,n,time)

        print("\nsaving arrays to %s" % home)
        np.save(home + "Z_txy.npy", Z_txy)
        np.save(home + "T_txy.npy", T)

    def plot_single_modes():

        home1   =   home + 'single_modes/'
        directory_checker(home1)
        if os.path.isfile(home + 'Z_ntxy.npy') == False:
            mk_array_single()
        Z_ntxy  =   np.load(home + 'Z_ntxy.npy')

        def plot_single(n):

            Z0n     =   Z_ntxy[n,0,:,:]
            Zmax    =   np.max(Z0n)
            w       =   omega[0,n]

            fig =   plt.figure()
            ax  =   plt.gca(projection='3d')
            ax.set_title('$\omega_{0,%s} = %.2f$' % (n,w))
            ax.set_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([0])
            surf    =   ax.plot_surface(X,Y,Z0n, cmap=color, vmin=-Zmax, vmax=Zmax)
            fig.colorbar(surf, cmap=color, shrink=.8)

            fig.savefig(home1 + 'm_0_n_%s.png' % (n))
            plt.close()

        for n in range(modes): plot_single(n)

    def mk_movie_single_modes():

        print("\nstarting single modes...")
        if os.path.isfile(home + 'Z_ntxy.npy') == False: mk_array_single()
        Z_ntxy  =   np.load(home + 'Z_ntxy.npy') # [n,t,x,y]
        T       =   np.load(home + 'T_ntxy.npy') # [n,t]
        print("loaded arrays")

        plt.close('all')
        FFMpegWriter    =   manimation.writers['ffmpeg']
        metadata        =   dict(title='Problem 2: Circular Drum', artist='Matplotlib')
        writer          =   FFMpegWriter(fps=40, metadata=metadata)

        fig             =   plt.figure()
        ax              =   fig.gca(projection='3d')

        with writer.saving(fig, home+"single_modes_animation.mp4", 100):

            for n in range(modes):
                print("\nmode: %s/%s..." % (n+1,modes) )

                fig.clear()
                ax      =   fig.gca(projection='3d')

                Zn0     =   Z_ntxy[n,0,:,:]
                Zmax    =   np.max(Zn0)

                surf            =   ax.plot_surface(X,Y,Zn0, cmap=color, vmin=-Zmax, vmax=Zmax)
                fig.colorbar(surf, shrink=0.5)

                for it,time in enumerate(T[n,:]):
                    print("time index: %s/%s..." % (it+1,fpm))
                    ax.clear()
                    ax.set_title('n:%s , t:%.2f' % (n,time) )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([0])
                    ax.set_zlim(-Zmax,Zmax)
                    ax.plot_surface(X,Y,Z_ntxy[n,it,:,:], cmap=color, vmin=-Zmax, vmax=Zmax)
                    writer.grab_frame()
        plt.close()
        print("finished")

    def mk_movie_multiple_modes():

        print("\nstarted multiple modes...")
        if os.path.isfile(home + 'Z_txy.npy') == False: mk_array_multiple()
        Z_txy   =   np.load(home + 'Z_txy.npy')
        T       =   np.load(home + 'T_txy.npy')
        print("loaded arrays")

        plt.close('all')
        FFMpegWriter    =   manimation.writers['ffmpeg']
        metadata        =   dict(title='Problem 2: Circular Drum', artist='Matplotlib')
        writer          =   FFMpegWriter(fps=40, metadata=metadata)

        Z0      =   Z_txy[0,:,:]
        Zmax    =   np.max(Z0)

        fig     =   plt.figure()
        ax      =   fig.gca(projection='3d')
        surf    =   ax.plot_surface(X,Y,Z0, cmap=color, vmin=-Zmax, vmax = Zmax)
        fig.colorbar(surf, shrink=.8)

        with writer.saving(fig, home+"multiple_modes_animation.mp4", 100):

            for it,time in enumerate(T):
                print("time index: %s/%s" % (it+1,len(T)) )
                ax.clear()
                ax.set_title('t:%.2f' % time)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([0])
                ax.set_zlim(-Zmax,Zmax)
                ax.plot_surface(X,Y,Z_txy[it,:,:], cmap=color, vmin=-Zmax, vmax=Zmax)
                writer.grab_frame()
        plt.close()
        print("finished...")

    # plot_single_modes()
    # mk_movie_single_modes()
    mk_movie_multiple_modes()

#===============================================================================
""" my own problem """
# initial condition constants
constsants  =   {'a':1, 'rho0':1/2, 'phi0':np.pi/4, 'sigma':1/3, 'epsilon':1/5}
#===============================================================================

# def my_problem():
#
#     home    =   parent + 'My_Problem/'
#     directory_checker(parent)
#     directory_checker(home)
#
#     """ conditions """
#     # 1) Z(rho=a,phi,t)        = 0
#     # 2) dZ/drho (rho=a,phi,t) = 0
#     # 3) Zdot(rho,phi,t=0)     = 0
#     # 4) Z(rho,phi,t=0)        = -A0 Jm( omega_00 (rho-rho0) / v ) cos( m (phi-phi0) )
#
#     def solve_boundary():
#
#         """ set up expression """
#         # indecies
#         m,n =   sy.symbols('m n',integer=True,positive=True)
#         # constants
#         phim,v,a,rho0,phi0,sigma,epsilon    =   sy.symbols('phi_m v a rho_0 phi_0 sigma epsilon', real=True,positive=True)
#         # variables
#         rho,phi,t,omega1,omega2 =   sy.symbols('rho phi t omega omega_0',real=True,positive=True)
#         # Functions
#         A   =   sy.symbols('A',real=True)
#         J   =   besselj(m,omega1*rho/v)
#         Z   =   A * J * sy.cos(m*phi) * sy.cos(omega1*t + phim)
#         Z0  =   epsilon * besselj(2,omega2*rho/v) * sy.cos(2*phi)
#
#         """ plug in condition 3 """
#         lhs     =   Z.diff(t)
#         lhs     =   lhs.subs(t,0)
#         eq      =   sy.Eq( lhs , 0 )
#         phim1   =   solve(eq,phim)
#         Z       =   Z.subs(phim, m*sy.pi)
#
#         """ plug in condition 4 """
#         lhs     =   Z.subs(t,0)
#         rhs     =   Z0
#         # do othogonal cosine integral
#         m1      =   sy.symbols('m_1', integer=True, positive=True)
#         f       =   sy.cos(m1*phi)
#         limits  =   (phi,0,2*sy.pi)
#         lhs     =   sy.integrate(lhs*f,limits)
#         rhs     =   sy.integrate(rhs*f,limits)
#         lhs     =   lhs.subs(m1,m)
#         lhs     =   lhs.subs(m,2)
#         rhs     =   rhs.subs(m1,2)
#         # do an orthoganal bessel function integral
#         omega2  =   sy.symbols('omega_2',integer=True,positive=True)
#         f       =   rho * besselj(2,omega2*rho/v)
#         limits  =   (rho,0,a)
#         lhs     =   sy.integrate(lhs*f,limits)
#         rhs     =   sy.integrate(rhs*f,limits)
#
#     solve_boundary()


#===============================================================================
""" run module """
#===============================================================================

def run():
    # plot_fig1()
    # plot_fig2()
    # exercise_51()
    # exercise_52()
    # problem_2()
    my_problem()

def Z_test(m,n):
    # Z_t(Amn,m,n,t)
    return Z_t(.1,m,n,0)
