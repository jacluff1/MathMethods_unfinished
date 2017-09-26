import sympy as sy
from sympy.solvers import solve

#===============================================================================
""" auxillary functions """
#===============================================================================

def pde_solver(pde,f,var,c1,c2):
    gf  =   sy.dsolve(pde,f)
    f   =   gf.rhs
    f   =   f.subs(C1,c1)
    f   =   f.subs(C2,c2)
    return f

def diffusion_constant(sigma,rho,C):
    return sigma / (rho*C)

def FO_boundary_solver(func,var,var_pos,const_var,icon_value):
    f1          =   func.subs(var,var_pos)
    eq1         =   sy.Eq( icon_value , f1 )
    const_val   =   solve(eq1,const_var)[0]
    return func.subs(icon_value,const_var)

def SO_boundary_solver(func,var,var_pos1,var_pos2,const_var1,const_var2,icon_value1,icon_value2):
    f1          =   func.subs(var,var_pos1)
    eq1         =   sy.Eq( icon_value1 , f1 )
    return eq1

#===============================================================================
""" symbolic variables and functions """
#===============================================================================

#variables
x,y,t           =   sy.symbols('x y t', positive=True, real=True)

# constants
mu,eta,lamda    =   sy.symbols('mu eta lambda',real=True)
C1,C2           =   sy.symbols('C1 C2')
A,B,C,D,E,F     =   sy.symbols('A B C D E F')

#===============================================================================
""" numerical constants """
#===============================================================================

# use eta_teflon
sig_tef =   .25     # W m^-1 K^-1
rho_tef =   2200    # kg m^-3
C_tef   =   970     # J kg^-1 K^-1
eta_tef =   diffusion_constant(sig_tef,rho_tef,C_tef)

#===============================================================================
""" 2D diffusion """
#===============================================================================

def Temp_diffusion(eta_num):

    # set up default kwarg arguments
    T0 = 200
    X0 = Xa = Yb = 0
    Y0 = 350
    # if kwargs.has_key('T0'): T0 = kwargs['T0']
    # if kwargs.has_key('X0'): X0 = kwargs['X0']
    # if kwargs.has_key('Xa'): X0 = kwargs['Xa']
    # if kwargs.has_key('Y0'): X0 = kwargs['Y0']
    # if kwargs.has_key('Yb'): X0 = kwargs['Yb']

    # functions
    X       =   sy.Function('X')(x)
    Y       =   sy.Function('Y')(y)
    T       =   sy.Function('T')(t)

    # sode
    dex     =   X.diff(x,2) + (mu-lamda)*X
    dey     =   Y.diff(y,2) + lamda*Y
    det     =   T.diff(t) + mu*eta*T

    # solve de(s)
    X       =   pde_solver(dex,X,x,A,B)
    Y       =   pde_solver(dey,Y,y,C,D)
    T       =   pde_solver(det,T,t,E,F)
    Phi     =   X*Y*T

    # solve initial/boundary conditions
    # substitute numerical value of eta into T(t)
    Tn      =   T.subs(eta,eta_num)
    # Tn      =   FO_boundary_solver(T,t,0,E,T0)


    return X,Y,Tn




#===============================================================================
