import sympy as sy
from sympy.solvers import solve

C1,C2           =   sy.symbols('C1 C2')
A,B,C,D,E,F,G   =   sy.symbols('A B C D E F G')

def iterate_symbols(eq):
    sym =   eq.free_symbols
    sy.pprint(sym)
    sy.pprint(eq)
    solve_for   =   "n"
    while solve_for == "n":
        for s in sym:
            solve_for   =   input("solve for %s? (y/n)" % s)
            if solve_for == "y": return s
        again   =   input("try again? (y/n)")
        if again == "n":
            return None

def dirchlet_condition(function,variable,position,cond_value):
    """ solve a boundary condition where f(x) is given at the boundary """
    f           =   function.subs(variable,position)
    eq          =   sy.Eq( f , cond_value )
    s           =   iterate_symbols(eq)
    s1          =   solve(eq,s)[0]
    f1          =   function.subs(s,s1)
    sy.pprint(f1)
    return f1

def neumann_condition(function,variable,position,cond_value):
    """ solve a boundary condition where f'(x) is given at the boundary """
    df          =   sy.diff(function,variable)
    df          =   df.subs(variable,position)
    eq          =   sy.Eq( df , cond_value )
    s           =   iterate_symbols(eq)
    s1          =   solve(eq,s)[0]
    f1          =   function.subs(s,s1)
    pprint(f1)
    return f1

class condition:
    def __init__ (self,position,value,cond_type):
        """
        position:   variable value at condition
        value:      condition value
        type:       condition type (initial, dirchlet, neumann, cauchy, periodic)
        """
        self.pos   =   position
        self.val   =   value
        self.type  =   cond_type

class differential_equation:
    def __init__ (self,diff_equation,function,variable,constants,conditions):
        """
        diff_equation:  differential equation
        function:       function symbol
        variable:       differential variable
        constants:      linear constants - list
        conditions:     conditions to apply - list
        """
        self.de     =   diff_equation
        self.f      =   function
        self.var    =   variable
        self.cons   =   constants
        self.conds  =   conditions

        fx          =   sy.dsolve(self.de,self.f).rhs
        fx          =   fx.subs(C1,self.cons[0])
        fx          =   fx.subs(C2,self.cons[1])
        self.fx     =   fx

        # if self.conds != None:
        #     for c in self.conds:
        #         if c.type == 'dirchlet':
        #             self.fx     =   dirchlet_condition(self.fx,self.var,c.pos,c.val)
        #         elif c.type == 'neumann':
        #             self.fx     =   neumann_condition(self.fx,self.var,c.pos,c.val)
                # make other conditions later

class solution:
    def __init__ (self,diff_equations,conditions):
        """
        diff_equations: list of differential equation class objects
        conditions:     list of conditions to apply
        """
        self.des    =   diff_equations
        self.conds  =   conditions

        gs          =   1
        # plug in boundary conditions
        for de in self.des:
            if de.conds != None:
                print("\n")
                sy.pprint(de.fx)
                for c in de.conds:
                    if c.type == 'dirchlet':
                        de.fx     =   dirchlet_condition(de.fx,de.var,c.pos,c.val)
                    elif c.type == 'neumann':
                        de.fx     =   neumann_condition(de.fx,de.var,c.pos,c.val)
                    # make other conditions later
            gs  *=  de.fx

        # general solution
        self.gs     =   gs

#===============================================================================
""" tests """
#===============================================================================

x,y,t           =   sy.symbols('x y t', real=True,positive=True)
eta,mu,lamda    =   sy.symbols('eta mu lamda',real=True)
a,b             =   sy.symbols('a b',real=True,positive=True)
T0,T1           =   sy.symbols('T_0 T_1',real=True,positive=True)

X   =   sy.Function('X')(x)
Y   =   sy.Function('Y')(y)
T   =   sy.Function('t')(t)
Z   =   X*Y*T

# write separated differential equations
dx  =   X.diff(x,2) + (mu - lamda)*X
dy  =   Y.diff(y,2) + lamda*Y
dt  =   T.diff(t) + mu*eta*T

# write boundary conditions
x0      =   condition(0,0,'neumann')
xa      =   condition(a,0,'neumann')
yb      =   condition(b,0,'neumann')
y0      =   condition(0,T1,'dirchlet')
Phi0_xy =   condition(0,T0,'initial')

dex =   differential_equation(dx,X,x,[A,B],[x0,xa])
dey =   differential_equation(dy,Y,y,[C,D],[y0,yb])
det =   differential_equation(dt,T,t,[E,F],None)

def test():

    gs  =   solution([dex,dey,det],[Phi0_xy])
    return gs
