import casadi as ca

# import sys
# sys.path.append('/usr/local/lib')
# sys.path.append('/usr/lib')
# # for path in sys.path:
#     print(path)


# Symbols/expressions
x = ca.SX.sym('x')
y = ca.SX.sym('y')
z = ca.SX.sym('z')
f = x**2+100*z**2
g = z+(1-x)**2-y

nlp = {}                 # NLP declaration
nlp['x']= ca.vertcat(x,y,z) # decision vars
nlp['f'] = f             # objective
nlp['g'] = g             # constraints

opts = {'ipopt.linear_solver': 'ma57'}

# Create solver instance
F = ca.nlpsol('F','ipopt',nlp,opts);

# Solve the problem using a guess
F(x0=[2.5,3.0,0.75],ubg=0,lbg=0)