#    Copyright (C) 2018 Abdul Razzaq Farooqi, abdul.farooqi[AT]uni-rostock.de
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time

from timeit import default_timer as timer
startime=timer() 
# parameters
delta = 680E-6 # m
Ha = 1E6 # Pa
k = 3E-15 # m2/Pa.s
LambdaS = 8.3E6 # Pa
j0 = 3.8 # A/m2
ke = -2.18E-8 # V/Pa
ki = -2.07E-8 # V/Pa
thickness = delta 

f = np.arange(0.005, 1, 0.001)

def gamma(w):
    return np.sqrt( 1.0j * (2*np.pi*w) / (Ha*k) )
def lambdaC(w):
    gm = gamma(w)
    return Ha * gm * delta * ( np.tanh( (gm*delta)/2) )
def lambdaF(w):
    gm = gamma(w)
    return Ha * gm * delta * ( 1.0 / np.tanh(gm*delta) )
def CurrentfromStress(w):
    lc = lambdaC(w)
    lf = lambdaF(w)
    return (-ki*j0) / (1.0j*2*np.pi*w*delta) * (LambdaS*lc) / (LambdaS+lf)

lq = [np.absolute(CurrentfromStress(w))*1e-3 for w in f]
angleCGS = [np.angle(CurrentfromStress(w), deg=True) for w in f]

#################################################################################################################################
#plt.subplot(2, 1, 1)
plt.semilogx(f, lq, 'k', lw = 1.5, linestyle = '--')
CGS_Mag_Points = [0.0050, 0.0093, 0.0118, 0.0228, 0.0344, 0.0554, 0.0854, 0.1062, 0.2111, 0.3097, 0.5142, 0.8176, 1.0096]
Exp_CGS_Mag = [5.8019, 4.7070, 4.1930, 2.9697, 2.3130, 1.8306, 1.1212, 0.9527, 0.5808, 0.4156, 0.5056, 0.2357, 0.2399]
             # Frank1987_Experimental_Current_Generated_Stress_Magnitude
e = [3.0387, 2.42025, 2.3948, 1.93, 1.5664, 1.1127, 0.74925, 0.61895, 0.3596, 0.3061, 0.74415, 0.3729, 0.3598]
plt.semilogx(CGS_Mag_Points, Exp_CGS_Mag, 'black', marker = 'o', lw = 1.0, markerfacecolor= 'black') 
plt.errorbar(CGS_Mag_Points, Exp_CGS_Mag, fmt = 'ko', yerr = e, elinewidth = 0.9, capsize = 4.0, capthick = 0.8, ecolor = 'black')
###########################
fini = 0.01
w = Constant(2.0*np.pi*fini)

# mesh
mesh = IntervalMesh(400, 0, thickness)
JRe = Constant(j0)
JIm = Constant(j0)

CG1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement(CG1, CG1))
W0 = FunctionSpace(mesh, "CG", 1)

top = CompiledSubDomain("near(x[0], side) && on_boundary", side = 0)
bottom = CompiledSubDomain("near(x[0], side) && on_boundary", side = thickness)

u0Top = Constant(0.0)
u0Bottom = Constant(0.0)

bcs = [DirichletBC(V.sub(0), u0Top, top), DirichletBC(V.sub(0), u0Bottom, bottom)]
(uRe, uIm) = TrialFunctions(V)
(vRe, vIm) = TestFunctions(V)

aRe = Ha*k*inner(grad(uRe), grad(vRe))*dx - Ha*k*inner(grad(uIm), grad(vIm))*dx - w * (uRe * vIm + uIm * vRe)*dx
aIm = Ha*k*inner(grad(uRe), grad(vIm))*dx + Ha*k*inner(grad(uIm), grad(vRe))*dx + w * (uRe * vRe - uIm * vIm)*dx

LRe = ki*inner(JRe, vRe)*dx - ki*inner(JIm, vIm)*dx
LIm = ki*inner(JRe, vIm)*dx + ki*inner(JIm, vRe)*dx

a = aRe + aIm
L = LRe + LIm

A = assemble(a)
b = assemble(L)

for bc in bcs:
    bc.apply(A, b)

u = Function(V)
uRe, uIm = u.split()

problem = LinearVariationalProblem(a, L, u, bcs)
solver = LinearVariationalSolver(problem)
aftersolveT=timer() 
fs3 = np.logspace(-2.3, 0, num = 20)
ys3 = []

for fi in fs3:
    w.assign(2*np.pi*fi)
    solver.solve()
    sigma = project(-Ha*grad(uRe)[0], W0)
    ys3.append(sigma(0)*1E-3)

plt.semilogx(fs3, ys3, lw = 1.0, color = 'k')
###############################################################################

plt.xlabel('Frequency (Hz)')
#plt.title('Current Generated Stress')
plt.ylabel('Current Generated Stress (kPa)')
plt.grid(True, which = 'both', axis = 'both')
plt.legend(['Analytical - Current Work', 'Experimental - Frank1987', 'FEM - Current Work'], loc='upper center', prop={'size': 11})
plt.xlim(0.0045, 1.05)
plt.ylim(-0.3, 8.9)

##################################################################################################################################

print(max(ys3))
V0_dofs = max(V.dofmap().dofs())+1
print("DOFs are : " + str(V0_dofs))
totime = aftersolveT-startime
print("Start time is : " + str(startime))
print("After solve time : " + str(aftersolveT))
print("Total time for Simulation : " + str(totime))
print("No. of meshes : " + str(mesh))
x = mesh.num_cells()
print("No. of mesh cells in x : " + str (x))
y = mesh.ufl_cell()
print("Shape of mesh : " + str (y))
