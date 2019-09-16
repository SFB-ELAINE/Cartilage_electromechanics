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

from timeit import default_timer as timer
startime=timer()
# parameters
delta = 680*1E-6 # m
Ha = 1E6 # Pa
k = 3.0E-15 # m2/Pa.s
LambdaS = 8.3E6 # Pa
u0 = 10E-6 # m
ke = -2.18E-8 # V/Pa
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
def StreamingPotential(w):
    lc = lambdaC(w)
    lf = lambdaF(w)
    return ke * (LambdaS * lc / (LambdaS+lf)) * u0/delta

ls = [np.absolute(StreamingPotential(w))*1e3 for w in f]

angleSP = [np.angle(StreamingPotential(w), deg=True) for w in f]
######################################################################################################################
#plt.subplot(2, 1, 1)
plt.semilogx(f, ls, 'black', lw = 1.5, linestyle = '--')
SP_Mag_Points = [0.0050, 0.0078, 0.0098, 0.0196, 0.0296,0.0488, 0.0768, 0.0977, 0.1935, 0.2882, 0.4775, 0.7658, 0.9991]
Exp_SP_Mag = [0.3825, 0.5531, 0.6694, 0.8613, 0.8749, 1.1365, 1.2039, 1.3739, 1.5700, 1.5711, 1.8576, 1.8423, 1.9549]
             # Frank1987_Experimental_Streaming_Potential_Magnitude
e = [0.2338, 0.3184 , 0.3522, 0.50855, 0.57545, 0.69295, 0.7570,  0.7963, 0.89025, 0.926, 1.0028, 1.05845, 1.0856]
plt.semilogx(SP_Mag_Points, Exp_SP_Mag, 'black', marker = 'o', lw = 1.0) 
plt.errorbar(SP_Mag_Points, Exp_SP_Mag, fmt = 'ko', yerr = e, elinewidth = 0.9, capsize = 4.0, capthick = 0.8, \
             ecolor = 'black')
#############################
fini = 0.005
w = Constant(2.0*np.pi*fini)

# mesh
mesh = IntervalMesh(460, 0, thickness)
fRe = Constant(0.0)
fIm = Constant(0.0)

CG1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement(CG1, CG1))
W0 = FunctionSpace(mesh, "CG", 1)

top = CompiledSubDomain("near(x[0], side) && on_boundary", side = 0)
bottom = CompiledSubDomain("near(x[0], side) && on_boundary", side = thickness)

u0Top = Constant(u0)
u0Bottom = Constant(0.0)

bcs = [DirichletBC(V.sub(0), u0Top, top), DirichletBC(V.sub(0), u0Bottom, bottom)]

(uRe, uIm) = TrialFunctions(V)
(vRe, vIm) = TestFunctions(V)

aRe = Ha*k*inner(grad(uRe), grad(vRe))*dx - Ha*k*inner(grad(uIm), grad(vIm))*dx - w * (uRe * vIm + uIm * vRe)*dx
aIm = Ha*k*inner(grad(uRe), grad(vIm))*dx + Ha*k*inner(grad(uIm), grad(vRe))*dx + w * (uRe * vRe - uIm * vIm)*dx

LRe = inner(fRe, vRe)*dx - inner(fIm, vIm)*dx
LIm = inner(fRe, vIm)*dx + inner(fIm, vRe)*dx

a = aRe + aIm
L = LRe + LIm
A = assemble(a)
b = assemble(L)

for bc in bcs:
    bc.apply(A, b)

u = Function(V)
uRe, uIm = u.split()

problem = LinearVariationalProblem(a, L, u, bcs)

aftersolveT=timer()
solver = LinearVariationalSolver(problem)

fs2 = np.logspace(-2.3, 0, num = 20)
ys2 = []

print(fs2)

for fi in fs2:
    w.assign(2*np.pi*fi)
    solver.solve()
    sigma = -Ha*grad(uRe)[0]
    LmbdaOC = sigma / (uRe(0)/delta)
    sigmaOC = project((LambdaS*LmbdaOC/(LambdaS+LmbdaOC))*(u0/delta), W0)
    potential = ke*(sigmaOC(delta)-sigmaOC(0))*1E3 # change to mV
    ys2.append(potential)

plt.semilogx(fs2, ys2, lw = 1.0, color = 'k', linestyle = '-')
###############################################################################
plt.xlabel('Frequency (Hz)')
#plt.title('Streaming Potential')
plt.ylabel('Streaming Potential (mV)')
plt.grid(True, which = 'both', axis = 'both')
plt.legend(['Analytical - Current Work', 'Experimental - Frank1987', 'FEM - Current Work'], loc='upper left', \
           prop={'size': 11})
plt.xlim(0.0045, 1.05)
plt.ylim(-0.1, 3.1)
##############################################################################################################
print(max(fs2))
print(max(ys2))
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
