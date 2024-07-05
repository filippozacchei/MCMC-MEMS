from dolfinx import mesh, fem, plot
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc

# Geometry and mesh
L, W, gap = 1.0, 0.1, 0.1  # Length, width of beams and gap
nx, ny = 50, 10
mesh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, W + gap + W])], [nx, 2 * ny], cell_type=mesh.CellType.triangle)

# Function spaces
V = fem.FunctionSpace(mesh, ("CG", 1))  # Space for potential
U = fem.VectorFunctionSpace(mesh, ("CG", 1))  # Space for displacement

# Boundary conditions
def top_beam(x):
    return np.isclose(x[1], W + gap + W)

def bottom_beam(x):
    return np.isclose(x[1], 0)

bc_potential_top = fem.dirichletbc(value=fem.Constant(mesh, 10), dofs=fem.locate_dofs_geometrical(V, top_beam), V=V)
bc_potential_bottom = fem.dirichletbc(value=fem.Constant(mesh, 0), dofs=fem.locate_dofs_geometrical(V, bottom_beam), V=V)
bcs_potential = [bc_potential_top, bc_potential_bottom]

# Electrostatic problem
phi = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
L = fem.Constant(mesh, 0) * v * ufl.dx
problem = fem.petsc.LinearProblem(a, L, bcs=bcs_potential)
phi_solution = problem.solve()

# Compute electrostatic forces
epsilon_0 = 8.854e-12  # Permittivity of free space
force_density = -0.5 * epsilon_0 * ufl.grad(phi_solution)**2
f = fem.Function(U)
f.interpolate(lambda x: force_density)

# Mechanical problem (linear elasticity for simplicity)
u = ufl.TrialFunction(U)
d = ufl.TestFunction(U)
E, nu = 210e9, 0.3  # Young's modulus and Poisson's ratio for steel
mu = E / (2 * (1 + nu))
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
sigma = lambda_ * ufl.div(u) * ufl.Identity(len(u)) + 2 * mu * ufl.sym(ufl.grad(u))
a_mech = ufl.inner(sigma, ufl.grad(d)) * ufl.dx
L_mech = ufl.dot(f, d) * ufl.dx
bcs_mech = [fem.dirichletbc(value=fem.Constant(mesh, (0, 0)), dofs=fem.locate_dofs_geometrical(U, bottom_beam), V=U)]
problem_mech = fem.petsc.LinearProblem(a_mech, L_mech, bcs=bcs_mech)
u_solution = problem_mech.solve()

# Post-processing and visualization
with fem.XDMFFile(MPI.COMM_WORLD, "potential_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(phi_solution)

with fem.XDMFFile(MPI.COMM_WORLD, "displacement_solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function
