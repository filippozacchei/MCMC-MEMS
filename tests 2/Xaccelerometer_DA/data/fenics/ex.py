import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.geometry import BoundingBoxTree, compute_collisions_points
from mpi4py import MPI
import ufl
import numpy as np

L       = 100.              # Length
E       = 160e9            # Young modulus
nu      = 0.22             # Poisson ratio
W       = 1              # Width
mu      = E/(2*(1+nu))     # Lame coefficient
rho     = 2320.            # Material density
delta   = W / L           
g       = 9.81             # Gravity acceleratiom
beta    = 1.25
lambda_ = E*nu/((1-2*nu)*(1+nu))             # Lame coefficient
F       = 1e6

J = (1./12.)*(W*(W**3))
k = 3*E*J/(L**3)
dx = F/k
print(dx)

domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, W]], [40, 5, 5], mesh.CellType.hexahedron)
V = fem.VectorFunctionSpace(domain, ("Lagrange", 2))

def left(x):
    return np.isclose(x[0], 0)
    
def right(x):
    return np.isclose(x[0], L)

fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

# Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two
marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

u_D = np.array([0, 0, 0], dtype=default_scalar_type)
left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_D, left_dofs, V)]

T = fem.Constant(domain, default_scalar_type((0, 0, F)))
f = fem.Constant(domain, default_scalar_type((0, 0, 0)))

metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

u = fem.Function(V)
v = ufl.TestFunction(V)

# Deformation gradient
def F(u):
    return ufl.Identity(len(u)) + ufl.grad(u)

def J(u):
    return ufl.det(F(u))

def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

def S(u):
    return 2.0 * mu * epsilon(u) + lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u))

def P(u):
    return ufl.dot(F(u),S(u))


F = ufl.inner(S(u), ufl.grad(v)) * dx - ufl.inner(v, f) * dx - ufl.inner(v, T) * ds(2)

problem = NonlinearProblem(F, u, bcs)


solver = NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1e-12
solver.rtol = 1e-12
solver.convergence_criterion = "incremental"

num_its, converged = solver.solve(u)

pyvista.start_xvfb()

# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = u.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color=None)
warped = grid.warp_by_vector("u", factor=1e0)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot("deflection.png")