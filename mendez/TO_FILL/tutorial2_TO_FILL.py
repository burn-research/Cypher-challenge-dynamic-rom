# %% [markdown]
# # Tutorial 2: one PDE, three discretisations  (version to fill in)
#
# We solve the forced viscous Burgers equation on $\Omega=(0,1)$
#
# $$\partial_t u + u\,\partial_x u = \nu\,\partial_{xx}u + \alpha f(x),
#   \qquad u(0,t)=1,\quad u(1,t)=0,\quad u(x,0)=1-x$$
#
# with a localised Gaussian source
# $f(x)=\exp[-(x-x_f)^2/(2\sigma_f^2)]$, $x_f=0.65$, $\sigma_f=0.08$,
# and parameters $\mu=(\nu,\alpha)$.
#
# Three methods:
#
# | part | method | unknowns |
# |---|---|---|
# | A | finite differences | point values $u_j$ |
# | B | spectral Galerkin, sine basis | coefficients $a_n(t)$ |
# | C | linear finite elements | nodal coefficients $a_n(t)$ |
#
# Parts B and C use the **same assembly code**. Only the basis differs.
# This notebook is self-contained: nothing to install beyond numpy,
# scipy and matplotlib.
#
# **There are seven TODO blocks.** None needs more than three lines.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

# ---- figure style, once, so every plot below matches the slides -----------
plt.rc('text', usetex=False)              # mathtext, not a LaTeX install
plt.rc('font', family='serif')            # serif, to match the deck
plt.rc('xtick', labelsize=12)             # tick labels readable from the back
plt.rc('ytick', labelsize=12)             #   of a lecture room
plt.rc('axes', labelsize=13, titlesize=13, grid=False)   # no grid, by choice
plt.rc('legend', fontsize=11, frameon=False)             # legend, no box
plt.rc('mathtext', fontset='cm')          # Computer Modern maths in labels
plt.rc('figure', dpi=130)                 # on-screen size; saved files use
                                          #   their own dpi

INK, MUTED = "#1F2933", "#4B5563"         # near-black for data, grey for notes
COLORS = ["#2E6FB7", "#D2691E", "#8E44AD", "#1E9E8A"]   # blue, orange, purple,
                                          #   teal: the four series colours

# Output folder: the slides' figures directory when it exists, else here.
from pathlib import Path
FIGDIR = Path.cwd()
for candidate in [Path("../../figures"), Path("../figures"), Path("figures")]:
    if candidate.is_dir():
        FIGDIR = candidate
        break


def finish(fig, name=None):
    """Fit the axes to their content, then show and optionally write the file.

    tight_layout removes the empty margins, so the figure in the notebook and
    the file on disk have the same, correctly fitted, layout.
    """
    fig.tight_layout()
    if name is not None:
        fig.savefig(FIGDIR / name, bbox_inches="tight")
        print("wrote", (FIGDIR / name).resolve())
    plt.show()

# %% [markdown]
# ## The problem
#
# The lifting $h(x)=1-x$ carries the boundary data, so that $v=u-h$ satisfies
# homogeneous conditions and, here, $v(x,0)=0$.

# %%
# ---- the problem ---------------------------------------------------------
X_F = 0.65        # where the source sits, in x.  A Gaussian bump centred here
SIGMA_F = 0.08    # how wide that Gaussian is: +/- 2 sigma covers 0.49 to 0.81


def source(x):
    """f(x): the fixed spatial shape of the forcing.

    A Gaussian of width SIGMA_F centred on X_F.  Its SHAPE never changes; the
    parameter alpha only scales it.  That is what lets the projected forcing
    vector be computed once and reused at any alpha.
    """
    return np.exp(-0.5 * ((x - X_F) / SIGMA_F) ** 2)


def lifting(x):
    """h(x) = 1 - x, which satisfies h(0)=1 and h(1)=0."""
    return 1.0 - x


# %% [markdown]
# ## Part A: the finite-difference reference solution
#
# Second-order central differences on a uniform grid, solving for $u$ itself.
# The two end values are held fixed, so only the interior evolves.

# %%
def fd_operators(n_x):
    """Central-difference matrices for the first and second derivative.

    Returns (x, dx, D1, D2) with D1 @ u approximating du/dx and D2 @ u
    approximating d2u/dx2, both to second order.

    The first and last ROWS are left at zero on purpose.  A zero row means
    "this node has no equation", which is how the Dirichlet conditions are
    imposed: the two boundary values never move from their initial value.
    Dense matrices are used because n_x is 200; at 10^6 points these would be
    sparse and nothing else in the file would change.
    """
    x = np.linspace(0.0, 1.0, n_x)        # uniform grid, endpoints included
    dx = x[1] - x[0]                      # spacing, 1/(n_x - 1)

    D1 = np.zeros((n_x, n_x))             # d/dx
    D2 = np.zeros((n_x, n_x))             # d2/dx2
    for j in range(1, n_x - 1):           # interior nodes only; rows 0 and -1
                                          #   stay zero, which pins the ends
        # TODO 1: fill the two central-difference stencils,
        #         u_x  ~ (u_{j+1} - u_{j-1}) / (2 dx)
        #         u_xx ~ (u_{j+1} - 2 u_j + u_{j-1}) / dx^2
        pass
    return x, dx, D1, D2


def fd_rhs(u, x, D1, D2, nu, alpha):
    """The right-hand side of the full-order model: du/dt at the current u.

    Three terms, and it is worth naming which parameter touches which:
        convection, nonlinear: u times its own slope, no parameter
        diffusion, linear:     the only term nu multiplies
        forcing, independent of u: the only term alpha touches
    """
    # TODO 2: assemble  -u * u_x + nu * u_xx + alpha * f(x).
    #         D1 @ u is u_x and D2 @ u is u_xx; the first product is
    #         ENTRYWISE (u times u_x point by point), not a matrix product.
    r = ...
    r[0] = 0.0          # u(0,t) = 1 for all t, so its derivative is zero
    r[-1] = 0.0         # likewise u(1,t) = 0
    return r


# %% [markdown]
# ## Time integration
#
# Two options, used by every part below.
#
# **RK4**, explicit, four right-hand side evaluations per step:
#
# $$a^{n+1}=a^{n}+\tfrac{\Delta t}{6}(k_1+2k_2+2k_3+k_4)$$
#
# **Semi-implicit**, for a system written as
# $M\dot a = c + L a + q(a)$: the stiff linear part is taken at $n+1$, the rest
# at $n$,
#
# $$(M-\Delta t\,L)\,a^{n+1} = M a^{n} + \Delta t\,\big(c+q(a^{n})\big).$$
#
# The matrix on the left never changes, so it is factorised once.

# %%
def integrate_rk4(rhs, y0, t_end, dt):
    """Classical fourth-order Runge-Kutta with a fixed step."""
    n_steps = int(round(t_end / dt))
    t = 0.0
    y = y0.copy()
    ts, ys = [t], [y.copy()]

    for _ in range(n_steps):
        # TODO 3: the four stages, then the weighted average
        k1 = ...
        k2 = ...
        k3 = ...
        k4 = ...
        y = ...
        t += dt
        ts.append(t)
        ys.append(y.copy())

    return np.array(ts), np.array(ys).T          # shape (n_dof, n_times)


def integrate_semi_implicit(M, L, const, nonlinear, y0, t_end, dt,
                            refactorise_every_step=False):
    """(M - dt L) y^{n+1} = M y^n + dt (const + nonlinear(y^n)).

    The matrix on the left depends only on M, L and dt, none of which change
    during the run.  So it is factorised ONCE, before the loop, and each step
    then costs only the two triangular sweeps of lu_solve, at O(n^2), instead
    of a fresh O(n^3) factorisation.

    Why LU and not Cholesky: M - dt L is not symmetric, because L carries the
    convective term.  Cholesky would apply to the diffusion-only case.

    Set refactorise_every_step=True to do the wasteful thing on purpose and
    measure the difference.
    """
    n_steps = int(round(t_end / dt))
    system = M - dt * L                      # constant throughout the run

    # ---- the single factorisation -------------------------------------
    # TODO 4a: factor the constant matrix once, here, outside the loop.
    lu = lu_factor(...)

    t = 0.0
    y = y0.copy()
    ts, ys = [t], [y.copy()]

    for _ in range(n_steps):
        # TODO 4b: build the right-hand side of the step
        rhs = ...

        if refactorise_every_step:
            lu = lu_factor(system)           # the same matrix, needlessly redone
        # ---- each step is only two triangular solves ------------------
        y = lu_solve(lu, rhs)

        t += dt
        ts.append(t)
        ys.append(y.copy())

    return np.array(ts), np.array(ys).T


# %% [markdown]
# ## The projection, written once
#
# Given a basis sampled on a quadrature rule, every operator of the lifted
# equation
#
# $$\partial_t v = \underbrace{(1-x)+\alpha f}_{\text{constant}}
#   + \underbrace{\nu\,\partial_{xx}v + v - (1-x)\partial_x v}_{\text{linear}}
#   \underbrace{-\,v\,\partial_x v}_{\text{quadratic}}$$
#
# is one weighted sum.  `Phi[q, n]` is $\phi_n$ at quadrature point $q$ and
# `dPhi[q, n]` its derivative there.

# %%
def assemble(x_q, w_q, Phi, dPhi):
    """Return the matrices of the projected system.

    M  a' = const + (alpha F) + (Lin - nu K) a + C:(a x a)

    with  M   mass matrix
          K   stiffness matrix (from integration by parts)
          Lin linear terms produced by the lifting
          C   quadratic convection tensor
    """
    W = w_q                                   # shorthand
    h = lifting(x_q)

    # TODO 5: every one of these is a weighted sum over the quadrature points.
    #         Phi[q, n] is phi_n at point q, dPhi[q, n] its derivative.
    M = ...                                   # <phi_m, phi_n>
    K = ...                                   # <phi_m', phi_n'>
    B = ...                                   # <phi_m, h phi_n'>
    Lin = M - B                               # the  +v - h v_x  terms
    G = ...                                   # <phi_m, h>
    F = ...                                   # <phi_m, f>

    # TODO 5b: C[m, n, k] = - sum_q w_q phi_m phi_n dphi_k
    C = ...

    return dict(M=M, K=K, Lin=Lin, G=G, F=F, C=C)


def reduced_rhs(op, a, nu, alpha):
    """The right-hand side  const + L a + C:(a x a)  of the projected system."""
    # TODO 6: the three pieces of  const + L a + C:(a x a)
    const = ...
    linear = ...
    quadratic = ...
    return const + linear + quadratic


# %% [markdown]
# ## Part B: the spectral basis
#
# $\phi_n(x)=\sqrt2\,\sin(n\pi x)$, which vanishes at both ends by construction.

# %%
def spectral_basis(n_b, n_q=2001):
    """The sine basis, sampled on a fine quadrature rule.

        phi_n(x) = sqrt(2) sin(n pi x),   n = 1 .. n_b

    Three properties earn this basis its place.  Each function vanishes at both
    ends, so the homogeneous conditions on v hold automatically.  The sqrt(2)
    normalises them, so the mass matrix comes out exactly the identity.  And
    they are eigenfunctions of d2/dx2, so diffusion becomes diagonal.

    Returns (x_q, w_q, Phi, dPhi): quadrature points and weights, then the
    basis and its derivative evaluated there.
    """
    x_q = np.linspace(0.0, 1.0, n_q)      # quadrature points
    w_q = np.full(n_q, x_q[1] - x_q[0])   # trapezoidal weights ...
    w_q[0] *= 0.5                         # ... halved at the two ends
    w_q[-1] *= 0.5

    Phi = np.zeros((n_q, n_b))            # Phi[q, n] = phi_n at point q
    dPhi = np.zeros((n_q, n_b))           # dPhi[q, n] = phi_n' at point q
    for n in range(1, n_b + 1):           # n is the wavenumber, starting at 1
        # TODO 7: phi_n = sqrt(2) sin(n pi x), and its derivative.
        #         Column n-1 holds mode n: the loop counts from 1, the array
        #         from 0.  Check afterwards that the mass matrix is the
        #         identity; if it is not, the sqrt(2) is missing.
        Phi[:, n - 1] = ...
        dPhi[:, n - 1] = ...
    return x_q, w_q, Phi, dPhi


# %% [markdown]
# ## Part C: the finite-element basis
#
# Piecewise linear hat functions on the interior nodes.  Same assembly code as
# part B: only `Phi` and `dPhi` change.

# %%
def fem_basis(n_elements, n_gauss=4):
    """Linear hat functions, sampled at the Gauss points of each element.

    One hat per INTERIOR node.  The two boundary nodes get none, which is how
    the homogeneous conditions on v are imposed: no basis function is nonzero
    there, so v cannot be either.

    Returned in the SAME format as spectral_basis, (x_q, w_q, Phi, dPhi, nodes),
    which is the whole trick of this tutorial: the assembly routine below never
    learns which basis it was given.

    Why Gauss points and not a uniform rule: a hat is piecewise linear, so
    products of two hats and a derivative are piecewise polynomials of low
    degree.  A four-point Gauss rule integrates them EXACTLY on each element,
    where a uniform rule across the kinks would not.  The quadrature is built
    element by element for the same reason.
    """
    nodes = np.linspace(0.0, 1.0, n_elements + 1)   # element boundaries
    dx = nodes[1] - nodes[0]                        # element length
    n_dof = n_elements - 1                          # interior nodes only

    # Gauss-Legendre points and weights, mapped from [-1, 1] to [0, 1].
    xi, wi = np.polynomial.legendre.leggauss(n_gauss)
    xi, wi = 0.5 * (xi + 1.0), 0.5 * wi

    # Lay those points down inside every element, and collect them all.
    x_q, w_q = [], []
    for e in range(n_elements):
        x_q.append(nodes[e] + dx * xi)    # the element's own Gauss points
        w_q.append(dx * wi)               # weights scaled by the element length
    x_q, w_q = np.concatenate(x_q), np.concatenate(w_q)

    Phi = np.zeros((len(x_q), n_dof))     # Phi[q, n] = hat n at point q
    dPhi = np.zeros((len(x_q), n_dof))
    for n in range(n_dof):
        xc = nodes[n + 1]                 # the node this hat is centred on
        # A hat is nonzero only on the two elements touching its node: it rises
        # linearly from zero at xc - dx to one at xc, then falls back to zero at
        # xc + dx.  These two boolean masks select the points on each side.
        left = (x_q >= xc - dx) & (x_q <= xc)
        right = (x_q > xc) & (x_q <= xc + dx)
        Phi[left, n] = (x_q[left] - (xc - dx)) / dx      # rising limb
        Phi[right, n] = ((xc + dx) - x_q[right]) / dx    # falling limb
        dPhi[left, n] = 1.0 / dx          # slope is constant on each side ...
        dPhi[right, n] = -1.0 / dx        # ... and jumps at the node itself

    return x_q, w_q, Phi, dPhi, nodes


# %% [markdown]
# ## Solve the three of them
#
# Parameters, and the reference finite-difference run.

# %%
# ---- the operating point solved by all three methods ---------------------
NU = 0.03         # viscosity.  Small enough to form a front, large enough that
                  #   the cell Peclet number stays at 0.17 on 200 points
ALPHA = 1.0       # amplitude of the source term
T_END = 1.0       # final time; every run below integrates from 0 to T_END

# ---- Part A: finite differences
n_x = 200         # grid points, boundaries included, so 198 unknowns evolve
x_fd, dx, D1, D2 = fd_operators(n_x)
u0 = lifting(x_fd)                              # the initial condition

dt_fd = 0.4 * dx**2 / NU                        # diffusive stability limit
t_fd, U_fd = integrate_rk4(lambda u: fd_rhs(u, x_fd, D1, D2, NU, ALPHA),
                           u0, T_END, dt_fd)
print(f"finite differences: {n_x} points, dt = {dt_fd:.2e}, "
      f"{len(t_fd)-1} steps")

# %%
def dt_spectral(nb, nu=None, safety=0.4, dt_max=2e-3):
    """A time step RK4 can take with nb sine modes.

    The stiffest eigenvalue of the diffusion operator is nu (nb pi)^2, and RK4
    is stable on the real axis for |lambda| dt < 2.78.  The cap dt_max is there
    for accuracy, not stability: with few modes the stability limit is so loose
    that the time discretisation would become the dominant error.

    Using the stability limit rather than a fixed small number is what makes
    the cost comparison below fair: each method is run with the step it is
    entitled to.
    """
    nu = NU if nu is None else nu
    return min(dt_max, safety * 2.78 / (nu * (nb * np.pi) ** 2))


# ---- Part B: spectral Galerkin
n_b = 16          # sine modes.  Chosen from the convergence study at the end:
                  #   16 already matches the 200-point reference
xq_s, wq_s, Phi_s, dPhi_s = spectral_basis(n_b)
op_s = assemble(xq_s, wq_s, Phi_s, dPhi_s)

a0 = np.zeros(n_b)                              # because v(x, 0) = 0
dt_sp = dt_spectral(n_b)
t_sp, A_sp = integrate_rk4(
    lambda a: reduced_rhs(op_s, a, NU, ALPHA), a0, T_END, dt_sp)
print(f"spectral: n_b = {n_b}, dt = {dt_sp:.2e}, {len(t_sp)-1} steps, "
      f"mass matrix is identity? "
      f"{np.allclose(op_s['M'], np.eye(n_b), atol=1e-8)}")

# %%
# ---- Part C: finite elements, with the semi-implicit solver
n_elements = 50   # linear elements, so 51 nodes and 49 interior unknowns
xq_f, wq_f, Phi_f, dPhi_f, nodes = fem_basis(n_elements)
op_f = assemble(xq_f, wq_f, Phi_f, dPhi_f)

n_dof = Phi_f.shape[1]
L_f = op_f["Lin"] - NU * op_f["K"]
const_f = op_f["G"] + ALPHA * op_f["F"]
quad_f = lambda a: np.einsum("mnk,n,k->m", op_f["C"], a, a, optimize=True)

t_fe, A_fe = integrate_semi_implicit(
    op_f["M"], L_f, const_f, quad_f, np.zeros(n_dof), T_END, 2e-3)
print(f"finite elements: {n_elements} elements, {n_dof} unknowns, "
      f"{len(t_fe)-1} steps with dt = 2e-3")


# %% [markdown]
# ## Reconstruct and compare
#
# Every method predicts $u = h + \sum_n a_n \phi_n$, except the
# finite-difference one, which gives $u$ directly.

# %%
def reconstruct(x, basis_fun, a):
    """Rebuild the field from its coefficients: u = h + sum_n a_n phi_n.

    Adding the lifting back is not optional.  Forget it and the result
    satisfies u(0) = 0 instead of 1, and every error measure below becomes
    meaningless.  It is the single most common mistake in this tutorial.
    """
    return lifting(x) + basis_fun(x) @ a


def spectral_at(x, n_b):
    """The sine basis evaluated at arbitrary points, for plotting.

    Same functions as spectral_basis, without the quadrature: the coefficients
    are already known by the time this is called, and only the shapes are
    wanted.  column_stack makes one column per mode, matching Phi's layout.
    """
    return np.column_stack([np.sqrt(2) * np.sin(n * np.pi * x)
                            for n in range(1, n_b + 1)])


def fem_at(x, nodes):
    """The hat functions evaluated at arbitrary points, for plotting.

    1 - |x - xc|/dx is the hat centred on xc written in one expression, and
    np.maximum clips it at zero outside the two elements that touch the node.
    nodes[1:-1] skips the two boundary nodes, which carry no basis function.
    """
    dx = nodes[1] - nodes[0]
    return np.column_stack([np.maximum(0.0, 1.0 - np.abs(x - xc) / dx)
                            for xc in nodes[1:-1]])


def at_time(t_array, Y, t_target):
    """The column of Y closest to t_target."""
    return Y[:, np.argmin(np.abs(t_array - t_target))]


# %%
t_show = 0.5
x_plot = np.linspace(0, 1, 400)

u_fd_t = at_time(t_fd, U_fd, t_show)
u_sp_t = reconstruct(x_plot, lambda x: spectral_at(x, n_b),
                     at_time(t_sp, A_sp, t_show))
u_fe_t = reconstruct(x_plot, lambda x: fem_at(x, nodes),
                     at_time(t_fe, A_fe, t_show))

fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.plot(x_fd, u_fd_t, color=INK, lw=2.4, label="finite differences")
ax.plot(x_plot, u_sp_t, color=COLORS[1], lw=1.8, ls="--",
        label=rf"spectral, $n_b={n_b}$")
ax.plot(x_plot, u_fe_t, color=COLORS[3], lw=1.8, ls=":",
        label=rf"finite elements, {n_dof} unknowns")
ax.set_xlabel(r"$x$")
ax.set_ylabel(rf"$u(x,\, t={t_show})$")
ax.set_xlim(0.0, 1.0)                     # axis tight in x
ax.margins(y=0.05)                        # a thin margin in y, nothing more
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "t2_three_methods_snapshot.png")

# %% [markdown]
# ## What the single factorisation buys
#
# The matrix $M-\Delta t\,L$ does not change during the run, so it is factorised
# once, before the loop.  Each step is then two triangular solves.  Running the
# same integration while refactorising at every step measures the difference.

# %%
import time


def average_time(function, repeats=20):
    t0 = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - t0) / repeats


# Time the two operations separately, on a system large enough to see them.
xq, wq, P, dP, nds = fem_basis(200)
op_big = assemble(xq, wq, P, dP)
n_big = P.shape[1]
system = op_big["M"] - 2e-3 * (op_big["Lin"] - NU * op_big["K"])
lu_big = lu_factor(system)
rhs_big = np.ones(n_big)

t_factor = average_time(lambda: lu_factor(system))
t_solve = average_time(lambda: lu_solve(lu_big, rhs_big))

n_steps = int(round(T_END / 2e-3))
print(f"system size                 {n_big}")
print(f"one factorisation, O(n^3)   {t_factor*1e3:8.3f} ms")
print(f"one solve, O(n^2)           {t_solve*1e3:8.3f} ms"
      f"    ({t_factor/t_solve:.0f} times cheaper)")
print(f"over {n_steps} steps:")
print(f"  factorise once            {(t_factor + n_steps*t_solve):8.3f} s")
print(f"  refactorise every step    {(n_steps*(t_factor + t_solve)):8.3f} s")

# The flag below runs the wasteful version for real, if you want to check.
# In this tutorial the saving is partly hidden, because the quadratic tensor
# contraction is itself O(n^3) per step and dominates the wall clock.  Knowing
# which term dominates is half of making a code fast.

# %% [markdown]
# ## An animation of the three solutions
#
# The same instants for all three methods, written to an animated GIF.
# Set `layout="stacked"` for three separate panels instead of one overlay.

# %%
import io

from PIL import Image


def make_animation(filename="t2_three_methods.gif", n_frames=50,
                   layout="overlay", fps=12, framedir="t2_frames"):
    """Animate the finite-difference, spectral and finite-element solutions.

    Writes an animated GIF and, in framedir, the same frames as numbered PNG
    files.  The slides replay those with \animategraphics, which needs the
    frames as separate images.
    """
    times = np.linspace(0.0, T_END, n_frames)
    xg = np.linspace(0, 1, 400)

    # Precompute every frame, so the drawing loop stays trivial.
    frames_fd, frames_sp, frames_fe = [], [], []
    Phi_sp = spectral_at(xg, n_b)
    Phi_fe = fem_at(xg, nodes)
    for t in times:
        frames_fd.append(at_time(t_fd, U_fd, t))
        frames_sp.append(lifting(xg) + Phi_sp @ at_time(t_sp, A_sp, t))
        frames_fe.append(lifting(xg) + Phi_fe @ at_time(t_fe, A_fe, t))

    lo = min(np.min(f) for f in frames_fd) - 0.05
    hi = max(np.max(f) for f in frames_fd) + 0.10

    styles = [("finite differences", INK, "-", 2.4),
              (rf"spectral, $n_b={n_b}$", COLORS[1], "--", 1.8),
              (rf"finite elements, {n_dof} unknowns", COLORS[3], ":", 2.0)]

    if layout == "overlay":
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        axes = [ax, ax, ax]
    else:
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True)

    lines = []
    for axis, (label, colour, ls, lw) in zip(axes, styles):
        line, = axis.plot([], [], color=colour, ls=ls, lw=lw, label=label)
        lines.append(line)
        axis.set_xlim(0, 1)
        axis.set_ylim(lo, hi)
        axis.spines[["top", "right"]].set_visible(False)
        if layout != "overlay":
            axis.set_title(label, fontsize=13)
        axis.set_ylabel(r"$u(x,t)$")

    axes[-1].set_xlabel(r"$x$")
    if layout == "overlay":
        # Above the axes, so that it never collides with the solution.
        axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                       ncol=3, fontsize=12)
    # The running label goes inside the axes, low on the left, where the
    # solution never passes and where it cannot be clipped.  It is read from
    # the back of a lecture room, so it is set large.
    title = axes[0].text(0.03, 0.36, "", transform=axes[0].transAxes,
                         fontsize=17, va="top")

    def update(k):
        for line, data in zip(lines, (frames_fd[k], frames_sp[k], frames_fe[k])):
            line.set_data(x_fd if line is lines[0] else xg, data)
        title.set_text(rf"$t = {times[k]:.3f}$" + "\n"
                       + rf"$\nu = {NU},\ \alpha = {ALPHA}$")
        return lines + [title]

    fig.tight_layout()
    # The frames are rendered by hand and collected with Pillow.  Going
    # through matplotlib.animation is fragile on interactive backends: the
    # animation installs a draw callback, and a later plt.show() redraws the
    # figure once its event source is already gone.  The loop below has no
    # event loop at all, so it behaves the same everywhere.
    #
    # One tight bounding box, computed once and reused.  Every frame has the
    # same layout, so the box is the same, and the frames all come out at
    # exactly the same size, which both the GIF and animategraphics need.
    update(0)
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.03)

    outdir = FIGDIR / framedir if framedir else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    images = []
    for k in range(n_frames):
        update(k)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches=bbox)
        buf.seek(0)
        images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE,
                                              colors=64))
        buf.close()
        if outdir is not None:
            fig.savefig(outdir / f"t2_frame_{k:03d}.png", dpi=100,
                        bbox_inches=bbox)

    images[0].save(FIGDIR / filename, save_all=True,
                   append_images=images[1:],
                   duration=int(round(1000.0 / fps)), loop=0)
    plt.close(fig)
    print("wrote", (FIGDIR / filename).resolve())
    if outdir is not None:
        print(f"wrote {n_frames} frames in {outdir.resolve()}"
              f"  (last index {n_frames - 1:03d}, {fps} fps)")


make_animation()

# %% [markdown]
# ## What one run costs
#
# The three methods solve the same problem to comparable accuracy, so the fair
# comparison is wall-clock time for one run over $[0,T]$.  Keep these numbers:
# in Tutorial 3 the POD-Galerkin model is measured against exactly this
# finite-difference baseline.

# %%
def time_run(function, repeats=3):
    """Best of a few repetitions, which is the least noisy estimate."""
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - t0)
    return best


cost_fd = time_run(lambda: integrate_rk4(
    lambda u: fd_rhs(u, x_fd, D1, D2, NU, ALPHA), u0, T_END, dt_fd))

cost_sp = time_run(lambda: integrate_rk4(
    lambda a: reduced_rhs(op_s, a, NU, ALPHA), np.zeros(n_b), T_END, dt_sp))

L_fe = op_f["Lin"] - NU * op_f["K"]
c_fe = op_f["G"] + ALPHA * op_f["F"]
q_fe = lambda a: np.einsum("mnk,n,k->m", op_f["C"], a, a, optimize=True)
cost_fe = time_run(lambda: integrate_semi_implicit(
    op_f["M"], L_fe, c_fe, q_fe, np.zeros(n_dof), T_END, 2e-3))

COSTS = {"finite differences": (len(x_fd) - 2, cost_fd),
         "spectral Galerkin": (n_b, cost_sp),
         "finite elements": (n_dof, cost_fe)}

print(f"{'method':<22}{'unknowns':>10}{'time [s]':>12}{'vs FD':>10}")
for name, (n, c) in COSTS.items():
    print(f"{name:<22}{n:>10d}{c:>12.3f}{cost_fd / c:>9.1f}x")

# %%
fig, ax = plt.subplots(figsize=(7.0, 3.6))
names = list(COSTS)
times = [COSTS[k][1] for k in names]
bars = ax.bar(range(len(names)), times,
              color=[INK, COLORS[1], COLORS[3]], width=0.55)
for k, (bar, name) in enumerate(zip(bars, names)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{COSTS[name][1]:.2f} s", ha="center", va="bottom", fontsize=11)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([rf"{n}" + "\n" + rf"${COSTS[n][0]}$ unknowns" for n in names])
ax.set_ylabel(r"wall clock for one run [s]")
ax.margins(y=0.18)
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "t2_cost.png")

# The three costs are not directly comparable as they stand: each solver uses
# the time step its own stability allows.  That is the point.  The explicit
# finite-difference run is limited by dt < dx^2 / nu, the spectral run by the
# largest eigenvalue of L, and the semi-implicit finite-element run by accuracy
# alone.  A reduced model changes the same three things at once, which is why
# Tutorial 3 measures the end-to-end time rather than the cost per step.

# %% [markdown]
# ## Convergence
#
# Relative $L^2$ error against the finite-difference reference, as the number of
# unknowns grows.
#
# Note what the reference is: a second-order scheme on 200 points, not the exact
# solution.  Both curves therefore flatten out once they reach the accuracy of
# that reference, and the plateau measures the reference, not the method.
#
# The abscissa is $n/N_{\rm FD}$, the size of the model relative to the
# reference.  Read at a fixed error, it is the compression factor: how much of
# the full-order model the basis has managed to remove.

# %%
def relative_error(u_ref_on_grid, u_on_grid, x):
    w = np.gradient(x)
    num = np.sqrt(np.sum(w * (u_on_grid - u_ref_on_grid) ** 2))
    den = np.sqrt(np.sum(w * u_ref_on_grid ** 2))
    return num / den


u_ref = at_time(t_fd, U_fd, t_show)

errs_sp, errs_fe = [], []
nb_list = [2, 4, 8, 16, 32]
ne_list = [10, 20, 40, 80, 160]

for nb in nb_list:
    xq, wq, P, dP = spectral_basis(nb)
    op = assemble(xq, wq, P, dP)
    t_, A_ = integrate_rk4(lambda a: reduced_rhs(op, a, NU, ALPHA),
                           np.zeros(nb), T_END, dt_spectral(nb))
    u_ = reconstruct(x_fd, lambda x: spectral_at(x, nb), at_time(t_, A_, t_show))
    errs_sp.append(relative_error(u_ref, u_, x_fd))

for ne in ne_list:
    xq, wq, P, dP, nds = fem_basis(ne)
    op = assemble(xq, wq, P, dP)
    Lf = op["Lin"] - NU * op["K"]
    cf = op["G"] + ALPHA * op["F"]
    qf = lambda a: np.einsum("mnk,n,k->m", op["C"], a, a, optimize=True)
    t_, A_ = integrate_semi_implicit(op["M"], Lf, cf, qf,
                                     np.zeros(P.shape[1]), T_END, 2e-3)
    u_ = reconstruct(x_fd, lambda x: fem_at(x, nds), at_time(t_, A_, t_show))
    errs_fe.append(relative_error(u_ref, u_, x_fd))

# %%
# The abscissa is the model size relative to the reference, n / N_FD.  The
# absolute count says how big the model is; the ratio says how much of the
# full-order model has been removed, which is the quantity a digital twin is
# judged on.
n_fd_dof = n_x - 2                        # the two boundary values are known

ratio_sp = [nb / n_fd_dof for nb in nb_list]
ratio_fe = [(ne - 1) / n_fd_dof for ne in ne_list]

fig, ax = plt.subplots(figsize=(6.8, 4.0))
ax.loglog(ratio_sp, errs_sp, "o-", color=COLORS[1], lw=1.8,
          label="spectral Galerkin")
ax.loglog(ratio_fe, errs_fe, "s--", color=COLORS[3], lw=1.8,
          label="finite elements")
ax.set_xlabel(rf"$n \, / \, N_{{\rm FD}}$, with $N_{{\rm FD}}={n_fd_dof}$")
ax.set_ylabel(r"$\|u - u_{\rm FD}\|_{L^2} \, / \, \|u_{\rm FD}\|_{L^2}$")
ax.margins(x=0.05, y=0.10)                # the data set the limits, not the axes
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# The same axis read as a plain count, on top.
count = ax.secondary_xaxis("top", functions=(lambda r: r * n_fd_dof,
                                             lambda n: n / n_fd_dof))
count.set_xlabel(r"number of unknowns $n$")

finish(fig, "t2_convergence.png")

for nb, e in zip(nb_list, errs_sp):
    print(f"spectral n_b = {nb:3d}   error = {e:.3e}")
for ne, e in zip(ne_list, errs_fe):
    print(f"FEM  {ne:3d} elements   error = {e:.3e}")

# %% [markdown]
# ## Questions
#
# 1. How many sine modes are needed before the spectral solution becomes
#    indistinguishable from the finite-difference reference?
# 2. On the convergence plot, which curve is straight on a log-log axis and
#    which would be straight on a semilog axis?  What does that say about the
#    two methods?
# 3. Which lines of code actually changed between part B and part C?
# 4. Replace the semi-implicit solver in part C by RK4.  How small must
#    $\Delta t$ become, and why?
