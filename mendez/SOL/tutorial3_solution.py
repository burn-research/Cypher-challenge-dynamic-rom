# %% [markdown]
# # Tutorial 3: POD and a POD-Galerkin reduced-order model
#
# The plan, in one line: run the expensive model a few times, let the data
# choose the basis, project onto it, and pay for the result.
#
# 1. **Offline.** Solve the finite-difference model of Tutorial 2 for several
#    parameter pairs $\boldsymbol{\mu}=(\nu,\alpha)$, grouped into two *classes*.
# 2. **Compress.** Take the leading POD modes of each class and merge them into
#    one **global basis**.
# 3. **Project.** Put that basis into the Galerkin machinery of Tutorial 2.
#    Nothing else changes: same lifting, same constant, linear and quadratic
#    terms, same time integrator.
# 4. **Online.** Predict a parameter pair that was never simulated, and measure
#    both the error and the time.
#
# The file is self-contained: it re-derives the finite-difference model rather
# than importing Tutorial 2, so it can be run on its own.
#
# **There are eight TODO blocks in the version to fill.**

# %%
import time
from pathlib import Path

import numpy as np                       # arrays and linear algebra
import matplotlib.pyplot as plt           # every figure below
from scipy.linalg import lu_factor, lu_solve   # LU once, then reuse each step

# ---- figure style, once, so every plot below matches the slides -----------
plt.rc('text', usetex=False)              # mathtext, not a LaTeX install
plt.rc('font', family='serif')            # serif, to match the deck
plt.rc('xtick', labelsize=12)             # tick labels readable from the back
plt.rc('ytick', labelsize=12)             #   of a lecture room
plt.rc('axes', labelsize=13, titlesize=13, grid=False)   # no grid, by choice
plt.rc('legend', fontsize=11, frameon=False)             # legend, no box
plt.rc('mathtext', fontset='cm')          # Computer Modern maths in labels
plt.rc('figure', dpi=130)                 # on-screen size; files are saved
                                          #   at their own dpi

INK, MUTED = "#1F2933", "#4B5563"         # near-black for data, grey for notes
COLORS = ["#2E6FB7", "#D2691E", "#8E44AD", "#1E9E8A"]   # blue, orange, purple,
                                          #   teal: the four series colours

# Where the figures are written.  The script may be run from the repository
# (in which case the slides' figures/ folder is two levels up) or on its own
# (in which case the current folder will do).  The first candidate that exists
# wins, so the same file works in both places and on Colab.
FIGDIR = Path.cwd()                       # fallback: write next to the script
for candidate in [Path("../../figures"), Path("../figures"), Path("figures")]:
    if candidate.is_dir():
        FIGDIR = candidate
        break


def finish(fig, name=None):
    """Fit the axes to their content, then show and optionally write the file.

    Called at the end of every plotting cell so that the figure shown in the
    notebook and the file written to disk have the same, correctly fitted
    layout.  Without tight_layout the two differ and labels get clipped.
    """
    fig.tight_layout()                    # remove the empty margins
    if name is not None:                  # None means "show, do not save"
        fig.savefig(FIGDIR / name, bbox_inches="tight")   # trim to the content
        print("wrote", (FIGDIR / name).resolve())
    plt.show()


# %% [markdown]
# ## The full-order model
#
# The problem of Tutorial 2, unchanged:
#
# $$\partial_t u + u\,\partial_x u = \nu\,\partial_{xx}u + \alpha f(x),
#   \qquad u(0,t)=1,\quad u(1,t)=0,\quad u(x,0)=1-x$$
#
# with $f(x)=\exp[-(x-x_f)^2/2\sigma_f^2]$, $x_f=0.65$, $\sigma_f=0.08$.
# The parameters are $\boldsymbol{\mu}=(\nu,\alpha)$.

# %%
# ---- the problem ---------------------------------------------------------
X_F = 0.65        # where the source sits, in x.  A Gaussian bump centred here
SIGMA_F = 0.08    # how wide that Gaussian is: +/- 2 sigma covers 0.49 to 0.81
T_END = 1.0       # final time.  Every run below integrates from 0 to T_END

# ---- the discretisation --------------------------------------------------
N_X = 200         # grid points, boundaries included, so 198 unknowns evolve


def source(x):
    """f(x): the fixed spatial shape of the forcing.

    A Gaussian of width SIGMA_F centred on X_F.  It does NOT depend on the
    parameters: alpha multiplies it, but its shape never changes.  That is what
    makes the reduced forcing vector reusable at any alpha.
    """
    return np.exp(-((x - X_F) ** 2) / (2.0 * SIGMA_F ** 2))


def lifting(x):
    """h(x) = 1 - x, the function that carries the boundary data.

    u(0)=1 and u(1)=0 are non-homogeneous, and a basis cannot satisfy them
    unless every basis function does.  Writing u = h + v with h(0)=1, h(1)=0
    moves the boundary data into h and leaves v with v(0)=v(1)=0, which any
    sine, hat or POD mode can satisfy.  Here it also gives v(x,0)=0, so the
    reduced model starts from a(0)=0.
    """
    return 1.0 - x


def fd_operators(n_x):
    """Central-difference matrices for the first and second derivative.

    Returns (x, dx, D1, D2) with D1 @ u approximating du/dx and D2 @ u
    approximating d2u/dx2, both to second order.

    The first and last ROWS are left at zero on purpose.  A zero row means
    "this node has no equation", which is how the Dirichlet conditions are
    imposed: the two boundary values never change from their initial value.
    Dense matrices are used because n_x is 200; at 10^6 points these would be
    sparse, and nothing else in the file would change.
    """
    x = np.linspace(0.0, 1.0, n_x)        # uniform grid, endpoints included
    dx = x[1] - x[0]                      # spacing, 1/(n_x - 1)

    D1 = np.zeros((n_x, n_x))             # d/dx
    D2 = np.zeros((n_x, n_x))             # d2/dx2
    for j in range(1, n_x - 1):           # interior nodes only; rows 0 and -1
                                          #   stay zero, which pins the ends
        # (u[j+1] - u[j-1]) / (2 dx)
        D1[j, j - 1], D1[j, j + 1] = -1.0 / (2 * dx), 1.0 / (2 * dx)
        # (u[j+1] - 2 u[j] + u[j-1]) / dx^2
        D2[j, j - 1], D2[j, j], D2[j, j + 1] = 1 / dx**2, -2 / dx**2, 1 / dx**2
    return x, dx, D1, D2


def fd_rhs(u, x, D1, D2, nu, alpha):
    """The right-hand side of the full-order model: du/dt at the current u.

    This is Burgers written out term by term:
        -u * (D1 @ u)      convection, the nonlinear term, u times its own slope
        +nu * (D2 @ u)     diffusion, linear, the only term nu multiplies
        +alpha * source(x) forcing, independent of u, the only term alpha touches
    """
    r = -u * (D1 @ u) + nu * (D2 @ u) + alpha * source(x)
    r[0] = 0.0        # u(0,t) = 1 for all t, so its time derivative is zero
    r[-1] = 0.0       # likewise u(1,t) = 0
    return r


def integrate_rk4(rhs, y0, t_end, dt, store_every=1):
    """Classical explicit fourth-order Runge-Kutta.

    rhs(y)      -> dy/dt at the state y
    y0          -> initial state
    store_every -> keep one state in this many.  It thins the OUTPUT only; every
                   step is still taken.  Used because a full-order run here is
                   3000 steps and 100 stored snapshots is plenty.

    Returns (ts, Y) with Y[:, k] the state at time ts[k].
    """
    n_steps = int(round(t_end / dt))      # how many steps to reach t_end
    t, y = 0.0, y0.copy()                 # copy: never modify the caller's y0
    ts, ys = [t], [y.copy()]              # the initial state is stored too
    for step in range(1, n_steps + 1):
        k1 = rhs(y)                       # slope at the start of the step
        k2 = rhs(y + 0.5 * dt * k1)       # slope at the midpoint, using k1
        k3 = rhs(y + 0.5 * dt * k2)       # midpoint again, using k2
        k4 = rhs(y + dt * k3)             # slope at the end, using k3
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)   # weighted average
        t += dt
        if step % store_every == 0 or step == n_steps:     # always keep the last
            ts.append(t)
            ys.append(y.copy())
    return np.array(ts), np.array(ys).T   # .T so columns are snapshots


def integrate_semi_implicit(M, L, const, nonlinear, y0, t_end, dt):
    """Stiff linear part implicit, everything else explicit.

    Solves  M y' = const + L y + nonlinear(y)  by

        (M - dt L) y^{n+1} = M y^n + dt (const + nonlinear(y^n))

    Taking L at n+1 removes the diffusive step limit, which is what lets the
    reduced runs use dt = 1e-2 where the explicit full-order run needs 3e-4.
    Almost all of the speed-up measured at the end of this file comes from
    that, not from the smaller state.

    The matrix M - dt L never changes during the run, so it is factorised ONCE
    before the loop and each step costs only the two triangular solves inside
    lu_solve, O(n^2) instead of O(n^3).
    """
    n_steps = int(round(t_end / dt))
    lu = lu_factor(M - dt * L)            # the single O(n^3) factorisation
    t, y = 0.0, y0.copy()
    ts, ys = [t], [y.copy()]
    for _ in range(n_steps):
        rhs = M @ y + dt * (const + nonlinear(y))   # everything known at time n
        y = lu_solve(lu, rhs)             # two triangular sweeps, O(n^2)
        t += dt
        ts.append(t)
        ys.append(y.copy())
    return np.array(ts), np.array(ys).T


def at_time(ts, Y, t):
    """The stored state closest to time t.  No interpolation: runs are stored
    densely enough that the nearest instant is within a fraction of a percent
    of t, and interpolating would blur a comparison between two solvers that
    do not share a time grid."""
    return Y[:, int(np.argmin(np.abs(ts - t)))]


# Build the grid and the operators once; every run below reuses them.
x_fd, dx, D1, D2 = fd_operators(N_X)      # grid, spacing, d/dx, d2/dx2
h_vec = lifting(x_fd)                     # h(x) sampled on the grid
f_vec = source(x_fd)                      # f(x) sampled on the grid


def dt_fom(nu):
    """Largest stable step for the EXPLICIT full-order run.

    RK4 on a diffusion term is stable while nu dt / dx^2 stays below about one;
    0.4 is a safety factor.  Note the dx^2: halving the mesh quarters the step
    and so quadruples the work.  This is the cost the reduced model escapes.
    """
    return 0.4 * dx**2 / nu


def run_fom(nu, alpha, store_every=30):
    """One full-order run at the parameter pair (nu, alpha).

    The initial condition is u(x,0) = 1 - x, which is h itself, so v(x,0) = 0.
    Returns (ts, U) with U[:, k] the field u(x, ts[k]).
    """
    return integrate_rk4(lambda u: fd_rhs(u, x_fd, D1, D2, nu, alpha),
                         h_vec.copy(),    # u(x, 0) = h(x) = 1 - x
                         T_END, dt_fom(nu),
                         store_every=store_every)


# %% [markdown]
# ## Part A: the offline stage
#
# Two **classes** of parameters, three runs each.  A class is a group of
# operating conditions that share a physical regime: here, a diffusive one and
# a convection-dominated one.  They are deliberately kept apart, because a
# basis trained on one of them alone will not serve the other.
#
# Each run is thinned to about a hundred stored instants.  Every stored column
# is a snapshot.

# %%
# The training design.  Two classes, three runs each, six runs in total.
#   nu = 0.10 : diffusion dominates, the solution is smooth and settles fast
#   nu = 0.01 : convection dominates, a steep front forms near x = 0.8
# The two values sit at the ends of the parameter range nu in [0.01, 0.10] on
# purpose: a basis has to cover the whole range, and the ends are where it is
# hardest.  alpha only scales the forcing, so three values per class sample it
# adequately.  "colour" is used by the figures, nothing more.
CLASSES = {
    "diffusive":  dict(nu=0.10, alphas=[0.5, 1.0, 1.5], colour=COLORS[0]),
    "convective": dict(nu=0.01, alphas=[0.5, 1.0, 1.5], colour=COLORS[1]),
}

offline_t0 = time.perf_counter()          # start the offline clock

runs = {}                                 # (nu, alpha) -> (times, U)
for name, spec in CLASSES.items():        # for each class ...
    for alpha in spec["alphas"]:          # ... and each forcing amplitude
        mu = (spec["nu"], alpha)          # the parameter pair of this run
        runs[mu] = run_fom(*mu)           # one full-order solve

cost_offline_runs = time.perf_counter() - offline_t0     # what the runs cost
n_snap_total = sum(U.shape[1] for _, U in runs.values()) # columns collected
print(f"{len(runs)} training runs, {n_snap_total} snapshots, "
      f"{cost_offline_runs:.2f} s")


# %% [markdown]
# ## The snapshot matrix
#
# The reduced model will be written for $v=u-h$, which vanishes at both ends,
# so the snapshots are taken of $v$ and not of $u$.  The lifting carries the
# boundary data and is not part of what has to be learned.
#
# $$\boldsymbol{D}=[\boldsymbol{d}_1,\dots,\boldsymbol{d}_{n_t}]\in\mathbb{R}^{n_s\times n_t},
#   \qquad \boldsymbol{d}_k=v(x_i,t_k;\boldsymbol{\mu}_k)$$

# %%
def snapshots_of(mu_list):
    """Build a snapshot matrix from the listed runs.

    Each column is v = u - h at one instant of one run, so the boundary data
    carried by h is subtracted out and every column vanishes at both ends.
    Learning h would be a waste: it is known exactly and never changes.

    runs[mu] is the pair (times, U); runs[mu][1] is U, of shape (n_s, n_stored).
    h_vec[:, None] makes h a column so numpy subtracts it from every column.
    """
    return np.hstack([runs[mu][1] - h_vec[:, None] for mu in mu_list])


D_all = snapshots_of(list(runs))          # every snapshot of every training run
print("dataset matrix:", D_all.shape, "  (n_s x n_t)")


# %% [markdown]
# ## The inner product, and weighted POD
#
# Section 2 again: the SVD of $\boldsymbol{D}$ is optimal in the Euclidean norm of
# $\mathbb{R}^{n_s}$, whereas the quantity we care about is the $L^2(\Omega)$
# error.  On a uniform grid the two differ only by a constant, but writing the
# weights explicitly is what makes the code correct on a non-uniform grid, and
# it is what makes the reduced mass matrix exactly the identity.
#
# With $\boldsymbol{W}_s=\mathrm{diag}(w_i)$ the quadrature weights,
#
# $$\tilde{\boldsymbol{D}}=\boldsymbol{W}_s^{1/2}\boldsymbol{D}=\tilde{\boldsymbol{\Phi}}\,\boldsymbol{\Sigma}\,\boldsymbol{\Psi}^\top,
#   \qquad \boldsymbol{\Phi}=\boldsymbol{W}_s^{-1/2}\tilde{\boldsymbol{\Phi}}
#   \quad\Longrightarrow\quad \boldsymbol{\Phi}^\top\boldsymbol{W}_s\boldsymbol{\Phi}=\boldsymbol{I}.$$

# %%
# Quadrature weights: the trapezoidal rule on [0, 1].  Interior points carry a
# full dx, the two endpoints half of one.  These weights turn a sum over grid
# points into an approximation of an integral over the domain, which is what
# makes the inner product below an L2 inner product and not a dot product.
w_s = np.full(N_X, dx)                    # dx everywhere ...
w_s[0] = w_s[-1] = 0.5 * dx               # ... except half at the two ends
sqrt_w = np.sqrt(w_s)                     # W^(1/2), used as a row scaling


def weighted_pod(D, n_keep=None):
    """POD of D in the W_s inner product, not the Euclidean one.

    Why the weighting matters: an ordinary SVD of D gives modes orthonormal
    under the dot product, whereas everything else in this file measures error
    in L2.  If the two disagree, the reduced mass matrix is not the identity
    and the model quietly solves the wrong problem.

    The fix is one line each way.  Scale the rows by sqrt(w) before the SVD, so
    the Euclidean geometry of the scaled matrix IS the weighted geometry of the
    original; then divide the left singular vectors by sqrt(w) to bring them
    back.  The result satisfies Phi^T W_s Phi = I exactly.

    Returns (Phi, sigma, Psi): spatial modes as columns, singular values, and
    temporal structures as columns.
    """
    # sqrt_w[:, None] * D scales row i by sqrt(w_i).  full_matrices=False gives
    # the thin SVD: only min(n_s, n_t) modes, which is all that exist.
    U, sigma, Vt = np.linalg.svd(sqrt_w[:, None] * D, full_matrices=False)
    Phi = U / sqrt_w[:, None]             # undo the scaling on the modes
    if n_keep is not None:                # optional truncation
        Phi, sigma, Vt = Phi[:, :n_keep], sigma[:n_keep], Vt[:n_keep]
    return Phi, sigma, Vt.T               # .T so Psi has modes as columns


def w_inner(A, B):
    """All the W_s inner products between the columns of A and those of B.

    Entry (m, n) is sum_i w_i A[i, m] B[i, n], that is <a_m, b_n>_W.  Used for
    three things: checking orthonormality (should give the identity), computing
    the coefficients of a projection, and forming the reduced operators.
    """
    return A.T @ (w_s[:, None] * B)


Phi_pool, sigma_pool, Psi_pool = weighted_pod(D_all)
print("orthonormality of the pooled modes: "
      f"{np.abs(w_inner(Phi_pool[:, :20], Phi_pool[:, :20]) - np.eye(20)).max():.2e}")


# %% [markdown]
# ## The spectrum
#
# One POD per class, plus one on the two classes pooled together.  The gap
# between them is the whole question: does one basis serve both regimes?

# %%
def modes_for_energy(sigma, level=0.999):
    """How many modes are needed to carry a given fraction of the energy.

    The energy of mode r is sigma_r squared, so:

        sigma**2                 the energy of each mode
        np.cumsum(...)           running total: mode 1, modes 1-2, 1-3, ...
        / np.sum(sigma**2)       divided by the total, giving E_1, E_2, ...

    That array increases from something below one up to exactly one.
    np.searchsorted finds the first position where it reaches `level`, and the
    +1 turns a 0-based index into a count of modes.  A worked example: if the
    running fractions were [0.90, 0.98, 0.9995, 1.0] and level were 0.999, then
    searchsorted returns 2 (the first index at or above 0.999) and the answer
    is 3 modes.
    """
    energy = np.cumsum(sigma**2) / np.sum(sigma**2)   # E_1, E_2, ..., E_n = 1
    return int(np.searchsorted(energy, level)) + 1    # index -> number of modes


# One POD per class.  Each class is compressed ON ITS OWN, exactly as it would
# be if the two measurement campaigns had happened years apart and nobody had
# kept the raw data from the first one.  Merging them comes later.
per_class = {}                            # class name -> everything about it

for name, spec in CLASSES.items():        # "diffusive", then "convective"

    # The parameter pairs belonging to this class.  spec["nu"] is one number
    # and spec["alphas"] is a list of three, so this builds
    #     [(nu, 0.5), (nu, 1.0), (nu, 1.5)]
    mus = [(spec["nu"], a) for a in spec["alphas"]]

    # Snapshots from ONLY those three runs, side by side: about 540 columns.
    # Then a weighted POD of them, giving this class's own modes.
    D_c = snapshots_of(mus)                        # (n_s, ~540)
    Phi_c, sigma_c, Psi_c = weighted_pod(D_c)      # modes, energies, time coeffs

    # Keep everything later cells will want from this class:
    #   Phi    the spatial modes, columns, W_s-orthonormal
    #   sigma  their energies, decreasing
    #   Psi    the temporal structures, columns
    #   colour the plotting colour, carried along so figures stay consistent
    #   mus    which runs went into it, needed to slice Psi per run below
    per_class[name] = dict(Phi=Phi_c, sigma=sigma_c, Psi=Psi_c,
                           colour=spec["colour"], mus=mus)

    n99 = modes_for_energy(sigma_c)                # 99.9% of this class
    # Format: name left-aligned in 12 columns, sigma_1 in 8 columns with 3
    # decimals.  sigma_1 is the amplitude of the most energetic structure, and
    # it is NOT normalised, so the two classes can be compared directly.
    print(f"{name:<12} sigma_1 = {sigma_c[0]:8.3f}   "
          f"modes for 99.9% of the energy: {n99}")

# The same two numbers for the pooled dataset, which was compressed earlier as
# Phi_pool, sigma_pool.  Pooling is the alternative to merging: throw every
# snapshot from both classes into one SVD.  It is the best a linear basis can
# do on this data, and it is the yardstick the merged basis is measured against.
n99_pool = modes_for_energy(sigma_pool)
print(f"{'pooled':<12} sigma_1 = {sigma_pool[0]:8.3f}   "
      f"modes for 99.9% of the energy: {n99_pool}")

# Read the printed table before moving on.  The diffusive class needs 2 modes
# for 99.9%, the convective one 5: a lower nu means a steeper front and a
# steeper front needs more modes.  And sigma_1 is 9.7 against 4.2, so the
# diffusive snapshots are the larger ones in absolute terms.  That imbalance is
# why ranking the merged basis by POOLED energy favours the diffusive class,
# and why the global basis is used at its full size instead of being truncated.


# %% [markdown]
# ## Part B: the global basis
#
# Take the leading $q$ modes of **each** class and merge them.  Modes from
# different classes are not orthogonal to one another, so the stack has to be
# orthonormalised, and that step also removes whatever the two classes share.
#
# A second weighted SVD does both at once: it orthonormalises, and its singular
# values say how much genuinely new direction each class contributed.

# %%
# How many modes each class contributes to the merged basis.  Four is enough
# for 99.9% of the diffusive class and close to it for the convective one, and
# with two classes it makes a global basis of eight.  Raising it makes the
# global basis better AND more expensive, in exact proportion.
Q_PER_CLASS = 4


def global_basis(q=Q_PER_CLASS, tol=1e-10):
    """Merge the leading q modes of every class into one orthonormal basis.

    Two steps, and both are needed.

    1. **Orthonormalise the stack.**  Modes from different classes are not
       orthogonal to one another, and the classes share whatever structure the
       physics imposes on both, so the stack is rank deficient.  A weighted SVD
       of the stack removes the redundancy and returns an orthonormal spanning
       set.  Its singular values measure the overlap: a value near sqrt(2) is a
       direction both classes agree on, one near 1 a direction only one class
       has.

    2. **Rank the result by energy.**  Step 1 orders the directions by how much
       the classes agree, which has nothing to do with how much of the data
       they explain.  Truncating that order would throw away energetic
       structures.  So the pooled snapshots are expressed in the new space and
       a second, small POD is run there.  This is the same trick as the method
       of snapshots: the expensive SVD is replaced by a small one.
    """
    # ---- Step 1: put the two sets of modes side by side --------------------
    # d["Phi"][:, :q] is the leading q modes of one class.  np.hstack glues the
    # classes together left to right, so with two classes and q = 4 the stack
    # is (n_s, 8).  Each half is orthonormal, but the two halves are NOT
    # orthogonal to each other: both regimes solve the same equation, so they
    # share structure.
    stack = np.hstack([d["Phi"][:, :q] for d in per_class.values()])

    # A weighted SVD of the stack does two things at once.  It returns an
    # orthonormal spanning set, and its singular values measure the overlap:
    #   near sqrt(2)  both classes hold this direction
    #   near 1        only one class holds it
    #   near 0        two modes that nearly coincide; their difference, which
    #                 carries nothing, and which the tolerance below removes
    # (Exactly: sigma = sqrt(1 +/- cos theta) with theta the principal angles
    # between the two subspaces, so the values pair first-with-last.)
    U, s_overlap, _ = np.linalg.svd(sqrt_w[:, None] * stack, full_matrices=False)
    # This tolerance is NOT a compression knob.  It exists to catch columns that
    # are linearly dependent to machine precision, which would make Q rank
    # deficient and Phi^T W Phi singular.  With tol = 1e-10 the threshold here
    # is 1.4e-10, while the smallest overlap value in this tutorial is 0.0028,
    # so nothing is ever dropped and the global basis keeps all 2q columns.
    # Deciding how many modes to KEEP is step 2's job, by energy, not this one.
    keep = s_overlap > tol * s_overlap[0]
    Q = U[:, keep] / sqrt_w[:, None]                 # unweight: spans the merge

    # ---- Step 2: rank the merged space by energy ---------------------------
    # Q is orthonormal but ordered by OVERLAP, which says nothing about how
    # much data each direction explains.  So write every pooled snapshot in
    # the Q basis and run a small POD on those coefficients.
    Z = w_inner(Q, D_all)                 # (p, n_t): snapshot coords in Q
    Uz, s_energy, _ = np.linalg.svd(Z, full_matrices=False)
    # Q @ Uz rotates Q inside its own span.  A rotation of an orthonormal basis
    # is still orthonormal, so only the labels change: same space, new order.
    return Q @ Uz, s_overlap[keep], s_energy


Phi_glob, s_overlap, s_glob = global_basis()
print(f"stacked {len(per_class)} x {Q_PER_CLASS} = "
      f"{len(per_class) * Q_PER_CLASS} modes "
      f"-> {Phi_glob.shape[1]} kept "
      f"(tolerance {1e-10 * s_overlap[0]:.2e}, smallest overlap "
      f"{s_overlap[-1]:.4f}: nothing is dropped)")
print("orthonormality of the global basis: "
      f"{np.abs(w_inner(Phi_glob, Phi_glob) - np.eye(Phi_glob.shape[1])).max():.2e}")
print("overlap spectrum:", np.array2string(s_overlap, precision=3,
                                           max_line_width=100))

# %% [markdown]
# ### Figure 1 of 10: the spectra
#
# **Left.** $\sigma_r/\sigma_1$ against the mode index, log ordinate, one curve
# per class plus the pooled dataset and the global basis. The convective
# spectrum lies above the diffusive one everywhere: a lower $\nu$ means a
# steeper front, and a steeper front needs more modes.
#
# **Right.** The *neglected* energy $1-E_{n_r}$, log ordinate. Plotted this way
# and not as $E_{n_r}$, which saturates at one after three modes and shows
# nothing.
# %%
N_SHOW = 20
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

ax = axes[0]
for name, d in per_class.items():
    ax.semilogy(np.arange(1, N_SHOW + 1), d["sigma"][:N_SHOW] / d["sigma"][0],
                "o-", color=d["colour"], lw=1.6, ms=4, label=name)
ax.semilogy(np.arange(1, N_SHOW + 1), sigma_pool[:N_SHOW] / sigma_pool[0],
            "s--", color=INK, lw=1.6, ms=4, label="both classes pooled")
n_g = len(s_glob)
ax.semilogy(np.arange(1, n_g + 1), s_glob / sigma_pool[0], "^:",
            color=COLORS[3], lw=1.6, ms=5, label="global basis")
ax.axvline(n_g + 0.5, color=MUTED, ls="-", lw=1.0)
ax.text(n_g + 0.5, 1.0, f"  global basis stops at {n_g}", color=MUTED,
        fontsize=10, va="top")
ax.set_xlabel(r"mode index $r$")
ax.set_ylabel(r"$\sigma_r \, / \, \sigma_1$")
ax.set_xticks([1, 5, 10, 15, 20])
ax.legend(loc="lower left")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
# The neglected energy, 1 - E, on a log axis.  The cumulative energy itself
# saturates at one within a couple of modes and shows nothing.
def neglected(sigma):
    return 1.0 - np.cumsum(sigma**2) / np.sum(sigma**2)


for name, d in per_class.items():
    ax.semilogy(np.arange(1, N_SHOW + 1), neglected(d["sigma"])[:N_SHOW], "o-",
                color=d["colour"], lw=1.6, ms=4, label=name)
ax.semilogy(np.arange(1, N_SHOW + 1), neglected(sigma_pool)[:N_SHOW], "s--",
            color=INK, lw=1.6, ms=4, label="both classes pooled")
for level, style, txt in [(1e-2, ":", r"$99\%$"), (1e-3, "--", r"$99.9\%$")]:
    ax.axhline(level, color=MUTED, ls=style, lw=1.0)
    ax.text(N_SHOW, level, "  " + txt, color=MUTED, fontsize=10, va="center")
ax.set_xlabel(r"number of modes $n_r$")
ax.set_ylabel(r"neglected energy $1-E_{n_r}$")
ax.set_xticks([1, 5, 10, 15, 20])
ax.spines[["top", "right"]].set_visible(False)

finish(fig, "t3_singular_values.png")

# %% [markdown]
# ### Figure 2 of 10: the modes of each class
#
# $\phi_1$ to $\phi_4$ against $x$, one panel per class, same colours in both so
# a mode can be compared across regimes. The leading modes have the same
# character in both, compressed to the right in the convective case.
# %%
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharex=True, sharey=True)
for ax, (name, d) in zip(axes, per_class.items()):
    for r in range(4):
        ax.plot(x_fd, d["Phi"][:, r], lw=1.8, color=COLORS[r],
                label=rf"$\phi_{{{r+1}}}$")
    ax.set_title(name, fontsize=13)
    ax.set_xlabel(r"$x$")
    ax.set_xlim(0, 1)                     # axis tight in x
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"$\phi_r(x)$")
# The two panels share a y axis on purpose: the convective modes really are
# larger near the wall, and a per-panel scaling would hide that.  A margin is
# added so the deepest excursion is not clipped by the axis frame.
axes[0].margins(y=0.08)
axes[0].legend(ncol=2, loc="lower left")
finish(fig, "t3_modes_per_class.png")

# %% [markdown]
# ### Figure 3 of 10: the global modes against the sines of Tutorial 2
#
# The first four modes of the merged, energy-ranked basis in colour, over the
# first four sine modes in grey dashes. The sines spread uniformly over $[0,1]$;
# the POD modes put their structure between $x\approx0.6$ and $x=1$, where the
# source sits and the front forms. The data-adapted basis knows where the
# physics is.
# %%
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for r in range(4):
    ax.plot(x_fd, Phi_glob[:, r], lw=2.0, color=COLORS[r],
            label=rf"global $\phi_{{{r+1}}}$")
n_sine = 4
for r in range(n_sine):
    ax.plot(x_fd, np.sqrt(2) * np.sin((r + 1) * np.pi * x_fd), lw=1.0,
            ls="--", color=MUTED)
ax.plot([], [], lw=1.0, ls="--", color=MUTED,
        label=r"sines of Tutorial 2")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\phi_r(x)$")
ax.set_xlim(0, 1)
ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "t3_modes_global.png")


# %% [markdown]
# ## The temporal structures
#
# $\psi_r(t)$ says how each spatial structure is switched on.  For parametric
# data there is one curve per training run, and comparing them shows how the
# parameters act on the same structures.

# %% [markdown]
# ### Figure 4 of 10: the temporal structures
#
# $\psi_r(t)$ for $r=1,2,3$, colour by class and line style by $r$, first
# training run of each class only. $\psi_1$ rises and saturates, carrying the
# relaxation to steady state; the higher ones peak early and decay, carrying the
# transient.
# %%
fig, ax = plt.subplots(figsize=(7.6, 4.2))
styles = ["-", "--", ":"]                 # one line style per mode index r

for name, d in per_class.items():
    # Psi has one ROW per snapshot column of D_c, and D_c was built by stacking
    # three runs side by side.  So its rows read:
    #     rows 0 .. n-1        the first run  (alpha = 0.5)
    #     rows n .. 2n-1       the second     (alpha = 1.0)
    #     rows 2n .. 3n-1      the third      (alpha = 1.5)
    # Only the first run is drawn, to keep the figure readable, so the first
    # n_per_run rows are taken.  d["mus"][0] is that first run's parameter
    # pair, runs[...] is its (times, U), [1] is U and .shape[1] its column
    # count, which is exactly n_per_run.
    n_per_run = runs[d["mus"][0]][1].shape[1]      # stored instants in one run
    t_axis = runs[d["mus"][0]][0]                  # and the matching times

    for r in range(3):                    # the three leading temporal modes
        ax.plot(t_axis,
                d["Psi"][:n_per_run, r],  # mode r, first run only
                styles[r], lw=1.8, color=d["colour"],
                label=rf"{name}, $\psi_{{{r+1}}}$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\psi_r(t)$")             # short: the rest is in the caption
ax.set_xlim(0, T_END)
ax.axhline(0.0, color=MUTED, lw=0.8, zorder=0)   # the sign of a mode is
                                          #   arbitrary, so mark the zero
ax.legend(ncol=3, fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.01))
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "t3_temporal.png")


# %% [markdown]
# ## Part C: the reduced model
#
# Substituting $u=h+v$ into the finite-difference right-hand side, with
# $h=1-x$ so that $\partial_x h=-1$ and $\partial_{xx}h=0$:
#
# $$\dot v = \underbrace{h+\alpha f}_{\text{constant}}
#          + \underbrace{v+\nu\,\boldsymbol{D}_2 v - h\odot(\boldsymbol{D}_1 v)}_{\text{linear}}
#          + \underbrace{-\,v\odot(\boldsymbol{D}_1 v)}_{\text{quadratic}}$$
#
# This is the same three-term structure as Tutorial 2.  With $v=\boldsymbol{\Phi}\boldsymbol{a}$
# and the Galerkin condition $\boldsymbol{\Phi}^\top\boldsymbol{W}_s(\dot v - \text{rhs})=0$:
#
# $$\dot{\boldsymbol{a}} = \boldsymbol{c}_r(\alpha) + \boldsymbol{L}_r(\nu)\,\boldsymbol{a}
#                + \mathbf{C}_r:(\mathbf{a}\otimes\mathbf{a})$$
#
# and every one of those is **affine in the parameters**, so they are assembled
# once and evaluated for free at any $\boldsymbol{\mu}$:
#
# $$\boldsymbol{c}_r(\alpha)=\boldsymbol{g}_r+\alpha\,\boldsymbol{f}_r,\qquad
#   \boldsymbol{L}_r(\nu)=\boldsymbol{A}_r+\nu\,\boldsymbol{B}_r .$$

# %%
def reduce_operators(Phi):
    """Project the full-order operators onto the columns of Phi.

    This is the heart of the tutorial, and it is the same projection as
    Tutorial 2 with a different Phi.  Substituting v = Phi a into

        dv/dt = (h + alpha f) + (v + nu v_xx - h v_x) + (-v v_x)
                |___constant__|  |______linear______|   |_quadratic_|

    and testing against each mode gives one reduced object per group.  Every
    one of them is computed HERE, once, offline, and none depends on nu or on
    alpha: that is what makes answering a new parameter nearly free.

    Returned as a dict with these entries, for a basis of n_r modes:
        g  (n_r,)              <phi_m, h>,          the lifting term
        f  (n_r,)              <phi_m, f>,          the forcing shape
        A  (n_r, n_r)          the nu-independent part of the linear operator
        B  (n_r, n_r)          the part nu multiplies, i.e. diffusion
        C  (n_r, n_r, n_r)     the quadratic tensor
    """
    n_r = Phi.shape[1]                    # how many modes this basis has

    # Precompute the three things every product below needs.  Doing it once
    # here rather than inside each line is what keeps this readable.
    WPhi = w_s[:, None] * Phi             # W_s Phi: row i of Phi scaled by w_i
    dPhi = D1 @ Phi                       # phi_n'(x_i),  first derivative
    ddPhi = D2 @ Phi                      # phi_n''(x_i), second derivative

    # --- the constant term, split so that alpha factors out ----------------
    # WPhi.T @ vector computes sum_i w_i phi_m(x_i) vector(x_i) for every m,
    # which is exactly the inner product <phi_m, vector>.
    g_r = WPhi.T @ h_vec                  # <phi_m, h>: from the lifting
    f_r = WPhi.T @ f_vec                  # <phi_m, f>: from the source shape
    #  ==> the constant term at any alpha is  g_r + alpha * f_r

    # --- the linear operator, split so that nu factors out -----------------
    # A_r holds the two terms that do not involve nu: the +v from the lifting,
    # and the convection of v by h.  h_vec[:, None] * dPhi is h(x_i) phi_n'(x_i).
    A_r = WPhi.T @ (Phi - h_vec[:, None] * dPhi)   # <phi_m, phi_n - h phi_n'>
    B_r = WPhi.T @ ddPhi                           # <phi_m, phi_n''>: diffusion
    #  ==> the linear operator at any nu is  A_r + nu * B_r

    # --- the quadratic tensor ----------------------------------------------
    # C_r[m, n, k] = - sum_i w_i phi_m(x_i) phi_n(x_i) phi_k'(x_i).
    # The einsum subscripts read: i is the grid index, summed away because it
    # does not appear on the right; m, n, k survive.  This is n_r^3 numbers and
    # costs n_s n_r^3 to build, which is why it is built once and never again.
    C_r = -np.einsum("i,im,in,ik->mnk", w_s, Phi, Phi, dPhi, optimize=True)

    return dict(g=g_r, f=f_r, A=A_r, B=B_r, C=C_r, n_r=n_r)


def rom_terms(op, nu, alpha):
    """Assemble the reduced right-hand side at one parameter pair.

    This is where the affine structure pays off.  No integral is evaluated and
    the grid is never touched: a new operating point costs one scalar multiply
    and one addition per stored object.  If a parameter entered non-affinely,
    say the source position x_f, then f_r would have to be rebuilt over all
    n_s grid points at every query, and the independence from n_s would be
    lost.  Recovering it is what DEIM and ECSW are for.
    """
    const = op["g"] + alpha * op["f"]     # c_r(alpha), an (n_r,) vector
    L = op["A"] + nu * op["B"]            # L_r(nu), an (n_r, n_r) matrix
    # The quadratic term as a function of a.  "mnk,n,k->m" contracts the tensor
    # against a twice, giving component m = sum_{n,k} C[m,n,k] a_n a_k.
    quad = lambda a: np.einsum("mnk,n,k->m", op["C"], a, a, optimize=True)
    return const, L, quad


DT_ROM = 1e-2                 # time step of every reduced run; see run_rom


def run_rom(op, nu, alpha, dt=DT_ROM):
    """Integrate the reduced model from a(0) = 0 to T_END.

    Two things make this fast, and it is worth separating them.

    1. The state is n_r instead of n_s, so each step touches less data.
    2. The step itself can be thirty times longer.  The full-order run is
       explicit and pays dt < 0.4 dx^2 / nu, about 3e-4; the reduced run is
       semi-implicit on a system of size n_r, so its step is limited by
       accuracy alone and 1e-2 is enough.

    Measured at the end of this file, essentially ALL of the speed-up is the
    second one.  At n_r = 8 the first is buried under interpreter overhead.

    M is the identity because the modes are W_s-orthonormal, so no mass matrix
    has to be inverted; the identity is passed explicitly to keep the
    integrator general.  a(0) = 0 because v(x, 0) = u(x, 0) - h(x) = 0.
    """
    const, L, quad = rom_terms(op, nu, alpha)      # cheap, no integrals
    eye = np.eye(op["n_r"])                        # the reduced mass matrix
    return integrate_semi_implicit(eye, L, const, quad,
                                   np.zeros(op["n_r"]),   # a(0) = 0
                                   T_END, dt)


def reconstruct(Phi, A):
    """Rebuild the field from reduced coefficients: u = h + Phi a.

    A is (n_r, n_times), so Phi @ A is (n_s, n_times) and the lifting is added
    back to every column.  Forgetting the lifting is the single most common
    error in this tutorial: the result then satisfies u(0) = 0 instead of 1,
    and every error measure is meaningless.
    """
    return h_vec[:, None] + Phi @ A


# %%
offline_t0 = time.perf_counter()
OPS = {n_r: reduce_operators(Phi_glob[:, :n_r])
       for n_r in (2, 3, 4, 5, 6, 7, 8)}
cost_offline_assembly = time.perf_counter() - offline_t0
print(f"reduced operators assembled for {len(OPS)} sizes "
      f"in {cost_offline_assembly:.2f} s")


# %% [markdown]
# ## Part D: predict a parameter that was never simulated
#
# The training set is $\nu\in\{0.02,0.05\}$, $\alpha\in\{0.5,1.0,1.5\}$.
# The test point sits between the two classes, at a value of $\alpha$ that was
# never run.

# %%
# The test point.  nu = 0.032 sits between the two training classes (0.01 and
# 0.10) and alpha = 1.2 is not one of the three trained values, so this pair
# was never simulated.  Predicting it is the whole purpose of the exercise.
MU_TEST = (0.032, 1.2)

# The reference: one full-order run at the test point, used only to MEASURE the
# reduced model.  A real digital twin would not have it; that is the point.
t_ref, U_ref = run_fom(*MU_TEST, store_every=30)
u_ref_end = at_time(t_ref, U_ref, 0.5)    # the instant every error below uses


def l2_error(u, u_exact):
    """Relative L2 error, measured in the same weighted inner product.

    sqrt(sum_i w_i (u - u_exact)^2) / sqrt(sum_i w_i u_exact^2).  The weights
    make it an approximation of ||u - u_exact||_L2 / ||u_exact||_L2, and using
    the SAME weights as the POD is what makes the projection error below a
    genuine lower bound: measure in one norm and optimise in another and the
    bound no longer holds.
    """
    num = np.sqrt(np.sum(w_s * (u - u_exact) ** 2))     # weighted error norm
    den = np.sqrt(np.sum(w_s * u_exact ** 2))           # weighted reference norm
    return num / den


rom_runs = {}
for n_r, op in OPS.items():
    t_r, A_r = run_rom(op, *MU_TEST)
    rom_runs[n_r] = (t_r, A_r)
    u_r = reconstruct(Phi_glob[:, :n_r], A_r)
    print(f"n_r = {n_r:2d}   error at t = 0.5: "
          f"{l2_error(at_time(t_r, u_r, 0.5), u_ref_end):.3e}")

# %% [markdown]
# ### Figure 5 of 10: prediction at a parameter never simulated
#
# $u(x, t=0.5)$ at $(\nu,\alpha)=(0.032, 1.2)$. Thick dark: the full-order
# solution on 198 unknowns. Thin: POD-Galerkin at $n_r = 2, 4, 8$. Two modes
# miss the front, four place it, eight are indistinguishable at this scale.
# %%
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.plot(x_fd, u_ref_end, color=INK, lw=2.6,
        label=rf"full order, {N_X - 2} unknowns")
for k, n_r in enumerate((2, 4, 8)):
    t_r, A_r = rom_runs[n_r]
    u_r = reconstruct(Phi_glob[:, :n_r], A_r)
    ax.plot(x_fd, at_time(t_r, u_r, 0.5), lw=1.8, ls=["--", ":", "-."][k],
            color=COLORS[k], label=rf"POD-Galerkin, $n_r={n_r}$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(rf"$u(x,\, t=0.5)$")
ax.set_xlim(0, 1)
ax.margins(y=0.05)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "t3_rom_vs_fom.png")


# %% [markdown]
# ## Two errors, separated
#
# The reduced solution can be no better than the best the space can do.  That
# best is the **projection** of the full-order solution onto the basis, and it
# costs nothing to compute once the full-order solution is known.  The gap
# between the two curves is the error of the reduced **dynamics**, and it is
# the only part a better time integrator could remove.

# %% [markdown]
# ### Figure 7 of 10: the two errors, and what separates them
#
# Relative $L^2$ error at $t=0.5$ against $n_r$, log ordinate.
#
# * **dashed, projection.** Reconstruct with coefficients taken from the *known*
#   full-order solution, $a = \Phi^T W_s (u-h)$. No time integration enters, so
#   this is the best any method could do on this space. It is not a model.
# * **solid, POD-Galerkin.** Reconstruct with coefficients from integrating the
#   reduced system.
#
# The vertical gap between them is the cost of the dynamics. A better
# integrator moves the solid curve down to the dashed one and no further; a
# better basis moves both.
# %%
def projection_error(n_r):
    """The best any method could do on this space: the projection error.

    Take the reference solution, which is KNOWN here, and project it onto the
    first n_r modes.  No time integration is involved, so no solver can improve
    this number: it is a property of the basis alone and it is the floor under
    the POD-Galerkin error.  Comparing the two is the diagnostic of the whole
    tutorial.  If they are close, the space is the limit; if the model error is
    far above, the projected dynamics is.
    """
    Phi_r = Phi_glob[:, :n_r]
    V_ref = U_ref - h_vec[:, None]
    V_proj = Phi_r @ w_inner(Phi_r, V_ref)
    u_proj = h_vec[:, None] + V_proj
    return l2_error(at_time(t_ref, u_proj, 0.5), u_ref_end)


nr_list = sorted(OPS)
err_rom = []
err_proj = []
for n_r in nr_list:
    t_r, A_r = rom_runs[n_r]
    u_r = reconstruct(Phi_glob[:, :n_r], A_r)
    err_rom.append(l2_error(at_time(t_r, u_r, 0.5), u_ref_end))
    err_proj.append(projection_error(n_r))

fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.semilogy(nr_list, err_proj, "s--", color=MUTED, lw=1.8,
            label="projection error, the best the space can do")
ax.semilogy(nr_list, err_rom, "o-", color=COLORS[1], lw=2.0,
            label="POD-Galerkin error")
ax.set_xlabel(r"number of modes $n_r$")
ax.set_ylabel(r"relative $L^2$ error at $t=0.5$")
ax.set_xticks(nr_list)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "t3_two_errors.png")


# %% [markdown]
# ## Why the basis has to be global
#
# Train on one class only, then predict **both** classes.  Each basis is used
# at its natural size: a single class contributed $q$ modes, the two classes
# together contributed $2q$, so the global basis is twice as large and that is
# exactly what it costs.
#
# The projection error is reported next to the model error on purpose.  It is
# the error of the *space*, computed without solving anything, and no
# projection method on that space can do better.

# %%
MU_PROBE = {"diffusive point": (0.10, 1.2),
            "convective point": (0.01, 1.2)}

bases = {"diffusive class only": per_class["diffusive"]["Phi"][:, :Q_PER_CLASS],
         "convective class only": per_class["convective"]["Phi"][:, :Q_PER_CLASS],
         "global basis": Phi_glob}

probe = {}
for pname, mu in MU_PROBE.items():
    t_p, U_p = run_fom(*mu, store_every=30)
    u_p_end = at_time(t_p, U_p, 0.5)
    V_p = U_p - h_vec[:, None]
    for bname, Phi_b in bases.items():
        u_proj = h_vec[:, None] + Phi_b @ w_inner(Phi_b, V_p)
        t_b, A_b = run_rom(reduce_operators(Phi_b), *mu)
        u_b = reconstruct(Phi_b, A_b)
        probe[(pname, bname)] = (l2_error(at_time(t_p, u_proj, 0.5), u_p_end),
                                 l2_error(at_time(t_b, u_b, 0.5), u_p_end))

print("\nerror at t = 0.5, each basis at its natural size")
print(f"{'basis':<24}{'n_r':>5}" + "".join(f"{p:>34}" for p in MU_PROBE))
print(f"{'':<24}{'':>5}" + "".join(f"{'projection':>17}{'POD-Galerkin':>17}"
                                   for _ in MU_PROBE))
for bname, Phi_b in bases.items():
    row = f"{bname:<24}{Phi_b.shape[1]:>5d}"
    for pname in MU_PROBE:
        e_proj, e_rom = probe[(pname, bname)]
        row += f"{e_proj:>17.3e}{e_rom:>17.3e}"
    print(row)

# A single class is excellent at home and an order of magnitude worse abroad.
# The global basis is within a factor of a few of the best at both points, and
# it is the only one of the three that can be used across the whole range.

# %% [markdown]
# ### Figure 8 of 10: why the basis has to be global
#
# Two panels, one per probe point, sharing an ordinate. Three bases, each at its
# natural size (four modes for one class, eight for the merged), and for each the
# projection error in grey beside the model error in orange.
#
# Read across the panels, not down. Each single-class basis is excellent at home
# and one to two orders of magnitude worse abroad. The global basis is the only
# one acceptable in both panels: not the best anywhere, and the only one usable
# everywhere.
# %%
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True)
labels = list(bases)
xpos = np.arange(len(labels))
ticks = [f"diffusive\nclass only\n$n_r={bases[labels[0]].shape[1]}$",
         f"convective\nclass only\n$n_r={bases[labels[1]].shape[1]}$",
         f"global\nbasis\n$n_r={bases[labels[2]].shape[1]}$"]
for ax, (pname, mu) in zip(axes, MU_PROBE.items()):
    ax.bar(xpos - 0.19, [probe[(pname, b)][0] for b in labels], width=0.36,
           color=MUTED, label="projection error")
    ax.bar(xpos + 0.19, [probe[(pname, b)][1] for b in labels], width=0.36,
           color=COLORS[1], label="POD-Galerkin error")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(ticks)
    ax.set_title(rf"$\nu={mu[0]},\ \alpha={mu[1]}$", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"relative $L^2$ error")
axes[0].legend(loc="lower center", bbox_to_anchor=(1.0, 1.14), ncol=2)
finish(fig, "t3_global_vs_local.png")


# %% [markdown]
# ## Part E: what it costs
#
# The comparison that matters for a digital twin is one full-order run against
# one reduced run, at the same operating point, end to end.  The offline cost
# is paid once; the online cost is paid at every query.

# %%
def time_run(function, repeats=3):
    """Wall-clock cost of one call, taken as the best of a few repetitions.

    The BEST and not the mean: a timing is a true cost plus interference from
    the operating system, which can only ever add.  The minimum of several runs
    is therefore the least biased estimate of the cost itself.
    """
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()          # perf_counter, not time(): monotonic
        function()
        best = min(best, time.perf_counter() - t0)
    return best


cost_fom = time_run(lambda: run_fom(*MU_TEST, store_every=10**6))
cost_rom = {n_r: time_run(lambda op=op: run_rom(op, *MU_TEST))
            for n_r, op in OPS.items()}

print(f"\nfull order, {N_X - 2} unknowns: {cost_fom * 1e3:8.1f} ms")
for n_r in nr_list:
    print(f"  n_r = {n_r:2d}: {cost_rom[n_r] * 1e3:8.2f} ms   "
          f"speed-up {cost_fom / cost_rom[n_r]:6.1f}x   "
          f"error {err_rom[nr_list.index(n_r)]:.1e}")

cost_offline = cost_offline_runs + cost_offline_assembly
print(f"\noffline: {cost_offline:.2f} s "
      f"({len(runs)} training runs + the SVD + the reduced operators)")
print(f"break-even after "
      f"{cost_offline / max(cost_fom - cost_rom[8], 1e-12):.0f} queries "
      f"at n_r = 8")

# %% [markdown]
# ### Figure 9 of 10: what it costs
#
# **Left.** Wall clock for one run in ms, log ordinate: the full-order model,
# then one bar per $n_r$. **Right.** Twin axes against $n_r$: error in orange on
# the left, speed-up in dark on the right.
#
# The orange bars are all the same height, which raises the question the next
# figure answers. Note also where the speed-up really comes from: the full-order
# run takes 3168 explicit steps and the reduced one 100, a ratio of 31.7 against
# a measured speed-up of about 32. Almost none of it is the smaller state.
# %%
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))

ax = axes[0]
ax.bar([0], [cost_fom * 1e3], width=0.55, color=INK)
ax.bar(range(1, len(nr_list) + 1), [cost_rom[n] * 1e3 for n in nr_list],
       width=0.55, color=COLORS[1])
ax.set_yscale("log")
ax.set_xticks(range(len(nr_list) + 1))
ax.set_xticklabels([f"FOM\n{N_X - 2}"] + [f"{n}" for n in nr_list])
ax.set_xlabel(r"number of unknowns")
ax.set_ylabel(r"one run [ms]")
ax.spines[["top", "right"]].set_visible(False)

# The reduced runs all cost about the same, so an error-against-time scatter
# would pile them up in one place.  What varies with n_r is the error; what
# stays flat is the speed-up.  Both, on twin axes.
ax = axes[1]
ax.semilogy(nr_list, err_rom, "o-", color=COLORS[1], lw=2.0)
ax.set_xlabel(r"number of modes $n_r$")
ax.set_ylabel(r"relative $L^2$ error", color=COLORS[1])
ax.tick_params(axis="y", labelcolor=COLORS[1])
ax.set_xticks(nr_list)
ax.spines[["top"]].set_visible(False)

ax2 = ax.twinx()
ax2.plot(nr_list, [cost_fom / cost_rom[n] for n in nr_list], "s--",
         color=INK, lw=1.8)
ax2.set_ylabel(r"speed-up over the full-order run", color=INK)
ax2.tick_params(axis="y", labelcolor=INK)
ax2.set_ylim(0, 1.25 * max(cost_fom / cost_rom[n] for n in nr_list))
ax2.spines[["top"]].set_visible(False)

finish(fig, "t3_cost.png")


# %% [markdown]
# ## Where the online cost goes
#
# At these sizes the wall clock is dominated by the Python loop, not by the
# linear algebra, and the total time barely moves with $n_r$.  Timing the three
# terms of the right-hand side separately shows what would happen at a size
# where that is no longer true: the constant term is free, the linear term is
# $O(n_r^2)$, and the quadratic contraction is $O(n_r^3)$.
#
# That last exponent is the reason hyper-reduction exists.

# %%
NR_SCAN = [4, 8, 16, 24, 32, 48, 64, 96]
cost_lin, cost_quad = [], []
for n_r in NR_SCAN:
    Phi_r = Phi_pool[:, :n_r]                 # the pooled basis, only for timing
    op = reduce_operators(Phi_r)
    a = np.ones(n_r)
    L = op["A"] + MU_TEST[0] * op["B"]
    cost_lin.append(time_run(lambda L=L, a=a: L @ a, repeats=200))
    cost_quad.append(time_run(
        lambda op=op, a=a: np.einsum("mnk,n,k->m", op["C"], a, a, optimize=True),
        repeats=200))

print(f"\n{'n_r':>5}{'linear [us]':>14}{'quadratic [us]':>17}{'ratio':>9}")
for n_r, cl, cq in zip(NR_SCAN, cost_lin, cost_quad):
    print(f"{n_r:>5}{cl*1e6:>14.1f}{cq*1e6:>17.1f}{cq/cl:>9.1f}")

# %% [markdown]
# ### Figure 10 of 10: where the online time goes
#
# Time for one evaluation of a single term, microseconds, both axes log. The
# linear term $L_r a$ and the quadratic contraction $C_r:(a \otimes a)$, with
# $n_r^2$ and $n_r^3$ reference slopes anchored at the last point.
#
# Below $n_r \approx 30$ both are flat: that plateau is the cost of calling into
# numpy, not arithmetic, and it is why the previous figure showed no dependence
# on $n_r$. Above it the quadratic term takes off. That exponent is why
# hyper-reduction exists.
# %%
fig, ax = plt.subplots(figsize=(7.0, 4.0))
nr = np.array(NR_SCAN, dtype=float)
ax.loglog(nr, np.array(cost_lin) * 1e6, "o-", color=COLORS[0], lw=1.8,
          label=r"linear term, $\mathbf{L}_r \mathbf{a}$")
ax.loglog(nr, np.array(cost_quad) * 1e6, "s-", color=COLORS[1], lw=1.8,
          label=r"quadratic term, $\mathbf{C}_r:(\mathbf{a}\otimes\mathbf{a})$")
# Reference slopes anchored at the LAST point, where the interpreter overhead
# is no longer what is being measured.  Below n_r about 30 both curves are flat:
# that plateau is the cost of calling into numpy, not of the arithmetic.
ref2 = np.array(cost_lin)[-1] * 1e6 * (nr / nr[-1]) ** 2
ref3 = np.array(cost_quad)[-1] * 1e6 * (nr / nr[-1]) ** 3
ax.loglog(nr, ref2, ":", color=MUTED, lw=1.2, label=r"$n_r^2$")
ax.loglog(nr, ref3, "--", color=MUTED, lw=1.2, label=r"$n_r^3$")

slope = np.polyfit(np.log(nr[-3:]), np.log(np.array(cost_quad)[-3:]), 1)[0]
print(f"measured exponent of the quadratic term over the last three sizes: "
      f"{slope:.2f}")
ax.set_xlabel(r"$n_r$")
ax.set_ylabel(r"one evaluation [$\mu$s]")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
finish(fig, "t3_online_scaling.png")


# %% [markdown]
# ## The animation
#
# The full-order solution against the reduced one at three basis sizes, with
# the frames also written out one by one so that the slides can replay them.

# %% [markdown]
# ### Figure 6 of 10: the same prediction, in time
#
# The full-order solution against the reduced ones over the whole run, written
# both as a GIF and as numbered frames for the slides. Watch where the $n_r=2$
# curve separates: worst during the transient, recovering as the solution
# settles, which is Figure 4 seen from the other side.
# %%
import io

from PIL import Image


def make_animation(filename="t3_rom_vs_fom.gif", n_frames=50, fps=12,
                   framedir="t3_frames", nr_show=(2, 4, 8)):
    """Full-order solution against the reduced ones, over the whole run."""
    times = np.linspace(0.0, T_END, n_frames)

    ref = np.array([at_time(t_ref, U_ref, t) for t in times]).T
    red = {}
    for n_r in nr_show:
        t_r, A_r = rom_runs[n_r]
        u_r = reconstruct(Phi_glob[:, :n_r], A_r)
        red[n_r] = np.array([at_time(t_r, u_r, t) for t in times]).T

    lo = min(ref.min(), min(v.min() for v in red.values())) - 0.05
    hi = max(ref.max(), max(v.max() for v in red.values())) + 0.10

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    line_ref, = ax.plot([], [], color=INK, lw=2.6,
                        label=rf"full order, {N_X - 2} unknowns")
    lines = {}
    for k, n_r in enumerate(nr_show):
        lines[n_r], = ax.plot([], [], lw=1.8, ls=["--", ":", "-."][k],
                              color=COLORS[k], label=rf"POD-Galerkin, $n_r={n_r}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t)$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              fontsize=12)
    label = ax.text(0.03, 0.36, "", transform=ax.transAxes, fontsize=17,
                    va="top")

    def update(k):
        line_ref.set_data(x_fd, ref[:, k])
        for n_r in nr_show:
            lines[n_r].set_data(x_fd, red[n_r][:, k])
        label.set_text(rf"$t = {times[k]:.3f}$" + "\n"
                       + rf"$\nu = {MU_TEST[0]},\ \alpha = {MU_TEST[1]}$")

    fig.tight_layout()
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
            fig.savefig(outdir / f"t3_frame_{k:03d}.png", dpi=100,
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
# ## Questions
#
# 1. The projection error and the POD-Galerkin error are not the same curve.
#    Which one would a better time integrator improve, and which one would only
#    a better basis improve?
# 2. Look at the overlap spectrum printed above. Some values sit at 1.414,
#    some near 1, some near zero. What does each range say about the two
#    classes, and what would the value be with three classes instead of two?
# 3. At which $n_r$ does the speed-up stop growing, and what is the cost that
#    stops it? Count the operations in the quadratic term.
# 4. Predict $\nu=0.005$, below everything in the training set. Does the error
#    grow because of the space or because of the dynamics? Use the projection
#    error to decide.
# 5. The offline stage cost a few seconds here. Write down what it would cost
#    if one full-order run took an hour, and how many online queries would be
#    needed before the reduced model pays for itself.

# %% [markdown]
# ## Answers
#
# **1. Which error does a better integrator improve?**  Only the POD-Galerkin
# curve, and only the part of it above the projection error.  The projection
# error is computed from the *known* full-order solution and contains no time
# integration at all, so no solver can move it; a larger or better basis moves
# both, because the projection error is the floor.  At the test point with
# $n_r=8$ the two are $9.7\cdot10^{-4}$ and $3.3\cdot10^{-3}$, a factor 3.4.
# At the diffusive probe the same basis gives $9.3\cdot10^{-6}$ against
# $1.6\cdot10^{-3}$, a factor 170: inside the training set it is the *dynamics*
# that limits the model, not the space.
#
# **2. The overlap spectrum.**  A direction every class contains appears $n_c$
# times in the stack, so its singular value is $\sqrt{n_c}$; a direction only
# one class has gives 1; two nearly identical modes give one large value and one
# near zero, the small one belonging to their difference and carrying nothing.
# Measured here, with two classes and four modes each:
# $(1.414, 1.414, 1.359, 1.202, 0.745, 0.392, 0.023, 0.003)$ — two directions
# shared exactly, two nearly so, two belonging to one class, two duplicates.
# With three classes the shared value would be $\sqrt3\approx1.732$.
#
# **3. Where does the speed-up stop growing?**  It never grows here: it is
# between 28 and 38 for every $n_r$ from 2 to 8.  The wall clock is spent in the
# Python loop, not in arithmetic — at $n_r=8$ the quadratic contraction is 512
# operations and takes 44 microseconds, nearly all of it the cost of entering
# `einsum`.  The scaling test shows both terms flat below $n_r\approx30$ and the
# quadratic one rising after, reaching 255 microseconds at $n_r=96$.
# Extrapolating, one reduced step would match one full-order run near
# $n_r\approx200$, which is $n_s$: there the reduced model has no reason to
# exist.
#
# **Where the speed-up actually comes from.**  Count steps, not unknowns.  The
# full-order run takes 3168 explicit steps at $\Delta t=3.16\cdot10^{-4}$; the
# reduced run takes 100 at $10^{-2}$.  That ratio alone is 31.7, and the
# measured speed-up is about 32.  Essentially all of it is the longer step, not
# the smaller state — the step can grow because the stiff operator is now an
# $n_r\times n_r$ matrix that is factorised once and treated implicitly.
#
# **4. Extrapolating to $\nu=0.005$.**  The space, not the dynamics.  At
# $n_r=8$: $\nu=0.032$ gives $9.7\cdot10^{-4}$ and $3.3\cdot10^{-3}$ (ratio
# 3.4), $\nu=0.01$ gives $6.8\cdot10^{-3}$ and $2.0\cdot10^{-2}$ (2.9), and
# $\nu=0.005$ gives $1.6\cdot10^{-2}$ and $2.7\cdot10^{-2}$ (1.6).  The
# projection error grows by a factor 17 while the ratio *falls*: the dynamics is
# not degrading, the basis simply has no front that steep in it.  That is the
# diagnostic — if both grow and the ratio holds, suspect the dynamics; if the
# projection error grows and the ratio shrinks, only new snapshots will help.
#
# **5. Offline economics.**  Here: 2.3 s offline, break-even after 11 queries.
# Not 6, because the six training runs are not equally priced — the run at
# $\nu=0.10$ needs 9900 explicit steps against 990 at $\nu=0.01$, so the bill is
# set by the most expensive corner of the parameter box.  Scaled to a one-hour
# full-order run and the same speed-up, the offline stage costs about 6 hours,
# one query costs 113 s, and break-even arrives after about 6 queries.  In
# general $N_{\rm break-even}\approx N_{\rm train}/(1-1/S)$, which for a large
# speed-up is just $N_{\rm train}$: the model pays for itself after roughly as
# many queries as it took runs to build.
