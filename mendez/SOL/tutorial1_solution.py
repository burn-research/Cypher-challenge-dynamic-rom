# %% [markdown]
# # Tutorial 1: approximating a function
#
# One equation, four settings.  On $\Omega=[-1,1]$ we approximate a target
# $u(x)$ as
#
# $$u(x)\;\approx\;u_{n_b}(x)=\sum_{n=1}^{n_b} c_n\,\phi_n(x)$$
#
# by making the error orthogonal to every basis function,
#
# $$\langle \phi_m, e\rangle = 0
# \qquad\Longleftrightarrow\qquad
# \boldsymbol{A}\boldsymbol{c} = \boldsymbol{b},
# \qquad A_{mn}=\langle\phi_m,\phi_n\rangle,\quad b_m=\langle\phi_m,u\rangle .$$
#
# | | basis | weight | sampling |
# |---|---|---|---|
# | 1 | monomials $x^{n}$ | 1 | uniform grid |
# | 2 | Fourier | 1 | uniform grid |
# | 3 | Chebyshev $T_n$ | $(1-x^2)^{-1/2}$ | uniform grid |
# | 4 | Chebyshev $T_n$ | Gauss-Lobatto | CGL points |
#
# Settings 3 and 4 use the **same basis** and differ only in where it is
# sampled.  This file is self-contained: nothing to install beyond numpy and
# matplotlib and scipy, and nothing to upload when running it on Colab.

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
from pathlib import Path

# ---- figure style, once, so every plot below matches the slides -----------
plt.rc('text', usetex=False)              # mathtext, not a LaTeX install
plt.rc('font', family='serif')            # serif, to match the deck
plt.rc('xtick', labelsize=12)             # tick labels readable from the back
plt.rc('ytick', labelsize=12)             #   of a lecture room
plt.rc('axes', labelsize=13, titlesize=13, grid=False)   # no grid, by choice
plt.rc('legend', fontsize=11, frameon=False)             # legend, no box
plt.rc('mathtext', fontset='cm')          # Computer Modern maths in labels
plt.rc('figure', dpi=140)                 # on-screen size; saved files use
                                          #   their own dpi

INK, MUTED = "#1F2933", "#4B5563"         # near-black for data, grey for notes
COLORS = ["#2E6FB7", "#D2691E", "#8E44AD", "#1E9E8A"]   # blue, orange, purple,
                                          #   teal: the four series colours
# The four settings compared throughout, in one fixed order.  SETTINGS holds
# the keys the code passes around, SHORT the labels for legends, and TICKS the
# same labels broken over two lines so they do not collide on an x axis.
SETTINGS = ["monomials", "fourier", "chebyshev_uniform", "chebyshev_cgl"]
SHORT = ["monomials", "Fourier", "Chebyshev, uniform", "Chebyshev, CGL"]
TICKS = ["monomials", "Fourier", "Chebyshev\nuniform", "Chebyshev\nCGL"]
# Settings 3 and 4 are the SAME basis on DIFFERENT points, which is the whole
# experiment: everything that differs between them comes from the sampling.

# Figures are written next to the slides when the folder exists.
FIGDIR = Path.cwd()
for candidate in [Path("../../figures"), Path("../figures"), Path("figures")]:
    if candidate.is_dir():
        FIGDIR = candidate
        break


def save(fig, name):
    """Tighten the layout, then write the file.

    tight_layout is called here so that the figure shown in the notebook and
    the file written to disk have the same, correctly fitted, layout.
    """
    fig.tight_layout()
    fig.savefig(FIGDIR / name, bbox_inches="tight")
    print("wrote", (FIGDIR / name).resolve())


# %% [markdown]
# ## The two targets
#
# One smooth, one with a kink at the origin.  They are chosen so that the
# *rate* of convergence differs, not just the constant.

# %%
def u_smooth(x):
    """The smooth target: infinitely differentiable everywhere on [-1, 1].

    A linear ramp plus a Gaussian-damped oscillation.  It is smooth but NOT
    periodic: u(-1) = -1 and u(1) = 1 differ, which is what makes the Fourier
    basis stall on it later.  Nothing else in the file depends on the formula.
    """
    return x + np.exp(-3.0 * x**2) * np.sin(2.0 * np.pi * x)


def u_kink(x):
    """The hard target: continuous, but its first derivative jumps at x = 0.

    One missing derivative is enough to slow every basis of smooth global
    functions to the same algebraic rate.  That is the point of having it.
    """
    return np.abs(x)


# %% [markdown]
# ## The four bases
#
# Each basis function is a column of `Phi`, so `Phi[j, n]` is $\phi_n(x_j)$.

# %%
def basis_monomials(x, n_b):
    """phi_n(x) = x^n, for n = 0 .. n_b-1.  The obvious choice, and a bad one.

    Returns Phi of shape (len(x), n_b) with Phi[j, n] = phi_n(x_j).  As n grows
    these become nearly indistinguishable from one another on [-1, 1], which is
    exactly what makes the Gram matrix ill-conditioned.
    """
    Phi = np.zeros((len(x), n_b))         # one column per basis function
    for n in range(n_b):
        Phi[:, n] = x**n                  # column n is x to the power n
    return Phi


def basis_fourier(x, n_b):
    """1, cos(pi x), sin(pi x), cos(2 pi x), sin(2 pi x), ...

    Period 2, which matches the length of [-1, 1].  The columns are filled in
    pairs, one cosine and one sine per wavenumber k, starting from the constant.
    The loop is written this way so that n_b can be even or odd: an odd n_b
    simply stops after a cosine.
    """
    Phi = np.zeros((len(x), n_b))
    Phi[:, 0] = 1.0                       # the constant, always column 0
    k, n = 1, 1                           # k: wavenumber.  n: column to fill
    while n < n_b:
        Phi[:, n] = np.cos(k * np.pi * x)
        n += 1
        if n < n_b:                       # guard: n_b may be even
            Phi[:, n] = np.sin(k * np.pi * x)
            n += 1
        k += 1                            # next wavenumber
    return Phi


def basis_chebyshev(x, n_b):
    """Chebyshev polynomials, built by their recurrence rather than by formula.

        T_0 = 1,  T_1 = x,  T_{n+1} = 2 x T_n - T_{n-1}

    The recurrence is used because it is numerically stable and costs one
    multiply-add per point per mode.  Evaluating cos(n arccos x) directly is
    equivalent in exact arithmetic and worse in floating point.

    Note that T_n spans the SAME space as the monomials of the same degree.
    Everything that differs between settings 1 and 3 is conditioning, not
    approximation power, and that is one of the lessons of this tutorial.
    """
    Phi = np.zeros((len(x), n_b))
    Phi[:, 0] = 1.0                       # T_0
    if n_b > 1:
        Phi[:, 1] = x                     # T_1
    for n in range(1, n_b - 1):           # then each one from its two before
        Phi[:, n + 1] = 2.0 * x * Phi[:, n] - Phi[:, n - 1]
    return Phi


def sample_basis(name, x, n_b):
    """Dispatch on the setting name.  Settings 3 and 4 share a basis, and only
    the points differ, which is why both fall through to the Chebyshev case."""
    if name == "monomials":
        return basis_monomials(x, n_b)
    if name == "fourier":
        return basis_fourier(x, n_b)
    return basis_chebyshev(x, n_b)        # both Chebyshev settings land here


# %% [markdown]
# ## Grids and quadrature rules
#
# The **Chebyshev-Gauss-Lobatto** points are the extrema of $T_N$ together with
# the two endpoints,
#
# $$x_k=\cos\left(\frac{k\pi}{N}\right),\qquad k=0,\dots,N,$$
#
# so they cluster near $x=\pm1$.  Their weights are $\pi/N$ inside and **half**
# that at the two ends; dropping the halving destroys the discrete
# orthogonality.

# %%
def grid_uniform(n_x):
    """Equally spaced points on [-1, 1] with trapezoidal weights.

    Returns (x, w) with sum_j w_j g(x_j) approximating the integral of g.  The
    weights are what turn a dot product into an approximate L2 inner product;
    without them the Gram matrix is not an approximation of anything.
    """
    x = np.linspace(-1.0, 1.0, n_x)       # endpoints included
    dx = x[1] - x[0]                      # spacing, 2/(n_x - 1)
    w = np.full(n_x, dx)                  # a full dx for interior points ...
    w[0] = w[-1] = dx / 2.0               # ... half of one at the two ends
    return x, w


def grid_cgl(n_points):
    """Chebyshev-Gauss-Lobatto points and their weights.

        x_k = cos(k pi / N),   k = 0 .. N,   N = n_points - 1

    These are the extrema of T_N plus the two endpoints, so they crowd towards
    x = +/-1.  With the weight w(x) = 1/sqrt(1-x^2) already folded in, the
    quadrature integrates T_m T_n w EXACTLY, which is why setting 4 is
    orthogonal to machine precision and setting 3 is not.

    The endpoint weights are HALF the interior ones.  Dropping that halving is
    the classic bug here: the discrete orthogonality is destroyed and the whole
    point of the setting is lost.

    Careful: taking n_points equal to n_b makes the system square and the
    projection silently becomes interpolation, with a measured error of 1e-16
    at the sample points and nothing at all in between.
    """
    N = n_points - 1                      # so there are N+1 points in total
    k = np.arange(N + 1)                  # 0, 1, ..., N
    x = np.cos(k * np.pi / N)             # decreasing, from +1 down to -1
    w = np.full(N + 1, np.pi / N)         # interior weight ...
    w[0] = w[-1] = np.pi / (2.0 * N)      # ... and half of it at the ends
    return x[::-1], w[::-1]               # reversed together, so x increases


def weight_chebyshev(x):
    """The Chebyshev weight w(x) = 1/sqrt(1-x^2).

    Chebyshev polynomials are orthogonal with respect to THIS weight, not with
    respect to plain integration.  It blows up at x = +/-1, which is why the
    two endpoints have to be dropped in setting 3 below.
    """
    return 1.0 / np.sqrt(1.0 - x**2)


def setting(name, n_b, n_x=401):
    """Assemble one of the four settings: (points, weights, basis, label).

    The four differ in exactly two ways, the basis and the points, and the
    whole tutorial is about separating those two effects:

      1 monomials         uniform points, no weight
      2 Fourier           uniform points, no weight
      3 Chebyshev uniform uniform points, Chebyshev weight
      4 Chebyshev CGL     CGL points, weight folded into them

    Settings 3 and 4 use the SAME basis functions, so they span the same space
    and give the same best approximation.  Everything that differs between them
    comes from where the points sit.
    """
    if name == "monomials":
        x, w = grid_uniform(n_x)
        return x, w, basis_monomials(x, n_b), SHORT[0]
    if name == "fourier":
        x, w = grid_uniform(n_x)
        return x, w, basis_fourier(x, n_b), SHORT[1]
    if name == "chebyshev_uniform":
        x, w = grid_uniform(n_x)
        # The Chebyshev weight is infinite at x = +/-1, so those two points
        # cannot be used.  Dropping them and multiplying the trapezoidal
        # weights by w(x) gives a quadrature that is CORRECT but not exact for
        # products of Chebyshev polynomials, which is the whole story of this
        # setting: right idea, wrong points.
        x, w = x[1:-1], w[1:-1]
        w = w * weight_chebyshev(x)
        return x, w, basis_chebyshev(x, n_b), SHORT[2]
    if name == "chebyshev_cgl":
        # Same basis, same number of points as setting 3, different placement.
        # Using the same count matters: otherwise the comparison would confound
        # the placement with the amount of data.
        x, w = grid_cgl(n_x)
        return x, w, basis_chebyshev(x, n_b), SHORT[3]
    raise ValueError("unknown setting: " + name)


# %% [markdown]
# ### Where the points sit
#
# Uniform points are equally spaced; CGL points crowd towards the ends.  That
# clustering is what removes the Runge phenomenon and what makes the discrete
# orthogonality exact.

# %%
n_show = 10
x_u, _ = grid_uniform(n_show)
x_c, _ = grid_cgl(n_show)

fig, ax = plt.subplots(figsize=(7.0, 3.4))

# The construction: CGL points are the cosines of equispaced angles.
theta_arc = np.linspace(0, np.pi, 400)          # a smooth half circle
ax.plot(np.cos(theta_arc), np.sin(theta_arc), color=MUTED, lw=1.0)

theta = np.linspace(0, np.pi, n_show)           # the equispaced angles
for t in theta:
    ax.plot([np.cos(t), np.cos(t)], [0.05, np.sin(t)], color=MUTED,
            lw=0.6, ls=":")
ax.plot(np.cos(theta), np.sin(theta), "o", ms=4.5, color=COLORS[0])

ax.plot(x_c, np.zeros_like(x_c), "o", ms=7, color=COLORS[3],
        label="Chebyshev-Gauss-Lobatto")
ax.plot(x_u, -0.22 * np.ones_like(x_u), "s", ms=6, color=COLORS[1],
        label="uniform")

# Equal aspect, so that the half circle is actually round.
ax.set_aspect("equal")
ax.set_xlim(-1.12, 1.12)
ax.set_ylim(-0.38, 1.18)
ax.set_xlabel(r"$x$")
ax.set_yticks([])
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_title(r"$x_k=\cos(k\pi/N)$: equispaced angles, projected down",
             fontsize=12)
ax.annotate("crowding", xy=(-0.98, 0.04), xytext=(-0.55, 0.30),
            fontsize=10, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
save(fig, "t1_cgl_points.png")
plt.show()

# %% [markdown]
# ## The projection
#
# Three short functions: the Gram matrix, the right-hand side, and a Cholesky
# solve.  The loops are written out rather than vectorised, because they are the
# definition.

# %%
def gram_matrix(Phi, w):
    """The Gram matrix of the normal equations: A[m, n] = <phi_m, phi_n>.

    Written as the explicit quadrature sum

        A[m, n] = sum_j w_j phi_m(x_j) phi_n(x_j)

    which is the discrete version of the integral of phi_m phi_n.  The double
    loop is deliberately literal: it could be written Phi.T @ (w[:, None] * Phi)
    in one line, and is in the later tutorials, but here the point is to see
    the definition.

    A is symmetric by construction, and positive definite as long as the basis
    functions are linearly independent ON THESE POINTS.  When it stops being
    so, Cholesky fails, and that failure is one of the results of the tutorial.
    """
    n_b = Phi.shape[1]                    # number of basis functions
    A = np.zeros((n_b, n_b))
    for m in range(n_b):
        for n in range(n_b):
            # one weighted inner product per entry
            A[m, n] = np.sum(w * Phi[:, m] * Phi[:, n])
    return A


def rhs_vector(Phi, w, u_samples):
    """The right-hand side of the normal equations: b[m] = <phi_m, u>.

        b[m] = sum_j w_j phi_m(x_j) u(x_j)

    Same quadrature as the Gram matrix, with u in place of one basis function.
    u_samples is u already evaluated at the points, not the function itself.
    """
    n_b = Phi.shape[1]
    b = np.zeros(n_b)
    for m in range(n_b):
        b[m] = np.sum(w * Phi[:, m] * u_samples)
    return b


def solve_cholesky(A, b):
    """Solve A c = b for a symmetric positive definite A.

    A = L L^T, then two triangular solves:

        L y = b     forward substitution,   O(n^2)
        L^T c = y   back substitution,      O(n^2)

    Both sweeps use solve_triangular, which exploits the triangular structure.
    A general solver would run a full LU on a matrix that is already
    triangular, costing O(n^3) and discarding the work Cholesky just did.

    Cholesky fails exactly when A stops being numerically positive definite,
    which is the signal that the basis has become dependent.  In that case we
    fall back to least squares, which is itself an informative outcome.
    """
    try:
        L = np.linalg.cholesky(A)                  # A = L L^T, the O(n^3) step
        y = solve_triangular(L, b, lower=True)     # L y = b,   forward, O(n^2)
        c = solve_triangular(L.T, y, lower=False)  # L^T c = y, back,    O(n^2)
        return c, True                             # True: A was still SPD
    except np.linalg.LinAlgError:
        # Cholesky refused, so A is no longer numerically positive definite.
        # Least squares still returns something usable, and the False flag is
        # recorded so the figures can mark where this happened.
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        return c, False


def project(name, u_function, n_b, n_x=401):
    """The whole pipeline for one setting: sample, assemble, solve, measure.

    Returns a dict holding everything the figures below need, so that a setting
    is computed once and looked at from several angles.
    """
    # 1. Sample.  Points, weights and the basis evaluated at those points.
    x, w, Phi, label = setting(name, n_b, n_x)
    u_samples = u_function(x)             # the target, at the same points

    # 2. Assemble the normal equations A c = b.
    A = gram_matrix(Phi, w)
    b = rhs_vector(Phi, w, u_samples)

    # 3. Solve.  ok is False if A had lost positive definiteness.
    c, ok = solve_cholesky(A, b)

    # 4. Measure, on a DIFFERENT grid.  This matters: if the error were
    #    measured at the same points used to build the system, a fit that
    #    happens to interpolate would score 1e-16 and look perfect while being
    #    useless in between.  2001 uniform points is far finer than any setting.
    x_fine, w_fine = grid_uniform(2001)
    u_fine = u_function(x_fine)                        # the truth, finely
    u_fine_approx = sample_basis(name, x_fine, n_b) @ c   # the fit, finely

    # Weighted L2 norms of the error and of the target, so the ratio below is
    # a relative error and settings with different amplitudes are comparable.
    err = np.sqrt(np.sum(w_fine * (u_fine - u_fine_approx) ** 2))
    nrm = np.sqrt(np.sum(w_fine * u_fine ** 2))

    return dict(x=x, w=w, Phi=Phi, label=label, A=A, b=b, c=c, cholesky_ok=ok,
                u=u_samples, error=err, relative_error=err / nrm,
                condition_number=np.linalg.cond(A))


# %% [markdown]
# ## Task 1: look at the bases
#
# Before running: which of them do you expect to be orthogonal, and with
# respect to which inner product?

# %%
x = np.linspace(-1, 1, 800)
n_basis_show = 8
fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), sharex=True, sharey=True)

for ax, name, title in zip(axes.ravel(), SETTINGS, SHORT):
    Phi = sample_basis(name, x, n_basis_show)
    for n in range(n_basis_show):
        ax.plot(x, Phi[:, n], lw=1.3,
                color=plt.cm.Blues(0.25 + 0.65 * n / (n_basis_show - 1)))
    ax.set_title(title)
    ax.set_ylim(-1.7, 1.7)
    ax.axhline(0, color="#B8C2CE", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

for ax in axes[1, :]:
    ax.set_xlabel(r"$x$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\phi_n(x)$")
save(fig, "t1_bases.png")
plt.show()

# %% [markdown]
# ## Task 2: orthogonality, twice
#
# A basis can be orthogonal as an integral and not as a sum.  That is the whole
# point of settings 3 and 4.

# %%
def off_diagonal_ratio(A):
    """How far A is from diagonal, as a single number between 0 and about 1.

        max |A[m, n]| over m != n,  divided by  max |A[m, n]| over everything

    Zero means a perfectly orthogonal basis.  The division by the largest entry
    makes it scale-free, so bases with wildly different amplitudes can be put
    in the same table.
    """
    D = np.diag(np.diag(A))               # A with its off-diagonal set to zero
    return np.max(np.abs(A - D)) / np.max(np.abs(A))


n_b = 8
ratios_cont, ratios_disc = [], []
print(f"{'setting':26s} {'continuous':>12s} {'discrete':>12s}")
for name, short in zip(SETTINGS, SHORT):
    _, w_f, Phi_f, _ = setting(name, n_b, n_x=4001)
    _, w_d, Phi_d, _ = setting(name, n_b, n_x=64)
    rc = off_diagonal_ratio(gram_matrix(Phi_f, w_f))
    rd = off_diagonal_ratio(gram_matrix(Phi_d, w_d))
    ratios_cont.append(rc)
    ratios_disc.append(rd)
    print(f"{short:26s} {rc:12.2e} {rd:12.2e}")

# %%
fig, ax = plt.subplots(figsize=(7.6, 3.6))
pos = np.arange(4)
ax.bar(pos - 0.19, np.maximum(ratios_cont, 1e-17), 0.34, color=COLORS,
       alpha=0.5, label="continuous")
ax.bar(pos + 0.19, np.maximum(ratios_disc, 1e-17), 0.34, color=COLORS,
       label="discrete, 64 points")
ax.set_yscale("log")
ax.set_xticks(pos)
ax.set_xticklabels(TICKS, fontsize=10)
ax.set_ylabel(r"$\max_{m \neq n} |A_{mn}| \, / \, \max_{mn} |A_{mn}|$")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "t1_orthogonality.png")
plt.show()

# %% [markdown]
# ## Task 3: show the Gram matrix
#
# On a linear colour scale a small off-diagonal entry is invisible, so use a
# logarithmic one.  Dark means large.
n_b, n_x = 8, 64

fig = plt.figure(figsize=(7.8, 6.4))
gs = fig.add_gridspec(
    2, 3,
    width_ratios=[1, 1, 0.08],
    wspace=0.45,
    hspace=0.45
)

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
]
cax = fig.add_subplot(gs[:, 2])   # colorbar spans both rows

for ax, name, short, col in zip(axes, SETTINGS, SHORT, COLORS):
    _, w, Phi, _ = setting(name, n_b, n_x)
    A = gram_matrix(Phi, w)
    A = np.log10(np.maximum(np.abs(A) / np.max(np.abs(A)), 1e-18))

    im = ax.imshow(A, cmap="Blues", vmin=-16, vmax=0)

    ax.set_title(short, fontsize=11, color=col)
    ax.set_xticks([0, n_b - 1])
    ax.set_yticks([0, n_b - 1])
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$m$")

cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$\log_{10} |A_{mn}| / \max |A_{mn}|$", fontsize=12)

save(fig, "t1_gram.png")
plt.show()

# %% [markdown]
# ## Task 4: conditioning
#
# $\mathrm{cond}(A)$ says how much round-off in $b$ is amplified in $c$.  Watch
# where the monomial curve crosses $10^{16}$.

# %%
from matplotlib.ticker import MaxNLocator

n_list = np.arange(2, 21)
fig, ax = plt.subplots(figsize=(6.8, 4.0))

for name, short, col in zip(SETTINGS, SHORT, COLORS):
    conds = []
    for nb in n_list:
        _, w, Phi, _ = setting(name, nb)
        conds.append(np.linalg.cond(gram_matrix(Phi, w)))
    ax.semilogy(n_list, conds, "o-", ms=4, lw=1.7, color=col, label=short)

ax.axhline(1 / np.finfo(float).eps, color=MUTED, ls="--", lw=1,
           label="double precision limit")

ax.set_xlabel(r"number of basis functions  $n_b$")
ax.set_ylabel(r"$\mathrm{cond}(A)$")
ax.set_ylim(1, 3e17)

# Integer ticks only on the x-axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.legend(loc="center left")
ax.spines[["top", "right"]].set_visible(False)

save(fig, "t1_conditioning.png")
plt.show()

# %% [markdown]
# ## Task 5: reconstruct, and measure the error

# %%
TARGET_LABEL = {
    "smooth": r"$u(x) = x + e^{-3x^2}\,\sin(2\pi x)$",
    "kink": r"$u(x) = |x|$",
}

for u_function, tag in [(u_smooth, "smooth"), (u_kink, "kink")]:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), sharex=True, sharey=True)
    x_plot = np.linspace(-1, 1, 800)

    for ax, name, short in zip(axes.ravel(), SETTINGS, SHORT):
        ax.plot(x_plot, u_function(x_plot), color=INK, lw=2.2, ls="--",label="exact")
        for nb, shade in zip([4, 8, 16], [0.4, 0.65, 0.95]):
            r = project(name, u_function, nb)
            ax.plot(x_plot, sample_basis(name, x_plot, nb) @ r["c"], lw=1.5,
                    color=plt.cm.Oranges(shade), label=rf"$n_b = {nb}$")
        ax.set_title(short)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].legend(loc="lower right")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$x$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$u(x)$")
    fig.suptitle(TARGET_LABEL[tag], fontsize=14)
    save(fig, f"t1_reconstruction_{tag}.png")
    plt.show()

# %%
n_list = np.arange(2, 27)
styles = [dict(ls="-", lw=3.4, alpha=0.5), dict(ls="-", lw=1.9),
          dict(ls="--", lw=1.9), dict(ls=":", lw=2.4)]

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True)
for ax, u_function, tag in zip(axes, [u_smooth, u_kink], ["smooth", "kink"]):
    for name, short, col, st in zip(SETTINGS, SHORT, COLORS, styles):
        errs = [project(name, u_function, nb)["relative_error"]
                for nb in n_list]
        ax.semilogy(n_list, errs, color=col, label=short, **st)
    ax.set_xlabel(r"number of basis functions  $n_b$")
    ax.set_title(TARGET_LABEL[tag], fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel(r"$\|u - u_{n_b}\|_{L^2} \, / \, \|u\|_{L^2}$")
axes[0].legend(loc="lower left")
save(fig, "t1_error.png")
plt.show()

# %% [markdown]
# ## Questions
#
# **On orthogonality.** Which families are orthogonal in the continuous sense?
# Which stay orthogonal once the integral becomes a sum?  Settings 3 and 4 use
# the same basis: where exactly does the difference come from?
#
# **On conditioning and accuracy.** At which $n_b$ does the monomial system
# become useless in double precision?  Does the *approximation* degrade at the
# same $n_b$, or noticeably later?  Why?
#
# **On the targets.** Which of the two sets the achievable accuracy, and why
# does the Fourier basis do so badly on the smooth one?

# %% [markdown]
# ## Answers
#
# **Orthogonality.** Fourier on a uniform grid and Chebyshev at CGL points are
# orthogonal both continuously and discretely.  Chebyshev on a uniform grid is
# orthogonal continuously but not discretely: the quadrature cannot integrate
# the product $\phi_m\phi_n w$ exactly, and the singular weight near $x=\pm1$
# makes it worse.  Monomials are orthogonal in neither sense.  Settings 3 and 4
# differ only in the placement of the points, which is enough to move the
# off-diagonal entries from $10^{-1}$ to $10^{-16}$.
#
# **Conditioning.** $\mathrm{cond}(A)$ for the monomials passes $10^{16}$ near
# $n_b\approx22$.  The approximation keeps improving until about the same
# point and only then turns around: losing digits in $c$ is not the same as
# losing accuracy in $u_{n_b}$, because the errors in the coefficients partly
# cancel when the sum is formed.
#
# **The targets.** The kink limits everything: no basis of smooth global
# functions represents $|x|$ well, and all four curves fall off algebraically.
# On the smooth target the three polynomial settings converge fast, while the
# Fourier basis stalls near $2\times10^{-1}$, because $u_{\rm smooth}$ is not
# periodic on $[-1,1]$ and the series is really reconstructing a function with a
# jump at the ends.
#
# **The overall lesson.** Settings 1, 3 and 4 span the *same space*, so they
# give the *same* best approximation; their error curves lie on top of each
# other until round-off separates them.  What differs is the conditioning of
# $A$, that is, how reliably the coefficients can be computed.
