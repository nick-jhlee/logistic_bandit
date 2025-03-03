import numpy as np

from numpy.linalg import LinAlgError
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
from scipy.optimize import linprog
from logbexp.utils.utils import sigmoid


def fit_online_logistic_estimate(arm, reward, current_estimate, vtilde_matrix, vtilde_inv_matrix, constraint_set_radius,
                                 diameter=1, precision=0.1):
    """
    ECOLog estimation procedure.
    """
    # some pre-computation
    sqrt_vtilde_matrix = sqrtm(vtilde_matrix)
    sqrt_vtilde_inv_matrix = sqrtm(vtilde_inv_matrix)
    z_theta_t = np.dot(sqrt_vtilde_matrix, current_estimate)
    z_estimate = z_theta_t
    inv_z_arm = np.dot(sqrt_vtilde_inv_matrix, arm)
    step_size = 1 / (1/4 + 1/(1 + diameter/2)) * 0.75 # slightly smaller step for stability
    iters = int((1/0.75) * np.ceil((9 / 4 + diameter / 8) * np.log(diameter / precision)))

    # few steps of projected gradient descent
    for _ in range(iters):
        pred_probas = sigmoid(np.sum(z_estimate * inv_z_arm))
        grad = z_estimate - z_theta_t + (pred_probas - reward) * inv_z_arm
        unprojected_update = z_estimate - step_size * grad
        z_estimate = project_ellipsoid(x_to_proj=unprojected_update,
                                       ell_center=np.zeros_like(arm),
                                       ecc_matrix=vtilde_matrix,
                                       radius=constraint_set_radius)
    theta_estimate = np.dot(sqrt_vtilde_inv_matrix, z_estimate)
    return theta_estimate


def fit_online_logistic_estimate_bar(arm, current_estimate, vtilde_matrix, vtilde_inv_matrix, constraint_set_radius,
                                     diameter=1, precision=0.1):
    """
    ECOLog estimation procedure to compute theta_bar.
    """
    # some pre-computation
    sqrt_vtilde_matrix = sqrtm(vtilde_matrix)
    sqrt_vtilde_inv_matrix = sqrtm(vtilde_inv_matrix)
    z_theta_t = np.dot(sqrt_vtilde_matrix, current_estimate)
    z_estimate = z_theta_t
    inv_z_arm = np.dot(sqrt_vtilde_inv_matrix, arm)
    step_size = 1 / (1 / 4 + 1 / (1 + diameter / 2)) * 0.75  # slightly smaller step for stability
    iters = int((1/0.75) * np.ceil((9 / 4 + diameter / 8) * np.log(diameter / precision)))

    #few steps of projected gradient descent
    for _ in range(iters):
        pred_probas = sigmoid(np.sum(z_estimate * inv_z_arm))
        grad = z_estimate - z_theta_t + (2*pred_probas - 1) * inv_z_arm
        unprojected_update = z_estimate - step_size * grad
        z_estimate = project_ellipsoid(x_to_proj=unprojected_update,
                                       ell_center=np.zeros_like(arm),
                                       ecc_matrix=vtilde_matrix,
                                       radius=constraint_set_radius)
    theta_estimate = np.dot(sqrt_vtilde_inv_matrix, z_estimate)
    return theta_estimate


def project_ellipsoid(x_to_proj, ell_center, ecc_matrix, radius, safety_check=False):
    """
    Orthogonal projection on ellipsoidal set
    :param x_to_proj: np.array(dim), point to project
    :param ell_center: np.array(dim), center of ellipsoid
    :param ecc_matrix: np.array(dimxdim), eccentricity matrix
    :param radius: float, ellipsoid radius
    :param safety_check: bool, check ecc_matrix psd
    """
    # start by checking if the point to project is already inside the ellipsoid
    ell_dist_to_center = np.dot(x_to_proj - ell_center, np.linalg.solve(ecc_matrix, x_to_proj - ell_center))
    is_inside = (ell_dist_to_center - radius ** 2) < 1e-3
    if is_inside:
        return x_to_proj

    # check eccentricity is symmetric PSD
    if safety_check:
        sym_check = np.allclose(ecc_matrix, ecc_matrix.T)
        psd_check = np.all(np.linalg.eigvals(ecc_matrix) > 0)
        if not sym_check or not psd_check:
            raise ValueError("Eccentricity matrix is not symetric or PSD")

    # some pre-computation
    dim = len(x_to_proj)
    sqrt_psd_matrix = sqrtm(ecc_matrix)
    y = np.dot(sqrt_psd_matrix, x_to_proj - ell_center)

    # opt function for projection
    def fun_proj(lbda):
        try:
            solve = np.linalg.solve(ecc_matrix + lbda * np.eye(dim), y)
            res = lbda * radius ** 2 + np.dot(y, solve)
        except LinAlgError:
            res = np.inf
        return res

    # find proj
    lbda_opt = minimize_scalar(fun_proj, method='bounded', bounds=(0, 1000), options={'maxiter': 500})
    eta_opt = np.linalg.solve(ecc_matrix + lbda_opt.x * np.eye(dim), y)
    x_projected = np.dot(sqrt_psd_matrix, eta_opt) + ell_center

    return x_projected


# Thanks to Brano for the code for G-optimal design
# D-optimal design
def d_grad(V, p, gamma=1e-6, return_grad=True):
    """Value of D-optimal objective and its gradient.

    V: n x d x d feature outer products
    p: distribution over n features (design)
    """
    n, d, _ = V.shape

    # inverse of the sample covariance matrix
    G = np.einsum("ijk,i->jk", V, p) + gamma * np.eye(d)
    invG = np.linalg.inv(G)

    # objective value (log det)
    sign, obj = np.linalg.slogdet(G)
    obj *= - sign
    if return_grad:
        # gradient of the objective
        M = np.einsum("kl,ilj->ikj", invG, V)
        dp = - np.trace(M, axis1=-2, axis2=-1)
    else:
        dp = 0

    return obj, dp

def fw_design(arm_list, pi_0=None, num_iters=100, tol=1e-6, A_ub=None, b_ub=None, printout=False):
    """Frank-Wolfe algorithm for design optimization.

    V: n x d x d feature outer products
    pi_0: initial distribution over n features (design)
    num_iters: maximum number of Frank-Wolfe iterations
    tol: stop when two consecutive objective values differ by less than tol
    A_ub: matrix A in design constraints A pi >= b
    b_ub: vector b in design constraints A pi >= b
    """
    X = np.array(arm_list)  # feature matrix
    V = np.einsum("ij,ik->ijk", X, X)  # feature outer products
    n, d, _ = V.shape

    if pi_0 is None:
        # initial allocation weights are 1 / n and they add up to 1
        pi = np.ones(n) / n
    else:
        pi = np.copy(pi_0)

    # initialize constraints
    if A_ub is None:
        A_ub_fw = np.ones((1, n))
        b_ub_fw = 1
    else:
        # last constraint guarantees that pi is a distribution
        A_ub_fw = np.zeros((A_ub.shape[0] + 1, A_ub.shape[1]))
        A_ub_fw[: -1, :] = A_ub
        A_ub_fw[-1, :] = np.ones((1, n))
        b_ub_fw = np.zeros(b_ub.size + 1)
        b_ub_fw[: -1] = b_ub
        b_ub_fw[-1] = 1

    # Frank-Wolfe iterations
    for iter in range(num_iters):
        # compute gradient at the last solution
        pi_last = np.copy(pi)
        last_obj, grad = d_grad(V, pi_last)

        if printout:
            print("%.4f" % last_obj, end=" ")

        # find a feasible LP solution in the direction of the gradient
        result = linprog(grad, A_ub_fw, b_ub_fw, bounds=[0, 1], method="highs")
        pi_lp = result.x
        pi_lp = np.maximum(pi_lp, 0)
        pi_lp /= pi_lp.sum()

        # line search in the direction of the gradient
        num_ls_iters = 100
        best_step = 0.0
        best_obj = last_obj
        for ls_iter in range(num_ls_iters):
            step = np.power(0.9, ls_iter)
            pi_ = step * pi_lp + (1 - step) * pi_last
            obj, _ = d_grad(V, pi_, return_grad=False)
            if obj < best_obj:
                # record an improved solution
                best_step = step
                best_obj = obj

        # update solution
        pi = best_step * pi_lp + (1 - best_step) * pi_last

        if last_obj - best_obj < tol:
            break
        iter += 1

    if printout:
        print()

    pi = np.maximum(pi, 0)
    pi /= pi.sum()
    return pi