import numpy as np
from collections import Counter
from scipy.stats import poisson, gamma, nbinom
from scipy.optimize import minimize
import ot

# Read counts from file
def read_counts(file_path):
    with open(file_path) as f:
        return np.fromiter(
            (int(line.split()[1]) for line in f if line.strip()),
            dtype=np.int32
        )

# Compute empirical histogram
def compute_empirical_histogram(counts):
    count_range = np.unique(counts)
    frequency = Counter(counts)
    histogram = np.array([frequency[k] for k in count_range], dtype=float)
    histogram /= histogram.sum()
    return histogram, count_range

# Fit Poisson, Gamma, NB
def fit_poisson(data):
    mean_value = float(data.mean())
    return mean_value

def fit_gamma(data):
    mean, var = data.mean(), data.var()
    epsilon = 1e-6
    alpha = mean**2 / (var + epsilon)
    beta = mean / (var + epsilon)
    return alpha, beta

def fit_negative_binomial(data):
    mu, var = data.mean(), data.var()
    r = mu**2 / (var - mu + 1e-6)
    p = r / (r + mu)
    return r, p

# Build model PDF
def model_pdf(x, counts, count_range):
    x1, x2 = sorted(x)

    error_region = counts[counts <= x1]
    true_region = counts[(counts > x1) & (counts <= x2)]
    artifact_region = counts[counts > x2]

    lam = fit_poisson(error_region)
    alpha, beta = fit_gamma(true_region)
    r, p = fit_negative_binomial(artifact_region)

    pdf = np.zeros_like(count_range, dtype=float)
    m1 = count_range <= x1
    m2 = (count_range > x1) & (count_range <= x2)
    m3 = count_range > x2

    pdf[m1] = poisson.pmf(count_range[m1], lam)
    pdf[m2] = gamma.pdf(count_range[m2], a=alpha, scale=1/beta)
    pdf[m3] = nbinom.pmf(count_range[m3], r, p)

    pdf /= pdf.sum()
    return pdf

# Define objective function exactly matching the style you asked
def objective_function(x, empirical_histogram, counts, count_range, cost_matrix):
    return ot.emd2(empirical_histogram, model_pdf(x, counts, count_range), cost_matrix, numItermax=100000000)

# Main optimization function
def find_cutoffs(file_path, x0, iterations=100000, tolerance=1e-8, verbose=True):
    counts = read_counts(file_path)
    empirical_histogram, count_range = compute_empirical_histogram(counts)
    positions = count_range[:, None]
    cost_matrix = ot.utils.dist(positions, positions, metric="euclidean")

    # Wrap objective to pass only x to minimize()
    def wrapped_objective(x):
        loss = objective_function(x, empirical_histogram, counts, count_range, cost_matrix)
        if verbose:
            print(f"x1={x[0]}, x2={x[1]}, loss={loss}")
        return loss

    result = minimize(
        wrapped_objective,
        x0=np.array(x0, dtype=float),
        method="L-BFGS-B",
        bounds=[(1, counts.max()), (2, counts.max())],
        options={"maxiter": iterations, "ftol": tolerance, "disp": verbose}
    )

    best_x1, best_x2 = np.round(result.x).astype(int)
    return best_x1, best_x2

if __name__ == "__main__":
    INPUT_FILE = (
        "/data/Kaoutar/dev/py/kmers/test_data/trimming/"
        "jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
    )

    best_x1, best_x2 = find_cutoffs(
        INPUT_FILE,
        x0=[25, 100000],    # initial guess
        iterations=100000,
        tolerance=1e-8,
        verbose=True,
    )

    print(f"\nFinal optimal cutoffs: x1 = {best_x1}, x2 = {best_x2}")
