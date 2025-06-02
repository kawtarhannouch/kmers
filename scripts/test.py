import numpy as np
from collections import Counter
from scipy.stats import poisson, gamma, nbinom
from scipy.optimize import minimize
import ot

def fit_poisson(data):
    mean_value = float(data.mean())
    print("Poisson λ =", mean_value)
    return mean_value

def fit_gamma(data):
    mean, var = data.mean(), data.var()
    epsilon = 1e-6
    alpha = mean ** 2 / (var + epsilon)
    beta = mean / (var + epsilon)
    print("Gamma α =", alpha)
    print("Gamma β =", beta)
    return alpha, beta

def fit_negative_binomial(data):
    mu, var = data.mean(), data.var()
    print(f"NB: μ= {mu}, var={var}")
    r = mu ** 2 / (var - mu + 1e-6)
    p = r / (r + mu)
    print("NB r =", r)
    print("NB p =", p)
    return r, p

class DistributionFitter:
    def __init__(self, file_path):
        self.counts = self._read_counts(file_path)
        self.count_range = np.unique(self.counts)
        self.empirical_histogram = self._compute_empirical_histogram()
        self.positions = self.count_range[:, None]
        print(self.positions)
        self.cost_matrix = ot.utils.dist(self.positions, self.positions, metric="euclidean")
        #print(self.cost_matrix)

    def _read_counts(self, path):
        with open(path) as f:
            return np.fromiter(
                (int(line.split()[1]) for line in f if line.strip()),
                dtype=np.int32
            )

    def _compute_empirical_histogram(self):
        frequency = Counter(self.counts)
        histogram = np.array([frequency[k] for k in self.count_range], dtype=float)
        histogram /= histogram.sum()
        return histogram

    def compute_model_pdf(self, x1, x2):
        x1, x2 = sorted((x1, x2))
        error_region = self.counts[self.counts <= x1]
        true_region = self.counts[(self.counts > x1) & (self.counts <= x2)]
        artifact_region = self.counts[self.counts > x2]

        lam = fit_poisson(error_region)
        alpha, beta = fit_gamma(true_region)
        r, p = fit_negative_binomial(artifact_region)

        pdf = np.zeros_like(self.count_range, dtype=float)
        m1 = self.count_range <= x1
        m2 = (self.count_range > x1) & (self.count_range <= x2)
        m3 = self.count_range > x2

        pdf[m1] = poisson.pmf(self.count_range[m1], lam)
        pdf[m2] = gamma.pdf(self.count_range[m2], a=alpha, scale=1 / beta)
        pdf[m3] = nbinom.pmf(self.count_range[m3], r, p)
        pdf /= pdf.sum()

        return pdf

    def compute_ot_loss(self, x_pair):
        x1, x2 = x_pair
        model_pdf = self.compute_model_pdf(x1, x2)
        print(model_pdf)
        loss = ot.emd2(self.empirical_histogram, model_pdf, self.cost_matrix, numItermax=100000000)
        print(f"loss: ot:{loss}")
        #print(f"Loss for x1={x1}, x2={x2}: {loss}")
        return loss

class CutoffOptimizer:
    def __init__(self, fitter, iterations=100000, tolerance=1e-8, verbose=True):
        self.fitter = fitter
        self.iterations = iterations
        self.tolerance = tolerance
        self.verbose = verbose

    def optimize(self, x1_init, x2_init):
        bounds = [(1, self.fitter.counts.max()), (2, self.fitter.counts.max())]

        def callback(xk):
            if self.verbose:
                print(f"Iteration: x1 = {xk[0]}, x2 = {xk[1]}")

        result = minimize(
            self.fitter.compute_ot_loss,
            x0=np.array([x1_init, x2_init], dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": self.iterations,
                "ftol": self.tolerance,
                "disp": self.verbose
            },
            callback=callback
        )

        best_x1, best_x2 = np.round(result.x).astype(int)
        return best_x1, best_x2

def find_cutoffs(file_path, x1_init, x2_init, iterations, tolerance, verbose=True):
    fitter = DistributionFitter(file_path)
    optimizer = CutoffOptimizer(fitter, iterations, tolerance, verbose)
    return optimizer.optimize(x1_init, x2_init)

if __name__ == "__main__":
    INPUT_FILE = (
        "/data/Kaoutar/dev/py/kmers/test_data/trimming/"
        "jellyfish_21_100N_lane1/100N_lane1_20220218000_S1_L001_001_kmers.txt"
    )

    best_x1, best_x2 = find_cutoffs(
        INPUT_FILE,
        x1_init= 25,
        x2_init=100000,
        iterations=100000,
        tolerance=1e-8,
        verbose=True,
    )

    print(f"\nFinal optimal cutoffs: x1 = {best_x1}, x2 = {best_x2}")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        init_values = torch.tensor([25, 10000])
        self.params = nn.Parameter(init_values)

    def forward(self):
        return self.params



