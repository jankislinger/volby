import json
import uuid
from multiprocessing import Pool

import numpy as np
import pandas as pd

NUM_PROC = 10
NUM_RUNS = 500
NUM_ITERATIONS = 100
NUM_ITER_WARMUP = 50


def main_parallel():
    pool = Pool(NUM_PROC)
    pool.map(run, range(NUM_RUNS))
    pool.join()


def run(seed=None):
    if seed is not None:
        np.random.seed(seed)
    counts_valid, counts_invalid = load_data()
    x = mcmc(counts_valid, counts_invalid, NUM_ITERATIONS, NUM_ITER_WARMUP)
    res = x.mean(0)
    print((res * 100).round(2))
    with open(f"results/{uuid.uuid4()}.json", "w") as f:
        json.dump(list(res), f)


def mcmc(counts_valid: np.ndarray, counts_invalid: np.ndarray, niter: int, warmup: int):
    num_regions, num_parties = counts_valid.shape
    assert counts_invalid.shape == (num_regions,)

    beta = counts_valid.sum() / counts_invalid.sum()
    prior_invalid = np.tile(np.array([1, beta]), (num_parties, 1))

    alphas = counts_valid.sum(0)
    alphas = alphas / alphas.mean()
    prior_party = np.tile(alphas, (num_regions, 1))

    res = []

    posterior_invalid = prior_invalid.copy()
    posterior_party = prior_party.copy()

    for i in range(warmup + niter):
        prob_invalid = sample_prob_invalid(posterior_invalid)
        prob_party = sample_prob_party(posterior_party)
        invalid_sampled = sample_invalid_counts(
            counts_invalid, prob_invalid, prob_party
        )

        posterior_invalid = get_conditional_invalid(prior_invalid, counts_valid, invalid_sampled)
        posterior_party = get_conditional_party(prior_party, counts_valid, invalid_sampled)

        if i >= warmup:
            res.append(prob_invalid)

    return np.row_stack(res)


def sample_prob_invalid(posterior_invalid: np.ndarray):
    # posterior_invalid (num parties, 2): beta distribution for probability of
    #  invalid vote given selected party
    prob_invalid = np.apply_along_axis(np.random.dirichlet, 1, posterior_invalid)
    return prob_invalid[:, 0]


def sample_prob_party(posterior_party: np.ndarray):
    # posterior_party (num regions, num_ parties): dirichlet distribution for
    #  probabilities of selecting a party given some region
    return np.apply_along_axis(np.random.dirichlet, 1, posterior_party)


def sample_invalid_counts(counts_invalid, prob_invalid, prob_party):
    prop_invalid_party = prob_invalid.reshape((1, -1)) * prob_party
    prop_invalid_party /= prop_invalid_party.sum(1, keepdims=True)
    return apply_multinomial(counts_invalid, prop_invalid_party)


def get_conditional_invalid(prior_invalid, counts_valid, invalid_sampled):
    counts_per_parties = np.column_stack(
        [
            invalid_sampled.sum(0),
            counts_valid.sum(0),
        ]
    )

    return prior_invalid + counts_per_parties


def get_conditional_party(prior_party, counts_valid, invalid_sampled):
    counts_per_parties = counts_valid + invalid_sampled
    return prior_party + counts_per_parties


def apply_multinomial(n_vec: np.ndarray, pvals_mat: np.ndarray) -> np.ndarray:
    res = []
    for n, pvals in zip(n_vec, pvals_mat):
        x = np.random.multinomial(n, pvals)
        res.append(x)
    return np.row_stack(res)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    counts_valid = pd.read_parquet("data/counts_valid.pq")
    counts_invalids = pd.read_parquet("data/counts_invalid.pq")

    return counts_valid.values, counts_invalids["invalid"].values


def run_on_dummy_data():
    nregions = 1000
    nparties = 10

    sizes = np.array([100] * nregions)
    true_prob_party = np.row_stack(
        [np.random.dirichlet([10] * nparties) for _ in range(nregions)]
    )
    true_prob_invalid = np.array([0.02, 0.03, 0.05] + [0.02] * (nparties - 3))
    true_counts = apply_multinomial(sizes, true_prob_party)
    true_invalid_counts = np.column_stack(
        [
            np.random.binomial(n, p)
            for p, n in zip(true_prob_invalid, true_counts.transpose())
        ]
    )
    true_valid_counts = true_counts - true_invalid_counts
    x = mcmc(true_valid_counts, true_invalid_counts.sum(1), 250, 50)
    print(x.mean(0))


if __name__ == "__main__":
    main_parallel()
