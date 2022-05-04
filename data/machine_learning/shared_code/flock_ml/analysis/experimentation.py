#
# Copyright © 2016-2021 Flock Freight, Inc.
#
import numpy as np
from scipy.stats import norm
from scipy.special import comb


def get_t_test_group_size_with_pairwise_correction(
    variance: float, delta: float, alpha: float = 0.05, power: float = 0.8, n_groups: int = 2
) -> int:
    """
    Get the sample size per group required to compare n means with given
    statistical power at a given significance level (alpha). Assumes the
    outcome variable is approximately normally distributed (with large
    samples it is OK to relax this assumption) and groups have the same
    variance. If n_groups > 2, this function applies the Bonferroni correction
    for multiple comparisons assuming all pairwise tests will be run.

    :param variance: the assumed variance of the outcome variable
    :param delta: the minimum difference in means that we aim to detect
    :param alpha: the maximum p-value to report statistical significance
    :param power: the desired probability of a true positive
    :param n_groups: number of groups with pairwise planned comparisons
    :return the required sample size per group for this test
    """

    # Assert input parameters define a valid hypothesis test
    assert variance > 0
    assert 0 < alpha < 1
    assert alpha < power < 1
    assert n_groups >= 2

    # Compute the required group size
    adjusted_alpha = alpha / comb(n_groups, 2)
    group_size = ((2 * variance) * (norm.ppf(1 - adjusted_alpha / 2) + norm.ppf(power)) ** 2) / (delta ** 2)
    return int(np.ceil(group_size))


def get_proportion_test_group_size_with_pairwise_correction(
    p1: float, p2: float, alpha: float = 0.05, power: float = 0.8, n_groups: int = 2
) -> int:
    """
    Get the sample size per group required to compare n proportions with given
    statistical power at a given significance level (alpha). If n_groups > 2,
    this function applies the Bonferroni correction for multiple comparisons
    assuming all pairwise tests will be run.

    :param p1: the base proportion to be compared
    :param p2: the second proportion that we aim to detect as different from p1
    :param alpha: the maximum p-value to report statistical significance
    :param power: the desired probability of a true positive
    :param n_groups: number of groups with pairwise planned comparisons
    :return the required sample size per group for this test
    """
    # Assert input parameters define a valid hypothesis test
    assert 0 < p1 < 1
    assert 0 < p2 < 1
    assert alpha < power < 1
    assert n_groups >= 2

    # Compute the required group size
    adjusted_alpha = alpha / comb(n_groups, 2)
    pbar = (p1 + p2) / 2
    qbar = 1 - pbar
    delta = np.abs(p2 - p1)
    group_size = (
        (
            norm.ppf(1 - adjusted_alpha / 2) * np.sqrt(2 * pbar * qbar)
            + norm.ppf(power) * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        )
        / delta
    ) ** 2
    return int(np.ceil(group_size))
