import math

from config import (A_LARGE_MU, A_MEDIUM_MU, A_SIGMA, A_SMALL_MU, ETA, FUZZY_EPS,
                    PROTOTYPES, Q_HIGH_MU, Q_LEVELS, Q_LOW_MU, Q_MEDIUM_MU, Q_SIGMA,
                    SCALE)

QUALITY_TERMS = ['Low', 'Medium', 'High']
AREA_TERMS = ['Small', 'Medium', 'Large']


def gaussian(x, mu, sigma):
    return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def quality_membership(q):
    low = 1.0 if q <= Q_LOW_MU else gaussian(q, Q_LOW_MU, Q_SIGMA)
    medium = gaussian(q, Q_MEDIUM_MU, Q_SIGMA)
    high = 1.0 if q >= Q_HIGH_MU else gaussian(q, Q_HIGH_MU, Q_SIGMA)
    return {'Low': low, 'Medium': medium, 'High': high}


def area_membership(a):
    small = 1.0 if a <= A_SMALL_MU else gaussian(a, A_SMALL_MU, A_SIGMA)
    medium = gaussian(a, A_MEDIUM_MU, A_SIGMA)
    large = 1.0 if a >= A_LARGE_MU else gaussian(a, A_LARGE_MU, A_SIGMA)
    return {'Small': small, 'Medium': medium, 'Large': large}


def expected_quality_score(q_probs):
    return sum(i * float(q_probs[i]) for i in range(Q_LEVELS)) / (Q_LEVELS - 1)


def area_ratio(mask):
    return mask.float().mean().item()


def fuzzy_densities():
    return SCALE * ETA, SCALE * (1.0 - ETA)


def choquet_integral(v_q, v_a, g_q, g_a):
    v_min, v_max = min(v_q, v_a), max(v_q, v_a)
    g_dominant = g_q if v_q >= v_a else g_a
    return v_min + (v_max - v_min) * g_dominant


def rule_consequents():
    g_q, g_a = fuzzy_densities()
    consequents = {}
    for m, q_term in enumerate(QUALITY_TERMS):
        for n, a_term in enumerate(AREA_TERMS):
            consequents[(q_term, a_term)] = choquet_integral(PROTOTYPES[m], PROTOTYPES[n], g_q, g_a)
    return consequents


CONSEQUENTS = rule_consequents()


def rule_activations(q_mu, a_mu):
    return {(q_term, a_term): q_mu[q_term] * a_mu[a_term]
            for q_term in QUALITY_TERMS for a_term in AREA_TERMS}


def defuzzify(activations, consequents):
    numerator = sum(alpha * consequents[key] for key, alpha in activations.items())
    denominator = sum(activations.values())
    return numerator / (denominator + FUZZY_EPS)


def compute_fuzzy_weight(q_probs, mask):
    q = expected_quality_score(q_probs)
    a = area_ratio(mask)
    activations = rule_activations(quality_membership(q), area_membership(a))
    return defuzzify(activations, CONSEQUENTS)


def compute_linear_weight(q_probs, mask, lam):
    return lam * expected_quality_score(q_probs) + (1.0 - lam) * area_ratio(mask)
