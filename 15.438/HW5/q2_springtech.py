"""
Q2: SpringTech structural credit model on a binomial tree.

Assets: A0 = $100M. Each year up by factor u or down by 1/u.
Zero-coupon debt: F = $85M, T = 4 years.
Risk-neutral p = 55%. Risk-free r = 3%.
Default payoff: if A_T >= F, debt holders receive F.
                if A_T  < F, debt holders receive 0.80 * A_T.
"""

import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq

def price_debt(u, p, r, F, A0, T, recovery_frac=0.80):
    """
    Price zero-coupon defaultable debt under a multiplicative binomial model.

    Returns: (price, ytm, prob_default, expected_recovery_rate_given_default,
              terminal_assets, terminal_debt_payoffs, terminal_probs)
    """
    d = 1 / u
    # Number of up moves k in T periods: 0..T
    k = np.arange(T + 1)
    A_T = A0 * (u ** k) * (d ** (T - k))   # terminal asset values
    probs = binom.pmf(k, T, p)              # risk-neutral probs

    # Debt payoff at maturity
    debt_payoff = np.where(A_T >= F, F, recovery_frac * A_T)

    # Today's price = PV of risk-neutral expected payoff
    price = np.sum(probs * debt_payoff) / (1 + r) ** T
    ytm = (F / price) ** (1 / T) - 1

    # Default = states where A_T < F
    default_mask = A_T < F
    p_default = float(np.sum(probs[default_mask]))
    if p_default > 0:
        # Expected recovery given default, expressed as fraction of face
        recovery_dollars = np.sum(probs[default_mask] * recovery_frac * A_T[default_mask])
        exp_recovery_rate = recovery_dollars / (p_default * F)
    else:
        exp_recovery_rate = float('nan')

    return price, ytm, p_default, exp_recovery_rate, A_T, debt_payoff, probs


def value_at_risk(price, debt_payoff, probs, level=0.05):
    """
    VaR at the given confidence level, expressed as loss vs current price.

    Convention (per class notes): VaR is the *worst expected loss* at a given
    confidence level under physical probabilities. Per the problem statement,
    we use the risk-neutral probabilities as a proxy for physical probs.

    On a discrete tree, take the worst payoff whose cumulative tail
    probability is <= level (i.e. the smallest payoff x such that
    P(payoff <= x) <= level). The 5% loss is at most this severe.
    """
    order = np.argsort(debt_payoff)            # worst -> best
    sorted_payoffs = debt_payoff[order]
    sorted_probs = probs[order]
    cum = np.cumsum(sorted_probs)

    # Largest k such that cum[k] <= level. If even cum[0] > level, the tail
    # is bracketed inside the worst node and we report that node.
    eligible = np.where(cum <= level)[0]
    idx = eligible[-1] if len(eligible) > 0 else 0
    var_payoff = sorted_payoffs[idx]

    var_loss = price - var_payoff              # loss vs initial investment
    return var_loss, var_payoff, cum, sorted_payoffs


def report(label, u, p, r, F, A0, T):
    price, ytm, pd, rec, A_T, payoff, probs = price_debt(u, p, r, F, A0, T)
    print(f"\n{'=' * 70}\n{label}  (u = {u}, sigma proxy = {(u-1)*100:.0f}%)\n{'=' * 70}")
    print(f"{'k(up)':>6}{'A_T ($M)':>12}{'Payoff ($M)':>14}{'RN prob':>12}")
    for i in range(T + 1):
        print(f"{i:>6}{A_T[i]:>12.4f}{payoff[i]:>14.4f}{probs[i]:>12.6f}")
    print(f"\nDebt price today:           ${price:.4f}M")
    print(f"Yield to maturity:          {ytm*100:.4f}%")
    print(f"RN prob of default at T:    {pd*100:.4f}%")
    print(f"E[recovery | default]:      {rec*100:.4f}% of face")

    # VaR at 5% over 4 years (treating RN probs as physical, per problem)
    var_loss, var_payoff, cum, sorted_payoffs = value_at_risk(
        price, payoff, probs, level=0.05)
    print(f"\nVaR analysis (5% level, 4 years, RN probs as physical):")
    print(f"  Sorted payoffs (worst -> best): {sorted_payoffs}")
    print(f"  Cumulative probs:               {cum}")
    print(f"  5%-VaR terminal payoff:         ${var_payoff:.4f}M")
    print(f"  5%-VaR loss vs current price:   ${var_loss:.4f}M")
    return price, ytm, pd, rec


# --- Part (a)-(c): u = 1.10 ---
report("Parts (a)-(c): u = 1.10", u=1.10, p=0.55, r=0.03, F=85, A0=100, T=4)

# --- Part (d): u = 1.15 ---
report("Part (d): u = 1.15", u=1.15, p=0.55, r=0.03, F=85, A0=100, T=4)
