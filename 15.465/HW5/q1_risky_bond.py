"""
Q1: Risky bond pricing with time-varying default and recovery rates.

Bond: 5-year, 6.5% annual coupon, $1,000 face, BB-rated.
Default probs (conditional on surviving to start of year):
    Year 1: 25%, Years 2-5: 5%
Recovery rates (% of face):
    Year 1: 29%, Years 2-5: 40%
Required expected return (discount rate): 5.375%

Convention (matches default.xls from class notes #7):
- If the bond defaults in year t, holder receives Recovery_t * Face at end of year t
  (no coupon that year).
- If it survives year t, holder receives the coupon (plus principal at maturity).
"""

import numpy as np
from scipy.optimize import brentq

# --- Inputs ---
F = 1000.0
c = 0.065
coupon = c * F  # $65
T = 5
r = 0.05375  # required expected return

# Time-varying conditional default probabilities and recovery rates
p_default = np.array([0.25, 0.05, 0.05, 0.05, 0.05])      # p_t for t = 1..5
recovery  = np.array([0.29, 0.40, 0.40, 0.40, 0.40])      # R_t for t = 1..5

# --- Survival probabilities ---
# S(t) = P(survive through end of year t) = prod_{i<=t} (1 - p_i)
# Probability of defaulting *in* year t = S(t-1) * p_t
survival_prior = np.empty(T)   # S(t-1): survival to start of year t
S = 1.0
for t in range(T):
    survival_prior[t] = S
    S *= (1 - p_default[t])

prob_default_in_t = survival_prior * p_default
prob_survive_t    = survival_prior * (1 - p_default)  # = S(t)

# --- Expected cash flows by year ---
# Year t cash flow:
#   if default in year t: Recovery_t * F
#   if survive year t and t < T: coupon
#   if survive year T: coupon + F
expected_cf = np.zeros(T)
for t in range(T):
    survive_payment = coupon + (F if t == T - 1 else 0.0)
    default_payment = recovery[t] * F
    expected_cf[t] = (prob_survive_t[t] * survive_payment
                      + prob_default_in_t[t] * default_payment)

# --- Price = PV of expected cash flows at required return r ---
years = np.arange(1, T + 1)
discount = (1 + r) ** (-years)
price = float(np.sum(expected_cf * discount))

# --- Yield to maturity: discount *promised* cash flows ---
# Promised: coupon every year + face at maturity
promised_cf = np.full(T, coupon)
promised_cf[-1] += F

def price_at_yield(y):
    return np.sum(promised_cf * (1 + y) ** (-years)) - price

ytm = brentq(price_at_yield, -0.5, 1.0)

# --- Output ---
print("=" * 70)
print("Q1: BB-rated bond pricing")
print("=" * 70)
print(f"{'Year':>5}{'p_default':>12}{'recovery':>12}{'S(t-1)':>12}"
      f"{'P(def in t)':>14}{'E[CF_t]':>12}{'PV':>12}")
for t in range(T):
    print(f"{t+1:>5}{p_default[t]:>12.4f}{recovery[t]:>12.4f}"
          f"{survival_prior[t]:>12.6f}{prob_default_in_t[t]:>14.6f}"
          f"{expected_cf[t]:>12.4f}{expected_cf[t]*discount[t]:>12.4f}")
print("-" * 70)
print(f"Price per $1,000 face: ${price:.4f}")
print(f"Yield to maturity:     {ytm*100:.4f}%")
print(f"Credit spread (YTM - r): {(ytm - r)*100:.4f}%")
