# Imports
import numpy as np

# XNPV function that takes cashflows, the parameter k and the discount rate r
def XNPV (CF, k, r):
  XNPV = 0.0
  for i in range (len(CF)):
    XNPV += CF[i] / (1+((1+k)**i)*r)**i
  return XNPV


# XIRR function to solve for the analog IRR
def XIRR (CF, k):
  # Tolerance for checking convergence
  tol = 1e-5
  count = 0
  # Initialization of variables
  low = 0
  high = 1
  mid = (low + high) / 2
  XNPV_mid = XNPV(CF,k,mid)

  while ( abs(XNPV_mid) > tol and count < 1e+4 ) :
    if XNPV_mid > 0 :
      # need to increase analog discount rate to make XNPV smaller
      low = mid
    else :
      # need to decrease analog discount rate to make XNPV larger
      high = mid
    mid = (low + high) / 2
    XNPV_mid = XNPV(CF,k,mid)
    count += 1
  return mid


# Test 1
CF1 = [-40,10,20,30]
k1 = 0.05
print(XIRR(CF1,k1))

# Test 2
CF2 = [-300,10,20,30,40,50,60,70,80,90,100]
k2 = 0.05
print(XIRR(CF2,k2))


# Question 1
CF3 = [-3000,1000,2000,1000]
k3 = 0.03
print(XIRR(CF3,k3))
