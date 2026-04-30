# Imports
import numpy as np

# Function that simulates a path
def firmValue (T, sigma, p , x_0) :
  x_current = x_0
  output = [x_current]

  for i in range (T) :
    epsilon = np.random.binomial(1,p) * 2 - 1
    x_new = x_current + sigma * epsilon
    output.append(x_new)
    x_current = x_new

  return np.array(output)


# Estimation of the credit default probability
def CD_probability (number_simulations, T, sigma, p, x_0, threshold_percent, tolerance) :
  threshold = x_0 * (1 - threshold_percent)
  count_belowthreshold = 0
  for i in range (number_simulations) :
    # Calculate series of firm values
    values = firmValue(T, sigma, p, x_0)
    # Check if any of the values are below threshold
    if (sum(values < threshold + tolerance) != 0) :
      count_belowthreshold += 1
  return print('Probability that values goes ' + str(int(threshold_percent * 100)) + '% below its initial valuation of ' + str(x_0) + ' is ' + str(count_belowthreshold/number_simulations))

# Test
T, sigma, p, x_0 = 120, 0.6, 0.5, 10
number_simulations = 100000
threshold_percent = 0.10
tolerance = 1e-7
print(CD_probability(number_simulations, T, sigma, p, x_0, threshold_percent, tolerance))


# Question 4
T, sigma, p, x_0 = 120, 0.6, 0.5, 10
number_simulations = 100000
threshold_percent = 0.25
tolerance = 1e-7
print(CD_probability(number_simulations, T, sigma, p, x_0, threshold_percent, tolerance))


# Question 5
import matplotlib.pyplot as plt
T, sigma, p, x_0 = 120, 0.6, 0.5, 10
number_simulations = 5
plt.xlabel("Time Periods")
plt.ylabel("Firm Valuation")
plt.title("Paths of Possible Firm Valuations")

for i in range (number_simulations) :
  plt.plot(firmValue(T, sigma, p, x_0))

plt.savefig("991594666_Figure1.pdf")
