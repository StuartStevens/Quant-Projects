import numpy as np
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# Defining a stochastic Heston model simulator for multiple paths, using Euler's method
def simulate_multiple_paths_heston(delta_t, r, kappa, theta, xi, rho, s_0, v_0, n_iterations, n_paths):
    # Generate time steps
    end_time = n_iterations * delta_t
    num_points = n_iterations + 1  # Include the endpoint
    times = np.linspace(0, end_time, num_points)

    # Covariance matrix with correlation rho
    cov_matrix = [[delta_t, rho * delta_t],
                  [rho * delta_t, delta_t]]

    # Generate a sample from a multivariate normal distribution with specified correlation
    # Note, we generate the sample for all paths at once for computational efficiency reasons
    correlated_random_walks = np.random.multivariate_normal([0, 0], cov_matrix, (n_paths, n_iterations))

    # Extract the individual Wiener processes from the multivariate normal sample
    dW_s = correlated_random_walks[:, :, 0]
    dW_v = correlated_random_walks[:, :, 1]

    # Initialize arrays for stock prices and volatilities
    prices = np.zeros((n_paths, n_iterations + 1))
    volatilities = np.zeros((n_paths, n_iterations + 1))
    prices[:, 0] = s_0
    volatilities[:, 0] = v_0


    # Simulate paths using the Heston model formula and Euler's method
    for i in range(1, n_iterations + 1):
        volatilities[:, i] = np.maximum(volatilities[:, i-1] + kappa * 
        (theta - volatilities[:, i-1]) * delta_t + xi * 
        np.sqrt(volatilities[:, i-1]) * dW_v[:, i-1], 0)

        prices[:, i] = prices[:, i-1] + prices[:, i-1] * 
        (r * delta_t + np.sqrt(volatilities[:, i-1]) * dW_s[:, i-1])

    return times, prices


# Defining a function which takes in the simulated paths, 
# and calculates risk neutral call option price at desired time
def option_price_calc(times,paths,T,t,r,k):

    n_paths = paths.shape[0]
    final_payoffs = np.zeros(n_paths)

    time_position = np.argmin(np.abs(times - T))
    final_prices = paths[:, time_position]
    
    # call option payoff filtering  
    final_payoffs = np.maximum(final_prices - k, 0)

    # Discounting payoffs
    average_payoff = np.mean(final_payoffs)
    discounted_payoff = np.exp(-r * (T-t)) * average_payoff

    return discounted_payoff

# Defining a function which takes in the simulated paths, 
# and calculates risk neutral up-and-in call option price at desired time
def barrier_upAndIn_price_calc(times, paths, T, t, r, k, barrier):
    time_position = np.argmin(np.abs(times - T))

    # Instead of checking all prices, we check if the maximum price crossed the barrier, 
    # Then store this as a boolean variable 
    max_prices_up_to_stop = np.max(paths[:, :time_position + 1], axis=1)
    crossed_barrier = max_prices_up_to_stop > barrier

    # For each of the paths that crossed the barrier, calculate the payoff
    final_prices_at_stop = paths[np.arange(paths.shape[0]), time_position]
    payoffs = np.where(crossed_barrier, np.maximum(final_prices_at_stop - k, 0), 0)

    # Averaging and disocunting 
    average_payoff = np.mean(payoffs)
    discounted_payoff = np.exp(-r * (T - t)) * average_payoff
    
    # Identify paths that crossed the barrier
    crossed_paths = paths[crossed_barrier]

    # Return both the discounted payoff and the paths that crossed the barrier
    return discounted_payoff, crossed_paths

# Defining a function which takes in the simulated paths, 
# and calculates risk neutral up-and-in call option price at desired time.
# The same method is used as above, except using the adjusted payoff filter
def barrier_upAndOut_price_calc(times, paths, T, t, r, k, barrier):
    time_position = np.argmin(np.abs(times - T))

    max_prices_up_to_stop = np.max(paths[:, :time_position + 1], axis=1)

    not_crossed_barrier = max_prices_up_to_stop < barrier

    final_prices_at_stop = paths[:, time_position]
    payoffs = np.where(not_crossed_barrier, np.maximum(final_prices_at_stop - k, 0), 0)

    average_payoff = np.mean(payoffs)
    discounted_payoff = np.exp(-r * (T - t)) * average_payoff

    not_crossed_paths = paths[not_crossed_barrier]

    return discounted_payoff, not_crossed_paths

# Simple BS calculator
def black_scholes_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Given the call price, and other inputs, solve for the implied vol under BS
def implied_volatility(S, K, T, r, market_price):
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma) - market_price
    try:
        return brentq(objective, 0.01, 2.0)
    except ValueError:
        return np.nan

# Code that generates equidistant feller parameters within a range, and then filters
# them based on if the feller condition is violated
# Then return a list of lists where each item is a valid combination of Heston params. 
def generate_feller_params(kappa_range, theta_range, xi_range):
    kappa, theta, xi = np.meshgrid(kappa_range, theta_range, xi_range, indexing='ij')
    feller_condition = 2 * kappa * theta > xi**2
    # Extract combos where feller condition is met
    valid_indices = np.where(feller_condition)
    valid_combinations = list(zip(kappa[valid_indices], theta[valid_indices], xi[valid_indices]))
    return valid_combinations

# Function that splits the parameters into a number of splits 
# This was used so we can run the simulation in batches
# We implemented this becuase we kept running out of RAM
def split_combinations(combinations, num_splits):
    split_size = len(combinations) // num_splits
    splits = [combinations[i * split_size:(i + 1) * split_size] for i in range(num_splits)]
    
    # Any left over combinations assigned in a round-robin fashion
    if len(combinations) % num_splits != 0:
        remaining_elements = combinations[num_splits * split_size:]
        for i, element in enumerate(remaining_elements):
            splits[i % num_splits].append(element)
    
    return splits


# Function that runs our simulator, and returns the paths, and parameters inputted
def simulate_paths_and_run_simulation(params):

    kappa, theta, xi, rho = params

    times, paths = simulate_multiple_paths_heston(delta_t, r, kappa, theta, xi, rho, 
    s_0, v_0**2, n_iterations, n_paths)

    return kappa, theta, xi, rho, times, paths


# Final function which takes in parameters and returns: 
# T, K, B, implied vol, and some option prices
def calculate_option_prices_for_params(params):
    
    kappa, theta, xi, rho, times, paths = params

    T_values = np.linspace(0.5, 1.5, volSurface_length)
    K_multipliers = np.linspace(0.8, 1.2, volSurface_length)
    Barrier_multipliers = np.linspace(1.1, 1.5, volSurface_length)

    # Prepare all T, K, and Barrier combinations
    T_grid, K_multiplier_grid, Barrier_multiplier_grid = np.meshgrid(T_values, K_multipliers, 
    Barrier_multipliers, indexing='ij')
    K_grid = s_0 * K_multiplier_grid.flatten()
    Barrier_grid = s_0 * Barrier_multiplier_grid.flatten()

    results = []

    # Calculate required outputs for a single triplet
    def calculate_for_one_triplet(T, K, Barrier):

        call_price = option_price_calc(times, paths, T, t, r, K)
        UIbarrier_price, _ = barrier_upAndIn_price_calc(times, paths, T, t, r, K, Barrier)
        UObarrier_price, _ = barrier_upAndOut_price_calc(times, paths, T, t, r, K, Barrier)
        implied_vol = implied_volatility(s_0, K, T, r, call_price)
        return(T,K,Barrier,kappa,theta,xi,rho,implied_vol,UIbarrier_price,UObarrier_price,call_price)

    # Use ThreadPoolExecutor to parallelize calculations for each (T, K, Barrier) triplet
    # This was used to utilise all cores available, to reduce computation times (ie. multi-threading)
    # This goes through all the triplets and makes the necessary calculations
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_for_one_triplet, T, K, Barrier)
                  for T, K, Barrier in zip(T_grid.flatten(), K_grid, Barrier_grid)]
        for future in futures:
            results.append(future.result())

    return results


# Constants
r = 0.08
s_0 = 1
v_0 = 0.2 
t = 0

# Prices calculated daily intervals, and 10k paths per simulation
delta_t = 1 / 365
n_iterations = int(1.5/ delta_t)
n_paths = 10000

# Generate valid parameter combinations
volSurface_length = 5

# Choosing appropriate Heston parameter ranges
combo_length = 6
kappa_range = np.linspace(0.1, 3, combo_length)
theta_range = np.linspace(0.01, 0.5, combo_length)
xi_range = np.linspace(0.1, 1, combo_length)
rho_range = np.linspace(-1, 1, combo_length)

# Generate params list using our function
valid_params = generate_feller_params(kappa_range, theta_range, xi_range)

# Form combinations with rho, ensuring all are unique
param_combinations = {(kappa, theta, xi, rho) for kappa, theta, xi in valid_params for rho in rho_range}

# Convert set to list for processing
param_combinations = list(param_combinations)

# Possiblity to repeat simulations for given params and (T,K,B), multiple times
repetitions = 1
repeated_combinations = param_combinations * repetitions


# Print the number of rows
# Used for general tracking and visibility
print(f"Total number of rows:                               {len(repeated_combinations*volSurface_length**3)}")
print(f"Total number of simulations ran:                    {int(len(repeated_combinations))}")
print(f"Number of unique heston-volSurface-barrier combos:  {len(param_combinations*volSurface_length**3)}")


# Split the valid combinations into five arrays
num_splits = 5
splits = split_combinations(param_combinations, num_splits)

# Example usage: Assign each split to a separate variable
split1, split2, split3, split4, split5 = splits

pool = Pool()
# Run the first batch simulations
# The parameter "split_x" was changed to run each of the splits separately and capture results 
# As mentioned this was due to RAM issue
paths_results = pool.map(simulate_paths_and_run_simulation, split5)
pool.close()
pool.join()

pool = Pool()
# Run the second batch option price calculations
option_prices_results = pool.map(calculate_option_prices_for_params, paths_results)
pool.close()
pool.join()

# Gather all the results in the form we require for our Neural Network training 
all_results = [item for sublist in option_prices_results for item in sublist]
nn_results = [(item[0], item[1], item[2], item[7],item[8], item[9], item[10]) for sublist in option_prices_results for item in sublist]

# Visual the results using pandas, and save them 
column_names = ['T', 'K', 'B', 'sigma_tk', 'UI Barrier Price', 'UO Barrier Price', 'Call Price']
df = pd.DataFrame([item for item in nn_results], columns=column_names)
display(df)