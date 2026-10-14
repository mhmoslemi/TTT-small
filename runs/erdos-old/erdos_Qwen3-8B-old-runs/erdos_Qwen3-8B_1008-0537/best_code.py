import numpy as np
import scipy.optimize as optimize
import time

def run(seed=42, budget_s=100, **kwargs):
    np.random.seed(seed)
    n_points = len(initial_h_values)
    target_sum = n_points / 2
    dx = 2.0 / n_points
    h = np.array(initial_h_values, dtype=float)
    
    def objective(h_vals):
        corr = np.correlate(h_vals, 1 - h_vals, mode='full')
        return np.max(corr) * dx
    
    def constraint_sum(h_vals):
        return np.sum(h_vals) - target_sum
    
    bounds = [(0.0, 1.0) for _ in range(n_points)]
    
    best_h = h.copy()
    best_c5 = objective(best_h)
    start_time = time.time()
    
    def generate_periodic(period):
        return np.clip(0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n_points) / period), 0, 1)
    
    def generate_binary(freq, duty):
        return np.clip(np.sin(2 * np.pi * freq * np.arange(n_points)) > 0, 0, 1) * duty
    
    def generate_triangular(freq, amp):
        t = np.linspace(0, 1, n_points)
        wave = amp * (2 * t - 1) * np.abs(1 - 2 * np.abs(2 * freq * t - 1))
        return np.clip(wave, 0, 1)
    
    def generate_sawtooth(freq, duty):
        t = np.linspace(0, 1, n_points)
        wave = 2 * duty * (1 - np.abs(2 * freq * t - 1))
        return np.clip(wave, 0, 1)
    
    guesses = [generate_periodic(p) for p in [3, 4, 5, 6, 7]]
    guesses += [generate_binary(f, d) for f, d in [(5, 0.3), (6, 0.2), (3, 0.4)]]
    guesses += [generate_triangular(f, a) for f, a in [(4, 0.6), (3, 0.5)]]
    guesses += [generate_sawtooth(f, d) for f, d in [(4, 0.25), (5, 0.2)]]
    guesses.append(np.random.rand(n_points))
    guesses.append(h)
    
    for guess in guesses:
        res = optimize.minimize(objective, guess, method='SLSQP', bounds=bounds,
                               constraints=[{'type': 'eq', 'fun': constraint_sum}],
                               options={'maxiter': 150, 'ftol': 1e-12, 'eps': 1e-10})
        if res.success and objective(res.x) < best_c5:
            best_c5 = objective(res.x)
            best_h = res.x
    
    for _ in range(4):
        if time.time() - start_time >= budget_s - 20:
            break
        res = optimize.differential_evolution(objective, bounds, popsize=50, maxiter=25,
                                             mutation=(0.8, 0.95), recombination=0.9, tol=1e-9)
        if res.success:
            res_slsqp = optimize.minimize(objective, res.x, method='SLSQP', bounds=bounds,
                                         constraints=[{'type': 'eq', 'fun': constraint_sum}],
                                         options={'maxiter': 50, 'ftol': 1e-12, 'eps': 1e-10})
            if res_slsqp.success and objective(res_slsqp.x) < best_c5:
                best_c5 = objective(res_slsqp.x)
                best_h = res_slsqp.x
    
    def refine_with_powell(h_init):
        res = optimize.minimize(objective, h_init, method='Powell', bounds=bounds,
                               constraints=[{'type': 'eq', 'fun': constraint_sum}],
                               options={'xtol': 1e-9, 'ftol': 1e-9, 'maxiter': 30})
        return res.x if res.success else h_init
    
    for _ in range(3):
        if time.time() - start_time >= budget_s - 10:
            break
        best_h = refine_with_powell(best_h)
        if objective(best_h) < best_c5:
            best_c5 = objective(best_h)
    
    return best_h, best_c5, n_points