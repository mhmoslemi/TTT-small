"""
Visualize a circle packing produced by a run_packing() function.

Usage:
    1. Paste your run_packing() below where indicated
    2. python plot_packing.py
       (or python plot_packing.py output.png   to save instead of display)

Validates the packing the same way reward.py does and prints any issues.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
import numpy.random as npr


# ====================================================================
# >>> PASTE YOUR run_packing() HERE <<<
# ====================================================================

import numpy as np
import scipy.optimize as optimize
import random

def generate_hexagonal_positions(n, spacing, offset_x, offset_y):
    positions = []
    for i in range(n):
        row = i // 5
        col = i % 5
        x = col * spacing + offset_x
        y = row * spacing + (col % 2) * spacing * 0.5 + offset_y
        positions.append([x, y])
    return np.array(positions)

def generate_zigzag_positions(n):
    positions = []
    for i in range(n):
        row = i // 5
        col = i % 5
        base_x = col * 0.2 + (0.3 if col == 0 or col == 4 else 0.0)
        base_y = row * 0.2 + (0.3 if row == 0 or row == 4 else 0.0)
        x = base_x
        y = base_y + (0.02 if (row + col) % 2 == 1 else 0.0)
        positions.append([x, y])
    return np.array(positions)

def generate_adaptive_positions(n):
    positions = []
    for i in range(n):
        row = i // 5
        col = i % 5
        base_x = col * 0.2 + (0.3 if col == 0 or col == 4 else 0.0)
        base_y = row * 0.2 + (0.3 if row == 0 or row == 4 else 0.0)
        x = base_x + (0.02 if row < 2 or row > 3 else 0.0)
        y = base_y + (0.02 if col < 2 or col > 3 else 0.0)
        positions.append([x, y])
    return np.array(positions)

def generate_random_positions(n):
    positions = np.random.rand(n, 2)
    return positions

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    edge_radius = 0.08
    initial_guesses = [
        (generate_hexagonal_positions(n, 0.23, 0.1, 0.1), np.full(n, 0.04)),
        (generate_zigzag_positions(n), np.full(n, 0.04)),
        (generate_adaptive_positions(n), np.full(n, 0.04)),
        (generate_random_positions(n), np.full(n, 0.04)),
    ]

    best_sum = 0.0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)

    for i, (positions, radii) in enumerate(initial_guesses):
        # Scale the initial positions and radii to fit in the unit square
        scale = min(1.0 / positions.max(), 1.0 / positions.min())
        scaled_positions = positions * scale
        scaled_radii = radii * scale

        # Add weighting to circles near edges and corners to encourage optimal packing
        edge_weights = np.zeros(n)
        corner_weights = np.zeros(n)
        for j in range(n):
            x, y = scaled_positions[j]
            if x < edge_radius or x > 1 - edge_radius or y < edge_radius or y > 1 - edge_radius:
                edge_weights[j] = 1.5
            if (x < edge_radius and y < edge_radius) or (x < edge_radius and y > 1 - edge_radius) or \
               (x > 1 - edge_radius and y < edge_radius) or (x > 1 - edge_radius and y > 1 - edge_radius):
                corner_weights[j] = 1.8

        scaled_radii *= edge_weights * corner_weights
        scaled_radii = np.clip(scaled_radii, 0, 1.0)

        def objective(x):
            centers = x[:2*n].reshape(n, 2)
            radii = x[2*n:]
            edge_bias = 0.0
            for j in range(n):
                x_j, y_j = centers[j]
                r_j = radii[j]
                if x_j < edge_radius or x_j > 1 - edge_radius or y_j < edge_radius or y_j > 1 - edge_radius:
                    edge_bias += r_j
            return -np.sum(radii) - edge_bias * 0.2

        def boundary_constraint(x):
            centers = x[:2*n].reshape(n, 2)
            radii = x[2*n:]
            left = centers[:, 0] - radii - 1e-10
            right = 1.0 - centers[:, 0] - radii - 1e-10
            bottom = centers[:, 1] - radii - 1e-10
            top = 1.0 - centers[:, 1] - radii - 1e-10
            return np.concatenate([left, right, bottom, top])

        def distance_constraint(x):
            centers = x[:2*n].reshape(n, 2)
            radii = x[2*n:]
            dist = np.zeros(n * (n - 1) // 2)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist[idx] = np.sqrt(dx * dx + dy * dy) - radii[i] - radii[j] - 1e-10
                    idx += 1
            return dist

        x0 = np.concatenate([scaled_positions.flatten(), scaled_radii])
        bounds = [(0, 1) for _ in range(2*n)] + [(0, 1) for _ in range(n)]
        cons = [
            {'type': 'ineq', 'fun': boundary_constraint},
            {'type': 'ineq', 'fun': distance_constraint}
        ]

        result_1 = optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            constraints=cons,
            tol=1e-4,
            options={'maxiter': 300}
        )

        result_2 = optimize.minimize(
            objective,
            result_1.x,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            tol=1e-6,
            options={'maxiter': 500}
        )

        centers = result_2.x[:2*n].reshape(n, 2)
        radii = result_2.x[2*n:]
        sum_radii = np.sum(radii)

        valid, msg = validate_packing(centers, radii)
        if not valid:
            print("Invalid packing:", msg)
            continue

        if sum_radii > best_sum:
            best_sum = sum_radii
            best_centers = centers
            best_radii = radii

    valid, msg = validate_packing(best_centers, best_radii)
    if not valid:
        print("Invalid packing:", msg)
        raise ValueError(msg)

    return (best_centers, best_radii, best_sum)

# import numpy as np
# import scipy.optimize as optimize
# import random

# def generate_hexagonal_positions(n, spacing, offset_x, offset_y):
#     positions = []
#     for i in range(n):
#         row = i // 5
#         col = i % 5
#         x = col * spacing + offset_x
#         y = row * spacing + (col % 2) * spacing * 0.5 + offset_y
#         positions.append([x, y])
#     return np.array(positions)

# def generate_nested_positions(n, spacing, offset_x, offset_y):
#     positions = []
#     for i in range(n):
#         level = i // 5
#         offset = i % 5
#         x = offset * spacing + offset_x
#         y = level * spacing + offset_y
#         positions.append([x, y])
#     return np.array(positions)

# def generate_random_positions(n):
#     positions = np.random.rand(n, 2)
#     return positions

# def generate_corner_positions(n):
#     positions = []
#     for i in range(n):
#         row = i // 5
#         col = i % 5
#         x = col * 0.2 + (0.3 if col == 0 or col == 4 else 0.0)
#         y = row * 0.2 + (0.3 if row == 0 or row == 4 else 0.0)
#         positions.append([x, y])
#     return np.array(positions)

# def generate_hybrid_positions(n):
#     positions = []
#     for i in range(n):
#         row = i // 5
#         col = i % 5
#         base_x = col * 0.2 + (0.3 if col == 0 or col == 4 else 0.0)
#         base_y = row * 0.2 + (0.3 if row == 0 or row == 4 else 0.0)
#         x = base_x + (0.02 if row < 2 or row > 3 else 0.0)
#         y = base_y + (0.02 if col < 2 or col > 3 else 0.0)
#         positions.append([x, y])
#     return np.array(positions)

# def generate_zigzag_positions(n):
#     positions = []
#     for i in range(n):
#         row = i // 5
#         col = i % 5
#         base_x = col * 0.2 + (0.3 if col == 0 or col == 4 else 0.0)
#         base_y = row * 0.2 + (0.3 if row == 0 or row == 4 else 0.0)
#         x = base_x
#         y = base_y + (0.02 if (row + col) % 2 == 1 else 0.0)
#         positions.append([x, y])
#     return np.array(positions)

# def generate_corner_aligned_positions(n):
#     positions = []
#     for i in range(n):
#         row = i // 5
#         col = i % 5
#         x = col * 0.2 + (0.3 if col == 0 or col == 4 else 0.0)
#         y = row * 0.2 + (0.3 if row == 0 or row == 4 else 0.0)
#         positions.append([x, y])
#     return np.array(positions)

# def generate_spiral_positions(n):
#     positions = []
#     for i in range(n):
#         angle = 2 * np.pi * i / n
#         radius = 0.2 + 0.1 * (i % 5)
#         x = 0.5 + radius * np.cos(angle)
#         y = 0.5 + radius * np.sin(angle)
#         positions.append([x, y])
#     return np.array(positions)

# def generate_radial_positions(n):
#     positions = []
#     for i in range(n):
#         angle = 2 * np.pi * i / n
#         radius = 0.2 + 0.1 * (i % 5)
#         x = 0.5 + radius * np.cos(angle)
#         y = 0.5 + radius * np.sin(angle)
#         positions.append([x, y])
#     return np.array(positions)

# def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
#     n = 26
#     edge_radius = 0.08
#     num_iterations = 5

#     def objective(x):
#         centers = x[:2*n].reshape(n, 2)
#         radii = x[2*n:]
#         edge_bias = 0.0
#         for j in range(n):
#             x_j, y_j = centers[j]
#             r_j = radii[j]
#             if x_j < edge_radius or x_j > 1 - edge_radius or y_j < edge_radius or y_j > 1 - edge_radius:
#                 edge_bias += r_j
#         return -np.sum(radii) - edge_bias * 0.2

#     def boundary_constraint(x):
#         centers = x[:2*n].reshape(n, 2)
#         radii = x[2*n:]
#         left = centers[:, 0] - radii - 1e-8
#         right = 1.0 - centers[:, 0] - radii - 1e-8
#         bottom = centers[:, 1] - radii - 1e-8
#         top = 1.0 - centers[:, 1] - radii - 1e-8
#         return np.concatenate([left, right, bottom, top])

#     def distance_constraint(x):
#         centers = x[:2*n].reshape(n, 2)
#         radii = x[2*n:]
#         dist = np.zeros(n * (n - 1) // 2)
#         idx = 0
#         for i in range(n):
#             for j in range(i + 1, n):
#                 dx = centers[i, 0] - centers[j, 0]
#                 dy = centers[i, 1] - centers[j, 1]
#                 dist[idx] = np.sqrt(dx * dx + dy * dy) - radii[i] - radii[j] - 1e-8
#                 idx += 1
#         return dist

#     initial_guesses = [
#         (generate_hexagonal_positions(n, 0.25, 0.1, 0.1), np.full(n, 0.04)),
#         (generate_nested_positions(n, 0.25, 0.1, 0.1), np.full(n, 0.04)),
#         (generate_random_positions(n), np.full(n, 0.04)),
#         (generate_hybrid_positions(n), np.full(n, 0.04)),
#         (generate_corner_positions(n), np.full(n, 0.04)),
#         (generate_zigzag_positions(n), np.full(n, 0.04)),
#         (generate_corner_aligned_positions(n), np.full(n, 0.04)),
#         (generate_spiral_positions(n), np.full(n, 0.04)),
#         (generate_radial_positions(n), np.full(n, 0.04)),
#     ]

#     best_sum = 0.0
#     best_centers = np.zeros((n, 2))
#     best_radii = np.zeros(n)

#     for i, (positions, radii) in enumerate(initial_guesses):
#         scale = min(1.0 / positions.max(), 1.0 / positions.min())
#         scaled_positions = positions * scale
#         scaled_radii = radii * scale

#         edge_weights = np.zeros(n)
#         for j in range(n):
#             x_j, y_j = scaled_positions[j]
#             if x_j < edge_radius or x_j > 1 - edge_radius or y_j < edge_radius or y_j > 1 - edge_radius:
#                 edge_weights[j] = 1.5 + 1.5 * (1 - np.min([x_j, 1 - x_j, y_j, 1 - y_j]) / edge_radius)
#         scaled_radii *= edge_weights
#         scaled_radii = np.clip(scaled_radii, 0, 1.0)

#         x0 = np.concatenate([scaled_positions.flatten(), scaled_radii])
#         bounds = [(0, 1) for _ in range(2*n)] + [(0, 1) for _ in range(n)]
#         cons = [
#             {'type': 'ineq', 'fun': boundary_constraint},
#             {'type': 'ineq', 'fun': distance_constraint}
#         ]

#         result_1 = optimize.minimize(
#             objective,
#             x0,
#             method='L-BFGS-B',
#             bounds=bounds,
#             constraints=cons,
#             tol=1e-4
#         )

#         result_2 = optimize.minimize(
#             objective,
#             result_1.x,
#             method='SLSQP',
#             bounds=bounds,
#             constraints=cons,
#             tol=1e-6
#         )

#         centers = result_2.x[:2*n].reshape(n, 2)
#         radii = result_2.x[2*n:]
#         sum_radii = np.sum(radii)

#         valid, msg = validate_packing(centers, radii)
#         if not valid:
#             print("Invalid packing:", msg)
#             continue

#         if sum_radii > best_sum:
#             best_sum = sum_radii
#             best_centers = centers
#             best_radii = radii

#     valid, msg = validate_packing(best_centers, best_radii)
#     if not valid:
#         print("Invalid packing:", msg)
#         raise ValueError(msg)

#     return (best_centers, best_radii, best_sum)
# ====================================================================
# Validation (same as reward.py)
# ====================================================================
def validate_packing(centers, radii):
    """
    Paper-compatible signature: returns (bool, str).
    Matches reward.py so model code calling validate_packing(...) works.
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values present"
    for i in range(n):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius {radii[i]}"
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12
                or y - r < -1e-12 or y + r > 1 + 1e-12):
            return False, f"Circle {i} at ({x},{y}) r={r} outside unit square"
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
            if dist < radii[i] + radii[j] - 1e-12:
                return False, f"Circles {i} and {j} overlap"
    return True, "ok"


def collect_issues(centers, radii, tol=1e-9):
    """Plotter-only helper. Returns a list of strings, one per problem found."""
    issues = []
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        issues.append("NaN values present")
    for i in range(n):
        if radii[i] < 0:
            issues.append(f"Circle {i} has negative radius {radii[i]:.6f}")
        x, y = centers[i]
        r = radii[i]
        if (x - r < -tol or x + r > 1 + tol
                or y - r < -tol or y + r > 1 + tol):
            issues.append(f"Circle {i} at ({x:.4f},{y:.4f}) r={r:.4f} outside unit square")
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
            gap = dist - (radii[i] + radii[j])
            if gap < -tol:
                issues.append(f"Circles {i},{j} overlap (gap={gap:.6f})")
    return issues


# ====================================================================
# Plotting
# ====================================================================
def plot_packing(centers, radii, sum_radii, save_to=None):
    n = len(radii)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Unit square
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, linewidth=1.5, edgecolor="black"))

    # Color by radius so it's easy to see who's big and who's tiny
    cmap = plt.get_cmap("viridis")
    rmax = max(radii.max(), 1e-9)
    issues = collect_issues(centers, radii)
    invalid_ids = set()
    for msg in issues:
        # Best-effort: pull circle indices out of error messages so we can flag them
        for tok in msg.replace(",", " ").split():
            if tok.isdigit():
                invalid_ids.add(int(tok))

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        color = cmap(r / rmax)
        edge = "red" if i in invalid_ids else "black"
        lw = 1.5 if i in invalid_ids else 0.5
        ax.add_patch(patches.Circle((x, y), r,
                                     facecolor=color, edgecolor=edge,
                                     linewidth=lw, alpha=0.65))
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=8, color="white", weight="bold")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True, alpha=0.3)

    title = f"n={n}  sum of radii = {sum_radii:.10f}, SOTA = 2.635983"
    if issues:
        title += f"  [INVALID: {len(issues)} issue(s)]"
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_to}")
    else:
        plt.show()


def main():
    centers, radii, sum_radii = run_packing()
    centers = np.asarray(centers)
    radii = np.asarray(radii).ravel()

    print(f"n = {centers.shape[0]}")
    print(f"sum of radii (returned)  = {sum_radii:.6f}")
    print(f"sum of radii (recomputed)= {radii.sum():.6f}")
    print(f"radii: min={radii.min():.4f} max={radii.max():.4f} mean={radii.mean():.4f}")

    issues = collect_issues(centers, radii)
    if issues:
        print(f"\nVALIDATION FAILED: {len(issues)} issue(s)")
        for msg in issues[:10]:
            print(f"  - {msg}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("\nValidation OK.")

    save_to = sys.argv[1] if len(sys.argv) > 1 else None
    plot_packing(centers, radii, sum_radii, save_to='out.png')


if __name__ == "__main__":
    main()