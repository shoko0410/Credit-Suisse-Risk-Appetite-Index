import numpy as np
import pandas as pd

def calculate_slope_old(x, y):
    """Original logic: Correlation"""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0
    x_norm = (x - np.mean(x)) / np.std(x)
    y_norm = (y - np.mean(y)) / np.std(y)
    cov_matrix = np.cov(x_norm, y_norm)
    return cov_matrix[0, 1]

def calculate_slope_new(x, y):
    """New logic: Regression Slope (Beta)"""
    if np.std(x) == 0 or np.std(y) == 0:
        return 0
    
    # Method 1: np.polyfit
    slope, intercept = np.polyfit(x, y, 1)
    
    # Method 2: Cov(x,y) / Var(x)
    # cov = np.cov(x, y)[0, 1]
    # var = np.var(x, ddof=1)
    # slope = cov / var
    
    return slope

def test_logic():
    print("Testing Slope Calculation Logic...")
    
    # Case 1: Perfect linear relationship y = 2x
    # Slope should be 2.0
    # Correlation should be 1.0
    x1 = np.array([1, 2, 3, 4, 5])
    y1 = 2 * x1
    
    old_val1 = calculate_slope_old(x1, y1)
    new_val1 = calculate_slope_new(x1, y1)
    
    # Case 1 (y = 2x)
    print(f"\nCase 1 (y = 2x):")
    print(f"  New Logic (Slope):       {new_val1:.4f} (Expected ~2.0)")
    assert abs(new_val1 - 2.0) < 1e-5, "New logic failed to calculate slope"

    # Case 2 (y = -0.5x + 1)
    x2 = np.array([1, 2, 3, 4, 5])
    y2 = -0.5 * x2 + 1
    new_val2 = calculate_slope_new(x2, y2)

    print(f"\nCase 2 (y = -0.5x + 1):")
    print(f"  New Logic (Slope):       {new_val2:.4f} (Expected ~-0.5)")
    assert abs(new_val2 - (-0.5)) < 1e-5, "New logic failed to calculate slope"

    # Case 3 (y = 1.5x + noise)
    np.random.seed(42)
    x3 = np.linspace(0, 10, 100)
    y3 = 1.5 * x3 + np.random.normal(0, 1, 100)
    new_val3 = calculate_slope_new(x3, y3)

    print(f"\nCase 3 (y = 1.5x + noise):")
    print(f"  New Logic (Slope):       {new_val3:.4f} (Expected ~1.5)")
    assert abs(new_val3 - 1.5) < 0.2, f"New logic slope {new_val3} too far from 1.5"

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_logic()
