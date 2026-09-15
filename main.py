import numpy as np
import pandas as pd

# 1. Sun'iy dataset
df = pd.DataFrame(
    {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],
        "C": [3, 6, 9, 12, 15],  # A + B
        "D": [5, 7, 8, 10, 12],
        "E": [10, 14, 16, 20, 24],  # 2 * D
        "F": [2, 4, 6, 8, 10],  # 2 * A
    }
)

print("Asl dataset:")
print(df)


# 2. Matrix rank
X = df.values

rank = np.linalg.matrix_rank(X)

print("\nMatrix rank:", rank)
print("Ustunlar soni:", X.shape[1])


# 3. Bog'liq ustunlarni topish

print("\nBog'liqliklar:")

print("C = A + B")
print("F = 2 * A")
print("E = 2 * D")


# 4. Ortiqcha ustunlarni tashlash
df_clean = df.drop(columns=["C", "E", "F"])

print("\nToza dataset:")
print(df_clean)

print("\nEski shape:", df.shape)
print("Yangi shape:", df_clean.shape)
