import numpy as np
from load_datasets import create_unbalance
from ours.external_scores import adjusted_rand_index_python, adjusted_rand_index_numpy, balanced_external


if __name__ == "__main__":
    data, y_true = create_unbalance()

    y_pred = np.copy(y_true)
    y_pred[y_pred > 3] = 3
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

    print("\nComparison with sklearn:")
    print("=" * 60)
    print("Adjusted Rand Index")
    print("=" * 60)
    print(f"Python implementation:                      {adjusted_rand_index_python(y_true, y_pred):.6f}")
    print(f"Numpy implementation:                       {adjusted_rand_index_numpy(y_true, y_pred):.6f}")
    print(f"Sklearn implementation:                     {adjusted_rand_score(y_true, y_pred):.6f}")
    print()
    print(f"Balanced (numpy) implementation:            {balanced_external(adjusted_rand_index_numpy, y_true, y_pred):.6f}")
    print(f"Balanced (sklearn) implementation:          {balanced_external(adjusted_rand_score, y_true, y_pred, method='macro'):.6f}")

    print(f"Balanced (sklearn) implementation [CHECK]:  {balanced_external(adjusted_rand_score, y_true, y_true, method='macro'):.6f}")

    print("=" * 60)
    print("Adjusted Mutual Information")
    print("=" * 60)
    print(f"Sklearn implementation:                     {adjusted_mutual_info_score(y_true, y_pred):.6f}")
    print()
    print(f"Balanced (sklearn) implementation:          {balanced_external(adjusted_mutual_info_score, y_true, y_pred, method='macro'):.6f}")

    print(f"Balanced (sklearn) implementation [CHECK]:  {balanced_external(adjusted_mutual_info_score, y_true, y_true, method='macro'):.6f}")
