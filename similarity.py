import numpy as np

def compute_similarity(a,b):
    diff = np.abs(
        a.astype(np.int16) -
        b.astype(np.int16)
    )

    mean_diff = diff.mean() / 255.0

    similarity = 1 - mean_diff * 10

    similarity = max(-1, min(1, similarity))
    return round(float(similarity), 5)