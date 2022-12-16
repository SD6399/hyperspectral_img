import numpy as np


def inrerp_embed(mat):
    count_canal = mat.shape[2]
    new_mat = np.zeros(mat.shape)
    for k in range(count_canal):
        for i in range(0, mat.shape[0], 2):
            for j in range(0, mat.shape[1], 2):
                new_mat[i, j, k] = mat[i, j, k]
                if (i + 2) < mat.shape[0]:
                    new_mat[i + 1, j, k] = (mat[i + 2, j, k] + mat[i, j, k]) * 0.5
                if (j + 2) < mat.shape[1]:
                    new_mat[i, j + 1, k] = (mat[i, j + 2, k] + mat[i, j, k]) * 0.5
                if ((j + 2) < mat.shape[0]) and ((j + 2) < mat.shape[1]):
                    new_mat[i + 1, j + 1, k] = (mat[i, j, k] + mat[i + 1, j, k] + mat[i, j + 1, k]) / 3

    return new_mat
