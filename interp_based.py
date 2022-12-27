import numpy as np


def interp_embed(mat, wm):
    interp_mat = np.zeros(mat.shape)

    for i in range(0, mat.shape[0], 2):
        for j in range(0, mat.shape[1], 2):
            interp_mat[i, j, :] = mat[i, j, :]
            if (i + 2) < mat.shape[0]:
                interp_mat[i + 1, j, :] = (mat[i + 2, j, :] + mat[i, j, :]) * 0.5
            if (j + 2) < mat.shape[1]:
                interp_mat[i, j + 1, :] = (mat[i, j + 2, :] + mat[i, j, :]) * 0.5
            if ((i + 2) < mat.shape[0]) and ((j + 2) < mat.shape[1]):
                interp_mat[i + 1, j + 1, :] = (mat[i, j, :] + mat[i + 1, j, :] + mat[i, j + 1, :]) / 3

    new_mat = np.zeros(mat.shape)
    count = 0
    embed=[]
    for k in range(0, interp_mat.shape[2]):
        for i in range(0, interp_mat.shape[0], 2):
            for j in range(0, interp_mat.shape[1], 2):
                d = np.zeros(3)
                n = np.zeros(3)
                if ((i+1) < interp_mat.shape[0]) and ((j+1) < interp_mat.shape[1]):
                    d[0] = np.abs(interp_mat[i, j, k] - interp_mat[i + 1, j, k])
                    d[1] = np.abs(interp_mat[i, j, k] - interp_mat[i, j + 1, k])
                    d[2] = np.abs(interp_mat[i, j, k] - interp_mat[i + 1, j + 1, k])
                    for ii in range(len(d)):
                        if d[ii] >= 1:
                            n[ii] = np.floor(np.log2(d[ii]))
                        else:
                            n[ii] = 0
                    new_mat[i, j, k] = interp_mat[i, j, k]

                    if n[0] != 0:
                        tmp = wm[int(count):int(count + n[0])]
                        tmp = int(''.join(str(x) for x in tmp), 2)
                        embed.append(tmp)
                        new_mat[i + 1, j, k] = interp_mat[i + 1, j, k] + tmp
                    else:
                        new_mat[i + 1, j, k] = interp_mat[i + 1, j, k]

                    if n[1] != 0:
                        tmp = wm[int(count + n[0]):int(count + n[0] + n[1])]
                        tmp = int(''.join(str(x) for x in tmp), 2)
                        embed.append(tmp)
                        new_mat[i, j + 1, k] = interp_mat[i, j + 1, k] + tmp
                    else:
                        new_mat[i, j + 1, k] = interp_mat[i, j + 1, k]
                    if n[2] != 0:
                        tmp = wm[int(count + n[0] + n[1]):int(count + n[0] + n[1] + n[2])]
                        tmp = int(''.join(str(x) for x in tmp), 2)
                        embed.append(tmp)
                        new_mat[i + 1, j + 1, k] = interp_mat[i + 1, j + 1, k] + tmp
                    else:
                        new_mat[i + 1, j + 1, k] = interp_mat[i + 1, j + 1, k]
                    count += int(n[0] + n[1] + n[2])

    return new_mat,embed


def interp_extract(cw,row):
    new_mat = np.zeros(cw.shape)
    extr_row=[]
    for i in range(0, cw.shape[0], 2):
        for j in range(0, cw.shape[1], 2):
            new_mat[i, j, :] = cw[i, j, :]
            if (i + 2) < cw.shape[0]:
                tmp=np.floor((cw[i + 2, j, :] + cw[i, j, :]) * 0.5)
                extr_row.append(cw[i + 1, j, :] - ((cw[i + 2, j, :] + cw[i, j, :]) * 0.5))
            if (j + 2) < cw.shape[1]:
                tmp2= np.floor((cw[i, j + 2, :] + cw[i, j, :]) * 0.5)
                extr_row.append(cw[i, j + 1, :] - ((cw[i, j + 2, :] + cw[i, j, :]) * 0.5))
            if ((i + 2) < cw.shape[0]) and ((j + 2) < cw.shape[1]):
                tmp3= np.floor(
                        (2*cw[i, j, :] + (cw[i + 2, j, :] + cw[i, j + 2, :])*0.5) / 3)
                extr_row.append(cw[i + 1, j + 1, :] - (
                        (2*cw[i, j, :] + (cw[i + 2, j, :] + cw[i, j + 2, :])*0.5) / 3))

    return extr_row
