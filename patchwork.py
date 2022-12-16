import math

import numpy as np
from statistics import mean


def patchwork_embed(X, wm, l,  seed):

    data_length = X.shape[0]
    size_part= math.ceil(X.shape[0] / l)
    shuf_order = np.arange(data_length)
    np.random.seed(seed)
    np.random.shuffle(shuf_order)

    pix_val = np.zeros((l, size_part, X.shape[1]))

    count = -1
    for i in range(l):
        for j in range(size_part):
            count += 1
            for cnl in range(X.shape[1]):
                if count < X.shape[0]:
                    pix_val[i, j, cnl] = X[shuf_order[count],cnl]

    count=0
    rn = np.array([x for x in range(0, size_part)])
    np.random.seed(seed)
    np.random.shuffle(rn)
    s1 = sorted(rn[:int(rn.size / 2)])
    s2 = sorted(rn[int(rn.size / 2):])
    for i in range(l):
        for j in range(size_part):
                if j in s1:
                    pix_val[i,j,:] += wm[count:count+X.shape[1]]
                if j in s2:
                    pix_val[i,j,:] -= wm[count:count+X.shape[1]]
                count+=X.shape[1]

    orig_matr=np.array([])
    for cnc in range(l):
        if cnc==0:
            orig_matr=pix_val[0,:,:]
        else:
            orig_matr = np.append(orig_matr,pix_val[cnc,:,:],axis=0)
    ost= X.shape[0] % l
    if ost!=0:
        shuf_matr=orig_matr[:-(l-ost),:]
    else:
        shuf_matr= orig_matr
    unshuf_order0 = np.zeros_like(shuf_order)
    unshuf_order0[shuf_order] = np.arange(len(shuf_order))

    orig_matr = np.zeros(X.shape)
    for i in range(X.shape[0]):
        orig_matr[i,:] = shuf_matr[unshuf_order0[i],:]

    return orig_matr


def patchwork_extract(cw,l,seed):
    data_length = cw.shape[0]
    size_part = math.ceil(cw.shape[0] / l)
    shuf_order = np.arange(data_length)
    np.random.seed(seed)
    np.random.shuffle(shuf_order)
    pix_val = np.zeros((l, size_part, cw.shape[1]))

    count = -1
    for i in range(l):
        for j in range(size_part):
            count += 1
            for cnl in range(176):
                if count < cw.shape[0]:
                    pix_val[i, j, cnl] = cw[shuf_order[count], cnl]

    diff_avg = []
    rn = np.array([x for x in range(0, size_part)])
    np.random.seed(seed)
    np.random.shuffle(rn)
    s1 = sorted(rn[:int(rn.size / 2)])
    s2 = sorted(rn[int(rn.size / 2):])

    for i in range(l):
        sum1 = 0
        sum2 = 0
        for j in range(size_part):
            for cnl in range(cw.shape[1]):

                if j in s1:
                    sum1 += pix_val[i, j, cnl]

                if j in s2:
                    sum2 += pix_val[i, j, cnl]

                avg1 = sum1 / len(s1)
                avg2 = sum2 / len(s2)
                diff_avg.append(avg1 - avg2)


    print(mean(diff_avg))
    return mean(diff_avg)


