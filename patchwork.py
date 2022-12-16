import math

import numpy as np
from statistics import mean


def patchwork_embed(mat, wm, l, list_gt, seed):
    # переделать под норм передачу контейнера

    data_length = len(list_gt)
    size_part= math.ceil(len(list_gt) / l)
    shuf_order = np.arange(data_length)
    np.random.seed(seed)
    np.random.shuffle(shuf_order)

    gt0 = []
    for i in list_gt:
        gt0.append(i[0])

    gt1 = []
    for i in list_gt:
        gt1.append(i[1])
    shuffled_data0 = np.array(gt0)[shuf_order]
    shuffled_data1 = np.array(gt1)[shuf_order]

    pix_val = np.zeros((l, size_part, mat.shape[1]))

    count = -1
    for i in range(l):
        for j in range(size_part):
            count += 1
            for cnl in range(176):
                if count < len(list_gt):
                    pix_val[i, j, cnl] = mat[shuffled_data0[count], shuffled_data1[count], cnl]

    count=0
    rn = np.array([x for x in range(1, size_part)])
    np.random.seed(seed)
    np.random.shuffle(rn)
    s1 = rn[:int(rn.size / 2)]
    s2 = rn[int(rn.size / 2):]
    for i in range(pix_val.shape[0]):
        for j in range(size_part):

            for cnl in range(176):
                if j in s1:
                    pix_val[i,j,cnl] += wm[count]

                if j in s2:
                    pix_val[i,j,cnl] -= wm[count]
                count+=1

    orig_matr=np.array([])
    for cnc in range(l):
        if cnc==0:
            orig_matr=pix_val[0,:,:]
        else:
            orig_matr = np.append(orig_matr,pix_val[cnc,:,:],axis=0)
    ost= len(list_gt) % l
    if ost!=0:
        shuf_matr=orig_matr[:-(l-ost),:]
    else:
        shuf_matr= orig_matr
    unshuf_order0 = np.zeros_like(shuf_order)
    unshuf_order1 = np.zeros_like(shuf_order)
    unshuf_order0[shuf_order] = np.arange(len(shuffled_data1))

    orig_matr = np.zeros((5211,176))
    for i in range(5211):
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
    s1 = rn[:int(rn.size / 2)]
    s2 = rn[int(rn.size / 2):]
    for cnl in range(176):

        for i in range(pix_val.shape[0]):
            sum1 = 0
            sum2 = 0
            for j in range(size_part):

                if j in s1:
                    sum1 += pix_val[i, j, cnl]

                if j in s2:
                    sum2 += pix_val[i, j, cnl]

            avg1 = sum1 / len(s1)
            avg2 = sum2 / len(s2)
            diff_avg.append(avg1 - avg2)

    print(diff_avg)
    print(mean(diff_avg))
    return mean(diff_avg)


