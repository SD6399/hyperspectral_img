import math

import numpy as np
from statistics import mean


def patchwork_embed(X, wm,alfa, l, seed):

    size_part= math.ceil(X.size / l)
    shuf_order = np.arange(X.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuf_order)

    np.random.seed(seed)
    shuff_matr = X[shuf_order, :]

    pix_val = np.resize(shuff_matr,(l,size_part,X.shape[1]))

    rn = np.array([x for x in range(0, size_part)])
    np.random.seed(seed)
    np.random.shuffle(rn)
    s1 = sorted(rn[:int(rn.size / 2)])
    s2 = sorted(rn[int(rn.size / 2):])

    pix_val_aft_emb = np.zeros(pix_val.shape)
    count = 0
    for j in range(size_part):
        #for k in range(pix_val.shape[2]):
        if j in s1:
            pix_val_aft_emb[:,j,:] = pix_val[:,j,:] + wm[count:count+X.shape[1]]*alfa
            count += X.shape[1]
        if j in s2:
            pix_val_aft_emb[:,j,:] = pix_val[:,j,:] - wm[count:count+X.shape[1]]*alfa
            count += X.shape[1]

    #rav_after_emb= np.ravel(pix_val_aft_emb)[:len(rav_X)]

    unshuf_order0 = np.zeros_like(shuf_order)
    unshuf_order0[shuf_order] = np.arange(len(shuf_order))

    orig_matr = np.resize(pix_val_aft_emb,X.shape)

    orig_matr = orig_matr[unshuf_order0, :]

    print("embed")
    return orig_matr


def patchwork_extract(cw,l,seed,alf):
    data_length = len(np.ravel(cw))
    size_part = math.ceil(cw.size / l)
    shuf_order = np.arange(data_length)
    np.random.seed(seed)
    np.random.shuffle(shuf_order)
    pix_val = np.zeros((l, size_part))

    rav_X=np.ravel(cw)

    count=-1
    for i in range(l):
        for j in range(size_part):
            count += 1
            if count < data_length:
                pix_val[i, j] = rav_X[shuf_order[count]]

    #pix_val = np.resize(cw, (l, size_part))

    rn = np.array([x for x in range(0, size_part)])
    np.random.seed(seed)
    np.random.shuffle(rn)
    s1 = sorted(rn[:int(rn.size / 2)])
    s2 = sorted(rn[int(rn.size / 2):])
    #for i in range(l):
    sum1 = np.zeros(l)
    sum2 = np.zeros(l)
    for j in range(size_part):
        if j in s1:
            sum1 += pix_val[:,j]
        if j in s2:
            sum2 += pix_val[:,j]

    avg1 = sum1 / len(s1)
    avg2 = sum2 / len(s2)
    diff_avg=(avg1 - avg2)

    cnt_need=(diff_avg > (alf/2)).sum()

    print(cnt_need/l)
    return cnt_need/l


