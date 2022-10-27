import numpy as np
import scipy.io
import csv
from PIL import Image
from scipy.ndimage import median_filter
from numpy import savetxt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn import svm
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statistics import mean


def init2matr(name):
    mat = scipy.io.loadmat(name + '_corrected.mat')
    gt = scipy.io.loadmat(name + '_gt.mat')
    shape = mat[name + '_corrected'].shape
    gt_np = gt[name + '_gt']
    mat_np = mat[name + '_corrected']
    mat_np = np.where(mat_np > 65000, 2 ** 16 - mat_np, mat_np)
    mat_np = np.where(mat_np < 0, np.abs(mat_np), mat_np)

    print("min значение пиксела ", np.min(mat_np))
    print("max значение пиксела ", np.max(mat_np))

    return shape, gt_np, mat_np


def classify_RFC(X, y, tst_sz):
    list_acc = []
    feat_list=[]
    for rs in range(40, 45):
        np.random.seed(rs)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst_sz, random_state=rs)

        np.random.seed(rs)
        rfc = RandomForestClassifier(random_state=42,n_estimators=100)
        rfc.fit(X_train, y_train)

        feat_list.append(rfc.feature_importances_)
        y_pred = rfc.predict(X_test)
        list_acc.append(metrics.accuracy_score(y_test, y_pred))
    print(list_acc)
    print("Accuracy RFC: %.3f" % mean(list_acc), name)

    return feat_list


def classify_XGB(X, y, tst_sz):
    list_acc = []
    feat_list = []
    for rs in range(40, 45):

        counter = Counter(y)

        sort_y = sorted(counter)
        new_y = [i for i in range(0, len(sort_y))]

        for i in range(len(sort_y)):
            for j in range(len(y)):
                if y[j] == sort_y[i]:
                    y[j] = new_y[i]

        np.random.seed(rs)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst_sz, random_state=rs)
        np.random.seed(rs)
        model = XGBClassifier(random_state=42,use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train)
        feat_list.append(model.feature_importances_)
        y_pred = model.predict(X_test)
        list_acc.append(metrics.accuracy_score(y_test, y_pred))

    print(list_acc)
    print("Accuracy XGB:  %.3f" % mean(list_acc), name)
    return feat_list


def classify_dec_tree(X, y, tst_sz):
    list_acc = []
    for rs in range(40, 45):
        np.random.seed(rs)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst_sz, random_state=rs)
        np.random.seed(rs)
        classifier = DecisionTreeClassifier(random_state=42)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        list_acc.append(metrics.accuracy_score(y_test, y_pred))
    print(list_acc)
    print("Accuracy Decision Tree: %.3f" % mean(list_acc), name)




def classify_SVM(X, y, tst_sz):
    list_acc = []
    for rs in range(40, 45):
        np.random.seed(rs)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst_sz, random_state=rs)
        np.random.seed(rs)
        classifier = svm.LinearSVC(random_state=rs, verbose=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        list_acc.append(metrics.accuracy_score(y_test, y_pred))
    print(list_acc)
    print("Accuracy SVM: %.3f" % mean(list_acc), name)


def disper_chanel(chn):
    cnt = 0
    disp_list = []

    while cnt < chn:
        tmp = mat_data[:, :, cnt].astype('int64')
        x_quadr = np.mean(tmp ** 2)
        x2 = (np.mean(tmp)) ** 2
        disp_list.append(x_quadr - x2)

        img = Image.fromarray(tmp.astype('uint8'))
        img.convert('RGB').save(r"image" + str(cnt) + ".png")
        cnt += 1

    return disp_list


def disper_noise():
    cnt = 0
    disp_list = []

    while cnt < shape[2]:
        img = mat_data[:, :, cnt].astype('int64')
        filt_img = median_filter(img, size=3)
        diff_img = img - filt_img
        disp_list.append(np.mean(diff_img ** 2) - (np.mean(diff_img)) ** 2)

        cnt += 1

    return disp_list


def old_embed(mat, wm, delt):
    cw = np.zeros((mat.shape[0], mat.shape[1], mat.shape[2]))
    wm1 = wm.reshape(shape[0], shape[1], shape[2])
    for cnt in range(0, shape[2]):
        cw[:, :, cnt] = np.around(
            (np.floor(mat[:, :, cnt] / (2 * delt[cnt])) * 2 * delt[cnt] + wm1[:, :, cnt] * delt[cnt]))

    return cw


def extract(c, cw, delt, ampl):
    w = (cw / delt - np.floor(c / 2 / delt) * 2) / ampl
    return w


def mse(actual, pred):
    return np.square(actual - pred).mean()


def embed_to_pix(cont, wm, delt, vol_wm, count_prior_canal):
    matrix = np.zeros((len(list_gt), shape[2]))

    rng = np.random.default_rng(42)
    random_pix = rng.choice(list_gt, int(len(list_gt) * vol_wm), replace=False).tolist()
    random_pix.sort()

    val=np.array(mfl)
    sort_val= np.array(mfl).sort

    if delt:
        sort_canal = np.array(mfl).argsort()[0:count_prior_canal]
    else:
        sort_canal = []

    for i in range(len(list_gt)):
        if i % 200==0:
            print(i)
        for cnt in range(shape[2]):
            if (cnt in sort_canal) and (list_gt[i] in random_pix):
                tmp = list_gt[i][0]
                tmp2 = list_gt[i][1]
                matrix[i, cnt] = np.around((np.floor(cont[tmp, tmp2, cnt] / (2 * delt[cnt])) * 2 *
                                            delt[cnt] + wm[i * cnt + cnt] * delt[cnt]))
            else:
                tmp = list_gt[i][0]
                tmp2 = list_gt[i][1]
                matrix[i, cnt] = cont[tmp, tmp2, cnt]

    return matrix


def search_alfa(mse_for_compare):
    mse_res1 = 0
    alfa = 0.01
    inc = 0.005

    while mse_res1 < mse_for_compare:
        dsp_noise = disper_noise()
        # dsp_canal=disper_chanel(shape[2])
        for i in range(len(dsp_noise)):
            # dsp_noise[i] = np.sqrt(dsp_canal[i])
            dsp_noise[i] *= alfa
        cw_img = old_embed(mat_data, wm, dsp_noise)
        mse_res1 = mse(mat_data, cw_img)
        print("alfa=", alfa, ":", mse_res1)

        alfa += inc

    return alfa - inc


name = 'indian_pines'

list_gt = []
# alfa=f(mse_for_compare)
# print("alfa=",alfa)
list_mse = []
shape, mat_gt, mat_data, = init2matr(name)
print(shape)
np.random.seed(42)
wm = np.random.randint(0, 2, size=(shape[0] * shape[1] * shape[2]))

for i in range(shape[0]):
    for j in range(shape[1]):
        if mat_gt[i][j] != 0:
            list_gt.append([i, j])

# dsp_noise=disper_noise()
# cw_try_img = embed(mat_data, wm,list_const)


dsp_canal = disper_chanel(shape[2])

# alfa=search_alfa()

# print("yes alfa=",alfa)
# for i in range(len(dsp_noise)):
#      #dsp_canal[i] = np.sqrt(dsp_canal[i])
#      dsp_noise[i]*=0.000367

matrix_must_pix_orig = np.zeros((len(list_gt), shape[2]))
matrix_must_pix = np.zeros((len(list_gt), shape[2]))
y = []
for i in range(len(list_gt)):
    y.append(mat_gt[list_gt[i][0], list_gt[i][1]])

for i in range(len(list_gt)):
    for cnt in range(shape[2]):
        tmp = list_gt[cnt]
        tmp2 = list_gt[cnt][0]

        matrix_must_pix_orig[i, cnt] = mat_data[list_gt[i][0], list_gt[i][1], cnt].astype('int32')

X1 = matrix_must_pix_orig.tolist()
# classify_SVM(X1, y, 0.25)
# classify_RFC(X1,y,0.25)
# classify_dec_tree(X1,y,0.25)
#mean_feat = classify_XGB(X1, y, 0.25)
mean_feat = classify_XGB(X1, y, 0.25)

# with open('KSC_features.csv', 'r') as f:
#     mean_feat = list(csv.reader(f, delimiter=","))
#

mean_feat_list = np.mean(mean_feat, axis=0)
mfl = mean_feat_list.tolist()
#


all_cnl=[50,75,103]

mse_list=[]

for cnl in np.arange(3.8e-05,3.81e-05,1e-08):
    dsp_canal = disper_chanel(shape[2])
    for i in range(len(dsp_canal)):
        dsp_canal[i] *= cnl
        #list_const = [13] * shape[2]
    #for chn in range(40,103,20):
    matrix_must_pix = embed_to_pix(mat_data, wm,dsp_canal,1, 80)
    # cw_img = old_embed(mat_data, wm, dsp_canal)
    #
    # for i in range(len(list_gt)):
    #     for cnt in range(shape[2]):
    #         tmp = list_gt[cnt]
    #         tmp2 = list_gt[cnt][0]
    #
    #         matrix_must_pix[i, cnt] = cw_img[list_gt[i][0], list_gt[i][1], cnt].astype('int32')

    X = matrix_must_pix.tolist()
    classify_XGB(X, y, 0.25)
    mse_list.append(mean_squared_error(matrix_must_pix, matrix_must_pix_orig))
    print("MSE new", mean_squared_error(matrix_must_pix, matrix_must_pix_orig),"for %d channel",cnl)
X = matrix_must_pix.tolist()
print(mse_list)
# classify_SVM(X, y, 0.25)
# classify_RFC(X, y, 0.25)
# classify_dec_tree(X, y, 0.25)
classify_XGB(X, y, 0.25)


#cw_img = old_embed(mat_data, wm, list_const)

# for i in range(len(list_gt)):
#     for cnt in range(shape[2]):
#         tmp = list_gt[cnt]
#         tmp2 = list_gt[cnt][0]
#
#         matrix_must_pix[i, cnt] = cw_img[list_gt[i][0], list_gt[i][1], cnt].astype('int32')



# classify_SVM(X, y, 0.25)
# classify_RFC(X, y, 0.25)
# classify_dec_tree(X, y, 0.25)
#classify_XGB(X, y, 0.25)

n_c = 12
pca = PCA(n_components=n_c)
XPCA = pca.fit_transform(matrix_must_pix)

X_PCA_list = XPCA.tolist()
print("PCA classify for %d components" % n_c)
# classify_SVM(X_PCA,y,0.25)
# classify_RFC(X_PCA,y,0.25)
# classify_dec_tree(X_PCA,y,0.25)
classify_XGB(X_PCA_list, y, 0.25)
