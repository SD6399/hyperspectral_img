import numpy as np
import scipy.io
import math
from skimage import io
from scipy.ndimage import median_filter
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from numpy import savetxt

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
from patchwork import patchwork_embed
from patchwork import patchwork_extract
# from interp_based import interp_embed
# from interp_based import interp_extract


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
    feat_list = []
    for rs in range(40, 45):
        np.random.seed(rs)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst_sz, random_state=rs)

        np.random.seed(rs)
        rfc = RandomForestClassifier(random_state=42, n_estimators=100)
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

    # params = {
    #     'eta': np.arange(0.3, 1, 0.2),
    #     'max_depth': range(3, 10, 1),
    #     'n_estimators': range(100, 601, 100),
    #     'colsample_bytree': np.arange(0.5, 1, 0.1)
    # }

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
        # objective='multi:softmax',eval_metric='mlogloss', colsample_bytree = 0.7, eta = 0.5, max_depth = 4, n_estimators = 200
        model = XGBClassifier(random_state=42, use_label_encoder=None, objective='multi:softmax',
                              eval_metric='mlogloss')
        # , colsample_bytree = 0.7, eta = 0.5, max_depth = 4, n_estimators = 200"""
        #
        # cv=5
        # grid_search = GridSearchCV(model, params, scoring="accuracy", n_jobs=-1, cv=cv)
        #
        # grid_search.fit(X_train, y_train)
        # print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
        #
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
        tmp = mat_data[:, :, cnt].astype('int32')
        x_quadr = np.mean(tmp ** 2)
        x2 = (np.mean(tmp)) ** 2
        disp_list.append((x_quadr - x2))

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


# встраивание во все каналы во всё изображение
def old_embed(mat, wm, delt):

    matrix = np.zeros((len(list_gt), shape[2]))

    for i in range(len(list_gt)):
        print(i)
        for cnt in range(shape[2]):

            tmp = list_gt[i][0]
            tmp2 = list_gt[i][1]
            matrix[i, cnt] = np.around((np.floor(mat[tmp, tmp2, cnt] / (2 * delt[cnt])) * 2 *
                                        delt[cnt] + wm[i * cnt + cnt] * delt[cnt]))

    return matrix


def extract(c, cw, delt, ampl):
    w = (cw / delt - np.floor(c / 2 / delt) * 2) / ampl
    return w


def mse(actual, pred):
    return np.square(actual - pred).mean()


def try_patch(c, wm):
    data_length = c.shape[0] * c.shape[1]

    shuf_order = np.arange(data_length)
    c_rav = np.ravel(c[:, :, 0])
    np.random.seed(42)
    np.random.shuffle(shuf_order)
    s1 = shuf_order[:int(len(shuf_order) / 2)]
    s2 = shuf_order[int(len(shuf_order) / 2):]
    for i in s1:
        c_rav[i] += wm[i]
    for i in s2:
        c_rav[i] -= wm[i]
    cw = c_rav.reshape((c.shape[0], c.shape[1]))
    return cw


def embed_to_pix(cont, wm, delt, vol_wm, count_prior_canal):
    matrix = np.zeros((len(list_gt), shape[2]))

    rng = np.random.default_rng(42)
    random_pix = rng.choice(list_gt, int(len(list_gt) * vol_wm), replace=False).tolist()
    random_pix.sort()
    short_rand = random_pix[0:int(len(random_pix))]

    mfl=np.arange(0,shape[2])
    if delt:
        sort_canal = mfl.argsort()[0:count_prior_canal]
    else:
        sort_canal = []

    for i in range(len(list_gt)):
        if i % 500 == 0:
            print(i)
        for cnt in range(shape[2]):
            if (cnt in sort_canal) and (list_gt[i] in short_rand):
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
        cw_img = patchwork_embed(mat_data, wm, dsp_noise)
        mse_res1 = mean_squared_error(mat_data, cw_img)
        print("alfa=", alfa, ":", mse_res1)

        alfa += inc

    return alfa - inc


def preprocess_X(data, list_gt):
    matrix_must_pix_orig = np.zeros((len(list_gt), shape[2]))
    for i in range(len(list_gt)):
        for cnt in range(shape[2]):
            matrix_must_pix_orig[i, cnt] = data[list_gt[i][0], list_gt[i][1], cnt].astype('int32')

    X = matrix_must_pix_orig.tolist()
    return X


def prepare_PCA(matrix_must_pix, n_comp):
    pca = PCA(n_components=n_comp)
    XPCA = pca.fit_transform(matrix_must_pix)
    X_PCA_list = XPCA.tolist()
    return X_PCA_list


def search_classify_pix():
    list_gt = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mat_gt[i][j] != 0:
                list_gt.append([i, j])
    return list_gt


def preprocess_Y(list_gt):
    y = []
    for i in range(len(list_gt)):
        y.append(mat_gt[list_gt[i][0], list_gt[i][1]])

    return y


def select_need_params(func, kef):
    embed_func = func
    for i in range(len(embed_func)):
        embed_func[i] *= kef

    return embed_func


name = 'indian_pines'

shape, mat_gt, mat_data, = init2matr(name)

list_gt = search_classify_pix()
X_orig = np.array(preprocess_X(mat_data, list_gt))
y = preprocess_Y(list_gt)
# mat_interp = interp_embed(mat_data, wm)
# X_interp = preprocess_X(mat_interp[0], list_gt)
# classify_XGB(X_interp, y, 0.25)
#mat_extract = interp_extract(mat_interp[0],mat_interp[1])
# print(mat_data==mat_extract)

#print(mean_squared_error(X_interp, X_orig))

# mfl = classify_XGB(X_orig, y, 0.25)
#
# mean_feat_list = np.mean(mfl, axis=0)
# mfl = mean_feat_list.tolist()
# mse_list = []


print(shape)
cc = [2.8e-05,3.9e-05]
for c in cc:

    patch_list = []
    xgbl = []
    #alfa = [c] * shape[2]
    row = select_need_params(disper_chanel(shape[2]), c)
    count_half = 100
    np.random.seed(42)
    wm = np.random.randint(0, 2, size=(shape[0] * shape[1] * shape[2] * 7))

    seed = 42

    #X = embed_to_pix(mat_data, wm,row,1,X.shape[2])
    X = patchwork_embed(X_orig, wm, row, count_half, seed)
    print("MSE new", mean_squared_error(X_orig, X))
    xgbl.append(classify_XGB(X, y, 0.25))
    #patch_list.append(patchwork_extract(X, count_half, seed, alfa))
    # mse_list.append(mean_squared_error(X_orig, X))





# print(patch_list,count_half,alfa)
# print(sum(patch_list)/len(patch_list))
# classify_SVM(X1, y, 0.25)
# classify_RFC(X1,y,0.25)
# classify_dec_tree(X1,y,0.25)

X_PCA = prepare_PCA(X_orig, 12)
classify_XGB(X_PCA, y, 0.25)
