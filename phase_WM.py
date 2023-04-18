import math
import matplotlib.pyplot as plt
import csv
from skimage import io
from reedsolo import RSCodec
from skimage.exposure import histogram
import cv2, os
import numpy as np
from PIL import Image, ImageFile
from helper_methods import sort_spis, img2bin
from helper_methods import csv2list, bit_voting
from reedsolomon import extract_RS, rsc, Nbit
from helper_methods import init2matr


def disp_pix(path):
    arr = io.imread(path + '/result' + str(0) + ".png").astype(float)
    cnt = 1
    list_diff = []
    while cnt < total_count:
        tmp = np.copy(arr)
        arr = io.imread(path + '/result' + str(cnt) + ".png").astype(float)

        diff_pix = np.abs(arr - tmp)
        print(np.mean(diff_pix), " frame ", cnt)
        list_diff.append(np.mean(diff_pix))
        cnt += 1

    avg = sum(list_diff) / len(list_diff)
    print(avg)
    upd_start = []
    for i in range(len(list_diff)):
        if abs((list_diff[i])) > (3 * avg):
            upd_start.append(i)

    print("frame with change scene", upd_start)
    return upd_start


def embed(hs_mat, my_i, tt):
    cnt = 0

    PATH_IMG = r"C:/Users/user/PycharmProjects/hyperspectral_img/RS_cod89x89.png"
    fi = math.pi / 2 / 2047

    st_qr = cv2.imread(PATH_IMG)
    st_qr = cv2.cvtColor(st_qr, cv2.COLOR_RGB2GRAY)
    v_list = variance_list
    for i in range(len(variance_list)):
        cnt += 5
        while cnt < variance_list[i]:

            img = Image.fromarray(hs_mat[0:89, 0:89, cnt].astype('uint8'))
            img.save(r"C:\Users\user\PycharmProjects\hyperspectral_img\hs_b4/result" + str(cnt) + ".png")

            temp = np.float32(fi) * np.float32(st_qr)
            wm = np.asarray((my_i * np.sin(cnt * tt + temp)))
            if my_i == 1:
                wm[wm > 0] = 1
                wm[wm < 0] = -1

            hs_mat[0:89, 0:89, cnt] = np.float32(hs_mat[0:89, 0:89, cnt] + wm)

            img = Image.fromarray(hs_mat[0:89, 0:89, cnt].astype('uint8'))
            img.save(r"C:\Users\user\PycharmProjects\hyperspectral_img\hs_a/result" + str(cnt) + ".png")
            if cnt % 50 == 0:
                print("wm embed", cnt)
            cnt += 1

    return hs_mat
    # print(shuf_order)


def read2list(file):
    # открываем файл в режиме чтения utf-8
    file = open(file, 'r', encoding='utf-8')

    # читаем все строки и удаляем переводы строк
    lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]

    file.close()

    return lines


def extract(hs, alf, tt):
    betta = 0.999

    cnt = 0

    g = np.asarray([])
    f = g.copy()
    f1 = f.copy()
    d = g.copy()
    d1 = d.copy()
    hs_smooth = np.zeros((89, 89, hs.shape[2]))

    while cnt < hs.shape[2]:

        a = hs[:, :, cnt]
        # g1=d1 # !!!!!!!!!!!!!
        d1 = f1
        if cnt == 0:
            f1 = a.copy()
            d1 = np.zeros((89, 89))
        else:
            f1 = np.float32(d1) * alf + np.float32(a) * (1 - alf)

        hs_smooth[:, :, cnt] = f1[0:89, 0:89]

        img = Image.fromarray(f1[0:89, 0:89].astype('uint8'))
        img.save(r"C:\Users\user\PycharmProjects\hyperspectral_img\smooth/result" + str(cnt) + ".png")
        if cnt % 30 == 0:
            print("first smooth", cnt)
        cnt += 1

    vot_sp = []
    cnt = - 5
    count = 0
    # вычитание усреднённого
    for ind in range(len(variance_list)):
        if cnt + 5 <= variance_list[ind]:
            cnt += 5
        else:
            cnt = variance_list[ind]
        while cnt < variance_list[ind]:

            a = hs[0:89, 0:89, cnt]

            f1 = hs_smooth[:, :, cnt]
            # f1=np.float32(f1)
            a1 = np.where(a < f1, f1 - a, a - f1)

            res_1d = np.ravel(a1)
            start_qr = np.resize(res_1d, (89, 89))

            # извлечение ЦВЗ
            arr = start_qr
            a = arr

            g = d
            d = f

            if cnt == 0:
                f = a
                d = f.copy()
                d = np.ones((89, 89))

            else:
                if cnt == 1:
                    f = 2 * betta * math.cos(tt) * np.float32(d) + np.float32(a)

                else:
                    f = 2 * betta * math.cos(tt) * np.float32(d) - (betta ** 2) * np.float32(g) + np.float32(a)

            yc = np.float32(f) - betta * math.cos(tt) * np.float32(d)
            ys = betta * math.sin(tt) * np.float32(d)
            c = np.cos(tt * cnt) * np.float32(yc) + np.sin(tt * cnt) * np.float32(ys)
            s = np.cos(tt * cnt) * np.float32(ys) - np.sin(tt * cnt) * np.float32(yc)
            # print("yc",np.max(yc),np.min(yc))
            # print("ys",np.max(ys), np.min(ys))
            # print("c",np.max(c), np.min(c))
            # print("s",np.max(s), np.min(s))
            fi = np.where(c < 0, np.arctan((s / c)) + np.pi,
                          np.where(s >= 0, np.arctan((s / c)), np.arctan((s / c)) + 2 * np.pi))
            fi = np.where(fi < -np.pi / 4, fi + 2 * np.pi, fi)
            fi = np.where(fi > 9 * np.pi / 4, fi - 2 * np.pi, fi)
            wm = 2047 * fi / 2 / math.pi

            a1 = wm
            l_kadr = a1

            fi = np.copy(l_kadr)
            fi_tmp = np.copy(fi)
            fi = (l_kadr * np.pi * 2) / 2047

            dis = []
            coord1 = np.copy(fi)

            coord2 = np.copy(fi)
            coord1 = np.where(fi < np.pi, (fi / np.pi * 2 - 1) * (-1),
                              np.where(fi > np.pi, ((fi - np.pi) / np.pi * 2 - 1), fi))
            # list001.append(coord1[0,0])
            coord2 = np.where(fi < np.pi / 2, (fi / np.pi / 2),
                              np.where(fi > 3 * np.pi / 2, ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
                                       ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1)))
            # list_phas.append(coord2[0, 0])

            coord1 = coord1[~np.isnan(coord1)]
            coord2 = coord2[~np.isnan(coord2)]
            hist, bin_centers = histogram(coord1, normalize=False)
            hist2, bin_centers2 = histogram(coord2, normalize=False)

            ver = []
            ver2 = []
            mx_sp = np.arange(bin_centers[0], bin_centers[-1], bin_centers[1] - bin_centers[0])
            for i in range(len(hist)):
                ver.append(hist[i] / sum(hist))
            mo = moment = 0
            for i in range(len(hist)):
                mo += bin_centers[i] * ver[i]
            for mx in mx_sp:
                dis.append(abs(mo - mx))

            pr1 = 0
            pr2 = 0
            for i in range(len(dis)):
                if min(dis) == dis[i]:
                    pr1 = bin_centers[i]

            dis2 = []
            mx_sp2 = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
            for i in range(len(hist2)):
                ver2.append(hist2[i] / sum(hist2))
            mo = 0
            for i in range(len(hist2)):
                mo += bin_centers2[i] * ver2[i]
            for mx in mx_sp2:
                dis2.append(abs(mo - mx))

            x = min(dis2)

            for i in range(len(dis2)):
                if x == dis2[i]:
                    pr2 = bin_centers2[i]

            # list_pr.append(pr1)
            # list_pr2.append(pr2)

            moment = np.where(pr1 < 0, np.arctan((pr2 / pr1)) + np.pi,
                              np.where(pr2 >= 0, np.arctan((pr2 / pr1)), np.arctan((pr2 / pr1)) + 2 * np.pi))

            mom_list.append(moment / (2 * np.pi))

            if np.pi / 4 <= moment <= np.pi * 2 - np.pi / 4:
                fi_tmp = fi - moment + 0.5 * np.pi * 0.5
                fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
                fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

            elif moment > np.pi * 2 - np.pi / 4:
                fi = np.where(fi < np.pi / 4, fi + 2 * np.pi, fi)
                fi_tmp = fi - moment + 0.5 * np.pi * 0.5
                fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
                fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

            elif moment < np.pi / 4:
                fi_tmp = fi - 2 * np.pi - moment + 0.5 * np.pi * 0.5
                fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
                fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

            # list001.append(fi_tmp[0,0])
            fi_tmp[fi_tmp < 0] = 0
            fi_tmp[fi_tmp > np.pi] = np.pi
            fi_tmp = fi_tmp * 255 / np.pi

            small_frame = fi_tmp
            cp = small_frame.copy()
            our_avg = np.mean(cp)
            cp = np.where(cp > our_avg, 255, 0)

            img = Image.fromarray(cp.astype('uint8'))
            img.save(r"C:\Users\user\PycharmProjects\hyperspectral_img\etxract_bin/result" + str(cnt) + ".png")
            if cnt % 50 == 0:
                print("wm extract", cnt)
            if cnt % 1 == 0:

                if extract_RS(io.imread(
                        r"C:\Users\user\PycharmProjects\hyperspectral_img\etxract_bin/result" + str(cnt) + ".png"),
                        rsc, Nbit) == b'':

                    stop_kadr1_bin.append(1)
                else:
                    stop_kadr1_bin.append(0)
                if count < cnt:
                    for q in range(cnt - count):
                        stop_kadr1.append(compare(
                            r"C:\Users\user\PycharmProjects\hyperspectral_img\etxract_bin/result" + str(cnt) + ".png"))
                        count+=1
                else:
                    stop_kadr1.append(compare(
                        r"C:\Users\user\PycharmProjects\hyperspectral_img\etxract_bin/result" + str(cnt) + ".png"))
                print(cnt, stop_kadr1)

            cnt += 1
            count += 1
    print(mom_list)


def compare(path):  # сравнивание извлечённого QR с исходным
    orig_qr = io.imread(r"C:\Users\user\PycharmProjects\hyperspectral_img\RS_cod89x89.png")
    orig_qr = np.where(orig_qr > 127, 255, 0)
    small_qr = orig_qr
    sr_matr = np.zeros((89, 89, 3))
    myqr = io.imread(path)
    myqr = np.where(myqr > 127, 255, 0)

    k = 0
    mas_avg = []
    for i in range(0, 89):
        for j in range(0, 89):

            if np.mean(small_qr[i, j]) == np.mean(myqr[i, j]):
                sr_matr[i, j] = 1
                mas_avg.append(1)
            else:
                sr_matr[i, j] = 0
                mas_avg.append(0)

    for i in mas_avg:
        if i == 1:
            k += 1
    return k / len(mas_avg)


def diff_pix_between_neugb(qr1, qr2):
    k = 0
    mas_avg = []
    for i in range(0, 89):
        for j in range(0, 89):

            if qr1[i, j] == qr2[i, j]:
                mas_avg.append(1)
            else:
                mas_avg.append(0)

    for i in mas_avg:
        if i == 0:
            k += 1
    return k


my_exit = []
my_exit1 = []
my_exit2 = []

squ_size = 4
for_fi = 6
# dispr=1

# графики-сравнения по различныи параметрам


# with open('change_sc.csv', 'r') as f:
#     change_sc = list(csv.reader(f))[0]
#
# change_sc = [eval(i) for i in change_sc]

# count=read_video(PATH_VIDEO)

rand_k = 0
total_count = 176

hm_list = []
alfa = 0.3
tetta = 3

sp = []

# for tr in np.arange(0,206,20):
vot_sp = []

shape, mat_gt, mat_data = init2matr("KSC")
for ampl in range(9, 10, 4):
    variance_list = disp_pix(r"C:\Users\user\PycharmProjects\hyperspectral_img\hs_b4")
    variance_list.append(176)
    print(variance_list)
    prot_hs = embed(mat_data, ampl, tetta)
    stop_kadr1 = []
    mom_list = []
    stop_kadr2 = []
    stop_kadr1_bin = []
    stop_kadr2_bin = []

    print('GEN')
    extract(prot_hs, alfa, tetta)
    print("all")

    print(ampl, "current percent", stop_kadr1)
    print(stop_kadr1_bin)

    fig, ax = plt.subplots()
    font = {
        'size': 12, }
    ax.plot(stop_kadr1, label='Percentage')
    # ax.plot(mom_list, label='moment')
    plt.title("KSC", fontdict=font)
    plt.xlabel("Number of channel ", fontdict=font)
    plt.ylabel("Accuracy of extraction", fontdict=font)

    plt.legend()
    ax.legend(fontsize=12)
    # plt.savefig(r'D:\HSI\pict1.png', dpi=300)

    plt.show()
