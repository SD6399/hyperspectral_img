import math
from skimage import io
from skimage.util import random_noise
from matplotlib import pyplot as plt
from skimage.exposure import histogram
import re
import cv2, os
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

size_quadr=16
size_qr=65

def sort_spis(sp):
    spp = []
    spb = []
    res = []
    for i in (sp):
        spp.append("".join(re.findall(r'\d', i)))
        spb.append("result")
    result = [int(item) for item in spp]
    result.sort()

    result1 = [str(item) for item in result]
    for k in range(len(sp)):
        res.append(spb[k] + result1[k] + ".png")
    return (res)


def big2small(st_qr):
    qr = np.zeros((size_qr, size_qr, 3))

    for i in range(0, 1040, size_quadr):
        for j in range(0, 1040, size_quadr):
            qr[int(i / size_quadr), int(j / size_quadr)] = np.mean(st_qr[i:i + size_quadr, j:j + size_quadr])

    return qr


def img2bin(img):
    k =0
    matr_avg = np.zeros((size_qr, size_qr))

    our_avg = np.mean(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            tmp=img[i,j]

            if tmp > our_avg:
                img[i, j] = 255
            else:
                img[i, j] = 0

            k+=1
    return img


def randomize_tiles_shuffle_blocks(a, M, N):
    m, n = a.shape
    b = a.reshape(m // M, M, n // N, N).swapaxes(1, 2).reshape(-1, M * N)
    np.random.shuffle(b)
    return b.reshape(m // M, n // N, M, N).swapaxes(1, 2).reshape(a.shape)


def read_video_cadr(path):
    vidcap = cv2.VideoCapture(path)

    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if (success == True):
            cv2.imwrite(r"D:\dk\university\nirs\sinus\frame%d.png" % count, image)

        print("записан кадр", count)

        if cv2.waitKey(10) == 27:
            break
        count += 1


def exp_smooth(path, alf):
    cnt = 0
    f1=np.asarray([])
    while (cnt < count):
        arr = np.asarray(Image.open(path + str(cnt) + ".png"))

        d1 = f1
        if (cnt == 0):
            f1 = arr.copy()
            d1 = np.zeros((1080, 1920))
        else:
            f1 = np.float32(d1) * alf + np.float32(arr) * (1 - alf)

        f1[f1 > 255] = 255
        f1[f1 < 0] = 0

        print("tmp kadr", cnt)
        cnt += 1
    return f1


def disp(path):
    cnt = 0
    arr = np.array([])

    d = []
    while (cnt < 3000):
        tmp = np.copy(arr)
        arr = np.asarray(Image.open(path + str(cnt) + ".png")).astype(float)
        if cnt == 0:
            d.append(0)

        else:
            razn = np.abs(arr-tmp)
            print(np.mean(razn)," kadr ", cnt)
            d.append(np.mean(razn))
        cnt += 1



    avg = sum(d) / len(d)
    print(avg)
    vyv = []
    for i in range(len(d)):
        if abs((d[i])) > (2*avg):
            vyv.append(i)

    print(vyv)
    print(len(vyv))
    return vyv


def embed(my_i, tt, count):


    
    
    cnt = 0
    PATH_IMG = r'D:\dk\university\nirs\some_qr.png'
    fi = math.pi / 2 / 255

    arr1 = io.imread(PATH_IMG)

    #arr1=cv2.cvtColor(arr1,cv2.COLOR_RGB2YCrCb)

    pict = np.zeros((1080, 1920, 3))
    # встраивание QR-кода в пустой контейнер большого размера
    pict[20:1060, 432:1472,0] = arr1

    list0 = []
    list1 = []
    for i in range(4, 1076, size_quadr):
        for j in range(0, 1920, size_quadr):

            if ([i, j] not in list1) and (pict[i, j, 0] == 255):
                list1.append([int(i / size_quadr), int(j / size_quadr)])
            else:
                list0.append([int(i / size_quadr), int(j / size_quadr)])
    arr = np.zeros((1080, 1920,3))
    arr[4:1076, :,0] = randomize_tiles_shuffle_blocks(pict[4:1076, :, 0], size_quadr, size_quadr)

    list0_new = []
    list1_new = []

    for i in range(4, 1076, size_quadr):
        for j in range(0, 1920, size_quadr):
            if ([i, j] not in list1_new) and (arr[i, j,0] == 255):
                list1_new.append([int(i / size_quadr), int(j / size_quadr)])
            else:
                list0_new.append([int(i / size_quadr), int(j / size_quadr)])

    while (cnt < count):
        imgg = (io.imread("D:/dk/university/nirs/sinus/frame%d.png" % cnt))

        a = cv2.cvtColor(imgg, cv2.COLOR_RGB2YCrCb)

        temp = np.float32(fi) * np.float32(arr)
        wm = np.asarray((my_i * np.sin(cnt * tt + temp)))
        if my_i == 1:
            wm[wm > 0] = 1
            wm[wm < 0] = -1

        a[:,:,0] = np.float32(a[:,:,0] + wm[:,:,0])
        a[a > 255] = 255
        a[a < 0] = 0


        a = cv2.cvtColor(a, cv2.COLOR_YCrCb2RGB)
        img = Image.fromarray(a.astype('uint8'))

        print("обработан кадр", cnt)
        img.convert('RGB').save(r"D:\dk\university\nirs\carrier\result" + str(cnt) + ".png")

        cnt += 1

    return list0, list1, list0_new, list1_new


def extract(alf, tt, rand_fr):

    PATH_VIDEO = r'D:\dk\university\nirs\carrier\True_RB_codH264.avi'
    vidcap = cv2.VideoCapture(PATH_VIDEO)
    vidcap.open(PATH_VIDEO)

    alf0 = 0.96
    betta = 0.999
    count = 0
    count = rand_fr

    success = True
    while success:
        success, image = vidcap.read()
        if (success == True):
            print('Read a new frame:%d ' % count, success)
            cv2.imwrite(r'D:\dk\university\nirs\extract\frame%d.png' % count, image)

        count += 1

    count = 3000

    # первичное сглаживание
    f1 = exp_smooth("D:/dk/university/nirs/extract/frame", alf)

    cnt = rand_fr

    change_sc=disp(r"D:\dk\university\nirs\extract\frame")
    change_sc.insert(0, 0)
    change_sc.append(count)
    g=np.asarray([])
    d = np.asarray([])
    f=d.copy()
    # вычитание усреднённого
    for scene in range(1,len(change_sc)):
        cnt=change_sc[scene-1]
        while (cnt <  change_sc[scene]):
            arr = np.asarray(Image.open(r"D:\dk\university\nirs\extract\frame" + str(cnt) + ".png"))


            a1 = np.asarray([])

            a1 = np.where(arr < f1, 0, arr - f1)

            print("diff", cnt)
            # извлечение ЦВЗ

            arr = a1
            a = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)

            g = d
            d = f

            if (cnt == change_sc[scene-1]):
                f = a[:, :, 0]
                d = f.copy()
                d = np.ones((1080, 1920))

            else:
                if (cnt == change_sc[scene-1] + 1):
                    f = 2 * betta * math.cos(tt) * np.float32(d) + np.float32(a[:, :, 0])

                else:
                    f = 2 * betta * math.cos(tt) * np.float32(d) - (betta ** 2) * np.float32(g) + np.float32(a[:, :, 0])

            yc = np.float32(f) - betta * math.cos(tt) * np.float32(d)
            ys = betta * math.sin(tt) * np.float32(d)
            c = math.cos(tt * cnt) * np.float32(yc) + math.sin(tt * cnt) * np.float32(ys)
            s = math.cos(tt * cnt) * np.float32(ys) - math.sin(tt * cnt) * np.float32(yc)

            flag = False

            fi = np.where(c < 0, np.arctan((s / c)) + np.pi,
                          np.where(s >= 0, np.arctan((s / c)), np.arctan((s / c)) + 2 * np.pi))
            fi = np.where(fi < -np.pi / 4, fi + 2 * np.pi, fi)
            fi = np.where(fi > 9 * np.pi / 4, fi - 2 * np.pi, fi)

            wm = 255 * fi / 2 / math.pi
            # wm[wm>255]=255
            # wm[wm<0]=0

            a1 = wm
            # a1 = cv2.cvtColor(a1, cv2.COLOR_YCrCb2RGB)
            img = Image.fromarray(a1.astype('uint8'))

            # print (wm)
            img.save(r'D:/dk/university/nirs/extract/wm/result' + str(cnt) + '.png')
            print('made', cnt)

            l_kadr = np.asarray(Image.open(r'D:/dk/university/nirs/extract/wm/result' + str(cnt) + '.png'))

            fi = np.copy(l_kadr)
            fi_tmp = np.copy(fi)
            fi = (l_kadr * np.pi * 2) / 255

            hist, bin_centers = histogram(fi, normalize=False)

            dis = []

            koord1 = np.copy(fi)

            koord2 = np.copy(fi)
            koord1 = np.where(fi < np.pi, (fi / np.pi * 2 - 1) * (-1),
                              np.where(fi > np.pi, ((fi - np.pi) / np.pi * 2 - 1), fi))
            koord2 = np.where(fi < np.pi / 2, (fi / np.pi / 2),
                              np.where(fi > 3 * np.pi / 2, ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
                                       ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1)))
            hist, bin_centers = histogram(koord1, normalize=False)
            hist2, bin_centers2 = histogram(koord2, normalize=False)

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
                    pr1 = (bin_centers[i])

            mx_sp2 = np.array([])
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

            moment = np.where(pr1 < 0, np.arctan((pr2 / pr1)) + np.pi,
                              np.where(pr2 >= 0, np.arctan((pr2 / pr1)), np.arctan((pr2 / pr1)) + 2 * np.pi))

            if (moment >= np.pi / 4 and moment <= np.pi * 2 - np.pi / 4):
                fi_tmp = fi - moment + 0.5 * np.pi * 0.5
                fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
                fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)


            elif (moment > np.pi * 2 - np.pi / 4):
                fi = np.where(fi < np.pi / 4, fi + 2 * np.pi, fi)
                fi_tmp = fi - moment + 0.5 * np.pi * 0.5
                fi_tmp = np.where(fi_tmp < -np.pi, fi_tmp + 2 * np.pi, fi_tmp)
                fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

            elif (moment < np.pi / 4):
                fi_tmp = fi - 2 * np.pi - moment + 0.5 * np.pi * 0.5
                fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
                fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

            print(my_exit)
            fi_tmp[fi_tmp < 0] = 0
            fi_tmp[fi_tmp > np.pi] = np.pi
            l_kadr = fi_tmp * 255 / (np.pi)

            cp = l_kadr.copy()
            imgc = Image.fromarray(cp.astype('uint8'))
            imgc.save(r"D:\dk\university\nirs\extract\wm\result" + str(cnt) + ".png")
            pict = np.zeros((1080, 1920))

            for i in range(len(list1)):
                pict[4 + list1[i][0] * size_quadr:4 + list1[i][0] * size_quadr + size_quadr, list1[i][1] * size_quadr:list1[i][1] * size_quadr + size_quadr] = \
                    cp[list1_new[i][0] * size_quadr:list1_new[i][0] *size_quadr + size_quadr, list1_new[i][1] * size_quadr: list1_new[i][1] * size_quadr + size_quadr]
            for i in range(len(list0)):
                pict[4 + list0[i][0] *size_quadr:4 + list0[i][0] * size_quadr + size_quadr, list0[i][1] * size_quadr:list0[i][1] * size_quadr + size_quadr] = \
                    cp[list0_new[i][0] * size_quadr:list0_new[i][0] * size_quadr + size_quadr, list0_new[i][1] * size_quadr: list0_new[i][1] * size_quadr + size_quadr]

            c_qr = pict[20:1060, 432:1472]

            small_new_qr=np.zeros((size_qr,size_qr))

            small_qr=big2small(c_qr)

            imgc = Image.fromarray(small_qr.astype('uint8'))
            imgc.save(r"D:\dk\university\nirs\extract\wm\wm_must\sorry\comparing" + str(cnt) + ".png")



            small_new_qr = img2bin(small_qr[:,:,0])
            mgc = Image.fromarray(small_new_qr.astype('uint8'))
            mgc.save(r"D:\dk\university\nirs\extract\wm\wm_must\compare/compar" + str(cnt) + ".png")
            if (cnt % 200) == 0 or (cnt == 2998):
                stop_kadr2.append(compare(r"D:\dk\university\nirs\extract\wm\wm_must\compare/compar" + str(cnt) + ".png"))

                print("mod 200 ", cnt)
            cnt += 1

    # повторное сглаживание
    count = 3000
    cnt = 0
    g2 = np.asarray([])
    f2 = np.copy(g2)
    alf2 = 0.95
    while (cnt < count - 1):

        arr = np.asarray(Image.open(r"D:\dk\university\nirs\extract\wm\wm_must\sorry\comparing" + str(cnt) + ".png"))
        # g2 - y(n-1)
        g2 = f2
        if (cnt == 0):
            f2 = arr.copy()
            g2 = np.zeros((size_qr, size_qr))
        else:
            # y(n)=alfa*y(n-1)+x(n)*(1-alfa)
            f2 = g2 * alf2 + arr * (1 - alf2)
            f2[f2 > 255] = 255

        img = Image.fromarray(f2.astype('uint8'))

        print("avg kadr", cnt)
        img.save(r"D:/dk/university/nirs/extract/wm/wm_must/wm2/result" + str(cnt) + ".png")
        cnt += 1

    count = 3000
    cnt = 0
    while (cnt < count - 1):
        if (cnt % 200) == 0 or (cnt == 2998):
            c_qr = io.imread(r"D:\dk\university\nirs\extract\wm\wm_must\wm2\result" + str(cnt) + ".png")
            c_qr = img2bin(c_qr[:,:,0])

            img1 = Image.fromarray(c_qr.astype('uint8'))
            img1.save(r"D:/dk/university/nirs/extract/wm/wm_must/result" + str(cnt) + ".png")
            stop_kadr1.append(compare(r"D:/dk/university/nirs/extract/wm/wm_must/result" + str(cnt) + ".png"))

        cnt += 1

    return (r"D:/dk/university/nirs/extract/wm/wm_must/result" + str(cnt - 1) + ".png")


def generate_video():
    image_folder = r'D:\dk\university\nirs\carrier'  # make sure to use your folder
    video_name = 'True_RB_codH264.avi'
    os.chdir(r"D:\dk\university\nirs\carrier")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_list = sort_spis(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    cnt = 0
    for image in sort_list:
        print("video is writing", cnt)
        video.write(cv2.imread(os.path.join(image_folder, image)))
        cnt += 1

    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


def compare(p):  # сравнивание извлечённого QR с исходным
    qr = io.imread(r'D:\dk\university\nirs\some_qr.png')
    qr = np.where(qr > 129, 255, 0)
    sm_qr=big2small(qr)
    sr_matr = np.zeros((size_qr, size_qr))
    myqr = io.imread(p)
    myqr = np.where(myqr > 129, 255, 0)

    k = 0
    mas_avg = []
    for i in range(0, size_qr):
        for j in range(0, size_qr):
            if np.mean(sm_qr[i, j]) == np.mean(myqr[i, j]):
                sr_matr[int(i), int(j)] = 1
                mas_avg.append(1)
            else:
                sr_matr[i, j] = 0
                mas_avg.append(0)

    for i in (mas_avg):
        if i == 1:
            k += 1
    return (k / len(mas_avg))


i = 1
my_exit = []
alfa = 0.93
tetta = 0.3
squ_size = 4
for_fi = 6
# dispr=1

stop_kadr1 = []
stop_kadr2 = []
stop_kadr3 = []

PATH_VIDEO = r'D:\dk\university\nirs\RealBarcaHighlights_HD (online-video-cutter.com).mp4'

# read_video_cadr(PATH_VIDEO)

rand_k = 0
count = 3000

while (tetta < 3.02):
    list0, list1, list0_new, list1_new = embed(i, tetta, count)
    print("number's shuffle squares", list0),
    print(list1)
    print(list0_new)
    print(list1_new)
    generate_video()

    sp = []
    a = extract(alfa, tetta, rand_k)

    my_exit.append(compare("D:/dk/university/nirs/extract/wm/wm_must/result2998.png"))
    print("current percent", stop_kadr1)
    print("current percent", stop_kadr2)
    print("current percent", stop_kadr3)
    # i+=1
    tetta += 1

print(my_exit)

fig = plt.figure()
ax = fig.add_subplot(111, label="1")
plt.plot([i for i in np.arange(0.91, 1, 0.02)], my_exit)
plt.plot([i for i in np.arange(1, 5.1, 1)], stop_kadr1)
plt.plot([i for i in np.arange(1, 5.1, 1)], stop_kadr2)
plt.plot([i for i in np.arange(1, 5.1, 1)], stop_kadr3)

plt.show()