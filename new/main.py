import numpy as np
import random
import scipy.io
import imageio
from PIL import Image
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

name='salinas_corrected'
mat = scipy.io.loadmat(name+'.mat')
shape=(mat[name].shape)


def disper_chanel(chn):

    cnt=0
    disp_list=[]
    x_quadr=[]
    x2=[]

    while cnt < chn:
        tmp=mat[name][:,:,cnt].astype('int32')
        x_quadr.append(np.mean(tmp**2))
        x2.append((np.mean(tmp)) ** 2)



        img = Image.fromarray(tmp.astype('uint8'))
        img.convert('RGB').save(r"D:\dk\postgraduate\hyperspectral\image"+str(cnt)+".png")
        cnt += 1

    for i in range(len(x2)):
        disp_list.append(x_quadr[i]-x2[i])

    return disp_list



def fliter():
    cnt=0
    disp_list=[]
    x_quadr = []
    x2 = []
    while cnt < shape[2]:
        img = mat[name][:, :, cnt].astype('int16')
        filt_img=median_filter(img, size=3)
        diff_img=img-filt_img
        x_quadr.append(np.mean(diff_img ))
        x2.append((np.mean(diff_img)) ** 2)
        disp_list.append(np.mean(diff_img ** 2) - (np.mean(diff_img)) ** 2)
        print(cnt)
        cnt+=1

    return disp_list

def embbed(c,wm,delt,ampl):

    cw=(np.floor(c/(2*delt))*2*delt+wm*delt*ampl)

    return cw

def extract(c,cw,delt,ampl):

    w = (cw/delt-np.floor(c/2/delt)*2)/ampl

    return w

#disper_chanel(shape[2])
dsp=fliter()
print(min(dsp),max(dsp))
plt.plot(dsp)
plt.title(name)
plt.xlabel('Номер канала')
plt.ylabel('Дисперсия')
plt.show()

wm = np.random.randint(0,2,size = (shape[0],shape[1]))
wm=np.where(wm==1,255,0)
hp_img=np.zeros((shape[0],shape[1],shape[2]))
for cnt in range(0,shape[2]):
    cw=embbed(mat[name][:, :, cnt],wm,5,dsp[cnt]/2500)
    hp_img[:,:,cnt]=cw
    mg = Image.fromarray(cw.astype('uint8'))
    mg.convert('RGB').save(r"D:\dk\postgraduate\hyperspectral\embed\contain"+str(cnt)+".png")

for cnt in range(0,shape[2]):
    extr_wm=extract(mat[name][:, :, cnt],hp_img[:,:,cnt],5,10)
    mg = Image.fromarray(extr_wm.astype('uint16'))
    mg.convert('RGB').save(r"D:\dk\postgraduate\hyperspectral\extract\wm" + str(cnt) + ".png")

