import os
import time
import pywt
import cv2
import json
import pickle
import itertools
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from tqdm import tqdm
from scipy import signal
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
# from stockwell import st
from scipy.signal import chirp


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
        # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


save_path = './generate_features_3/'


mitbih_path = './processed_data/'

# df = pd.read_excel('/home/vidhiwar/work_dir/ecg_spectogram/phase1/ecg_res/ECG_Data_Collect.xlsx')

if not os.path.exists(os.path.join(save_path,'mitbih_rl')):
    os.mkdir(os.path.join(save_path,'mitbih_rl'))

save_path = os.path.join(save_path, 'mitbih_rl')



if not os.path.exists(os.path.join(save_path,'train')):
    os.mkdir(os.path.join(save_path,'train'))
if not os.path.exists(os.path.join(save_path,'val')):
    os.mkdir(os.path.join(save_path,'val'))
    
if not os.path.exists(os.path.join(save_path,'train','images')):
    os.mkdir(os.path.join(save_path,'train','images'))
if not os.path.exists(os.path.join(save_path,'val','images')):
    os.mkdir(os.path.join(save_path,'val','images'))
    
if not os.path.exists(os.path.join(save_path,'train','images_full')):
    os.mkdir(os.path.join(save_path,'train','images_full'))
if not os.path.exists(os.path.join(save_path,'val','images_full')):
    os.mkdir(os.path.join(save_path,'val','images_full'))

for i in ['stft','cwt']:
    if not os.path.exists(os.path.join(save_path,'train','images',i)):
        os.mkdir(os.path.join(save_path,'train','images',i))
    if not os.path.exists(os.path.join(save_path,'val','images',i)):
        os.mkdir(os.path.join(save_path,'val','images',i))
        
    if not os.path.exists(os.path.join(save_path,'train','images_full',i)):
        os.mkdir(os.path.join(save_path,'train','images_full',i))
    if not os.path.exists(os.path.join(save_path,'val','images_full',i)):
        os.mkdir(os.path.join(save_path,'val','images_full',i))
    
if not os.path.exists(os.path.join(save_path,'train','ecg')):
    os.mkdir(os.path.join(save_path,'train','ecg'))
if not os.path.exists(os.path.join(save_path,'val','ecg')):
    os.mkdir(os.path.join(save_path,'val','ecg'))



data_loc = "./data"
files = [file for file in os.listdir(data_loc) if file.split(".")[-1] == "dat"]
# print(files)
idxs = [os.path.basename(f).split(".")[0] for f in files]
print(idxs)



import pandas as pd 
df = pd.read_csv("age_gender.csv")





fs = 360
STEP = 3600
win_sz = 72
overlap = 36

outer_win_sz = 180
outer_overlap = 90

zero = 1024
inner_step = 360
widths = np.concatenate([np.flip(np.arange(1, 181)),np.arange(1, 181)])
outer_widths = np.concatenate([np.flip(np.arange(1, 721)),np.arange(1, 721)])
g_dict = {'M':[0,1],'F':[1,0]}

l2n = {'N':0, 'L':1, 'R':2, 'V':3, 'A':4,'~':5}

n2l = {0:'N', 1:'L', 2:'R', 3:'V', 4:'A',5:'~'}

train = []
val = []

win = signal.windows.hann(win_sz)
outer_win = signal.windows.hann(outer_win_sz)
name = 0
l_defaullt = '~'
for j in tqdm(range(0,11,1)):
    i = df['#record'][j]
    if os.path.exists(os.path.join(mitbih_path,str(i) + '.pkl')):
        with open(os.path.join(mitbih_path,str(i) + '.pkl'), 'rb') as f:
            data = pickle.load(f)
        w = np.load(os.path.join(mitbih_path,str(i) + '.npy'))
        l_dict = {}
        for a in data:
            if a[2] in ['N', 'L', 'R', 'V', 'A']:
                l_dict[a[1]] = a[2]
            else:
                l_dict[a[1]] = '~'

        s = w[:,2]
        g = g_dict[df['Sex'][j]]
        age = 0
        if df['Age'][j] > 0 and df['Age'][j] < 100:
            age = df['Age'][j]/100
            
        
        #    break
        length = s.shape[0]
        start = 0
        end = STEP
        #     for i in range(0,int(np.floor(length/STEP))):
        while end < length:

            sample = {}
            
            inner_start = start
            inner_end = start+inner_step
            
            segment_images = []
            segment_ecg = []
            segment_id = 0
            f,t,zxx = signal.stft(s[start : end]-zero,fs, window=outer_win, nperseg=outer_win_sz, noverlap=outer_overlap, nfft=outer_win_sz, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
                # plt.figure(figsize=(33,21))
            fig = Figure()
            fig.subplots_adjust(0,0,1,1)
                #         fig.add_axes([0,0,1,1])
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            ax.pcolormesh(t, f, 20*np.log10(np.abs(zxx)), shading='gouraud')
            ax.axis('off')
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            
            cwt = signal.cwt(s[start:end]-zero,signal.ricker,outer_widths)
            labels = []
            
            if name%10 == 0:
                plt.imsave(os.path.join(save_path,'val','images_full','cwt',str(name)+'.jpg'),cwt)
                cwt = plt.imread(os.path.join(save_path,'val','images_full','cwt',str(name)+'.jpg'))
                cwt = cv2.resize(cwt,(224,224))
                plt.imsave(os.path.join(save_path,'val','images_full','cwt',str(name)+'.jpg'),cwt)
                c = cv2.resize(image,(224,224))
                plt.imsave(os.path.join(save_path,'val','images_full','stft',str(name)+'.jpg'),c)
                
            else:
                plt.imsave(os.path.join(save_path,'train','images_full','cwt',str(name)+'.jpg'),cwt)
                cwt = plt.imread(os.path.join(save_path,'train','images_full','cwt',str(name)+'.jpg'))
                cwt = cv2.resize(cwt,(224,224))
                plt.imsave(os.path.join(save_path,'train','images_full','cwt',str(name)+'.jpg'),cwt)
                c = cv2.resize(image,(224,224))
                plt.imsave(os.path.join(save_path,'train','images_full','stft',str(name)+'.jpg'),c)
            while inner_end <= end:
            
                f,t,zxx = signal.stft(s[inner_start : inner_end]-zero,fs, window=win, nperseg=win_sz, noverlap=overlap, nfft=win_sz, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
                    # plt.figure(figsize=(33,21))
                fig = Figure()
                fig.subplots_adjust(0,0,1,1)
                    #         fig.add_axes([0,0,1,1])
                canvas = FigureCanvas(fig)
                ax = fig.gca()
                ax.pcolormesh(t, f, 20*np.log10(np.abs(zxx)), shading='gouraud')
                ax.axis('off')
                canvas.draw()
                width, height = fig.get_size_inches() * fig.get_dpi()
                image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
                
                cwt = signal.cwt(s[inner_start:inner_end]-zero,signal.ricker,outer_widths)
                segment_ecg.append(str(name)+'_'+str(segment_id)+'.npy')
                segment_images.append(str(name)+'_'+str(segment_id)+'.jpg')
                if name%10 == 0:
                    plt.imsave(os.path.join(save_path,'val','images','cwt',str(name)+'_'+str(segment_id)+'.jpg'),cwt)
                    cwt = plt.imread(os.path.join(save_path,'val','images','cwt',str(name)+'_'+str(segment_id)+'.jpg'))
                    cwt = cv2.resize(cwt,(224,224))
                    plt.imsave(os.path.join(save_path,'val','images','cwt',str(name)+'_'+str(segment_id)+'.jpg'),cwt)
                    c = cv2.resize(image,(224,224))
                    plt.imsave(os.path.join(save_path,'val','images','stft',str(name)+'_'+str(segment_id)+'.jpg'),c)
                    np.save(os.path.join(save_path,'val','ecg',str(name)+'_'+str(segment_id)+'.npy'),s[inner_start : inner_end]-zero)
                else:
                    plt.imsave(os.path.join(save_path,'train','images','cwt',str(name)+'_'+str(segment_id)+'.jpg'),cwt)
                    cwt = plt.imread(os.path.join(save_path,'train','images','cwt',str(name)+'_'+str(segment_id)+'.jpg'))
                    cwt = cv2.resize(cwt,(224,224))
                    plt.imsave(os.path.join(save_path,'train','images','cwt',str(name)+'_'+str(segment_id)+'.jpg'),cwt)
                    c = cv2.resize(image,(224,224))
                    plt.imsave(os.path.join(save_path,'train','images','stft',str(name)+'_'+str(segment_id)+'.jpg'),c)
                    np.save(os.path.join(save_path,'train','ecg',str(name)+'_'+str(segment_id)+'.npy'),s[inner_start : inner_end]-zero)
                label_list = []
                for k in range(inner_start,inner_end):
                    if k in list(l_dict.keys()):
                        label_list.append(l_dict[k])
                
                if len(label_list) == 0:
                    l = default_label
                else:
                    l = most_common(label_list)
                    default_label = l
                labels.append(l2n[l])
                segment_id += 1
                inner_start += inner_step
                inner_end += inner_step
#             key = list(l_dict.keys())[0]
#             label_list = []
#             for k in range(start,end):
#                 if k in list(l_dict.keys()):
#                     label_list.append(l_dict[k])
            
#             l = most_common(label_list)
            
            sample['images'] = segment_images
            sample['images_full'] = str(name)+'.jpg'
            sample['ecg'] = segment_ecg
            sample['id'] = name
            sample['label'] = labels
            sample['age'] = age
            sample['gender'] = g
            if name%10 == 0:
                val.append(sample)
            else:
                train.append(sample)

            name += 1
            start += STEP
            end += STEP
ecg_train = {}
ecg_val = {}


ecg_train['data'] = train
ecg_val['data'] = val
ecg_train['labels'] = n2l
ecg_val['labels'] = n2l


with open(os.path.join(save_path,'train','ecg_labels.json'),'w') as f:
    json.dump(ecg_train,f,indent=2)

with open(os.path.join(save_path,'val','ecg_labels.json'),'w') as f:
    json.dump(ecg_val,f,indent=2)