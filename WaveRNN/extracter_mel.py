# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:41:09 2020

@author: e7789520
"""

from utils.dsp import *
import hdf5storage
import os
from os import walk
from os.path import join


def convert_file(path):
    y = load_wav(path)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak
    mel = melspectrogram(y)
    endframe= int(((len(y)-400)//160)+1)
    mel = mel[:,:endframe]
    # if hp.voc_mode == 'RAW':
    #     if hp.mu_law :
    #         quant = encode_mu_law(y, mu=2**hp.bits)
    #     else:
    #         quant = float_2_label(y, bits=hp.bits)
    # elif hp.voc_mode == 'MOL':
    #     quant = float_2_label(y, bits=16)
    return mel.astype(np.float32)#, quant.astype(np.int64)

wav_list=[]
path_list=[]
wavPath = r"E:\Code\Python\WaveRNN\WaveRNN_CCC\Data_raw\YU\YU_S1_DTW"
savepath = r"E:\Code\Python\WaveRNN\WaveRNN_CCC\Conveted_mel\YU_S1_mel80"
#root, dirs, files = os.walk(wavPath)
for root, dirs, files in os.walk(wavPath):
    #print(dirs)
    for f in files:
        fullpath = join(root, f)
        fullfile = join(f)
        path_list.append(fullpath)
        wav_list.append(fullfile)

with open(r"E:\Code\Python\WaveRNN\WaveRNN_CCC\Conveted_mel\YU_S1_mel80\target_list",'w') as asd:
    for i in path_list:
        asd.write(i+'\n')

if __name__ == '__main__':
    #with open(r"E:\Kevin\org_gatedcnn\VC\Target_list/target_list",'r') as asd:
        for i in path_list:
            mel = convert_file(i)
            hdf5storage.savemat(savepath+'/'+i.split('\\')[-2]+'_'+i.split('\\')[-1].split('.')[0],{'mel80':mel})
            
            
