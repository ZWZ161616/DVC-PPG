# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 22:44:01 2019

@author: 70609
"""
import os
import sys
import h5py
import copy
import librosa
import numpy as np
import scipy
#from scipy import signal
import scipy.io as sio
import pyworld


from tqdm import tqdm


eps_64 = np.finfo(np.float64).eps
eps_32 = np.finfo(np.float32).eps

def split_all(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def load_data_list(data_path):
    Data_list=list()
    data_ext=os.path.splitext(data_path)[1]
    if data_ext=='.mat':
        try:
            List_path = sio.loadmat(data_path)
            Data_key = list(List_path.keys())[-1]
            Data_str_list = List_path[Data_key][0,:]
            Data_list=[str(list_comment[0]) 
                for list_index,list_comment in tqdm(enumerate(Data_str_list[:]),desc='List Reading')]

        except:
            
            print('''.mat file version was save as -v7.3,
                  try using hdf5storage...''')
            import hdf5storage            
            List_path = hdf5storage.loadmat(data_path)
            Data_key = list(List_path.keys())[-1]
            Data_list = Data_list[Data_key]

    elif data_ext=='.list':
        with open(data_path) as f:
            Data_list = f.readlines()
            
    return Data_list

def load_wavs(wav_list, sr):

    wavs = list()
    for file_path in tqdm(wav_list,desc='Wav Reading'):
        wav, _ = librosa.load(file_path, sr = sr, mono = True)
        #wav = wav.astype(np.float64)
        wavs.append(wav)

    return wavs

def lps_extract(wavs, fs, frame_size, *args):
    
    if args:
        if len(args)>2:
           raise Exception("Too many arguments")
           
        frame_shift = args[0]
        fft_size = args[1]
        eps = args[2]
        
    else:
        frame_shift = frame_size//2
        fft_size = frame_size
        eps = eps_64
        
        
    logpows = list()
    phases = list()
    
    for wav in tqdm(wavs, desc = 'Log Power Spectrum Feature Extracting...'):
        wav = wav.astype(np.float64)
        if np.any(wav == 0):
            wav[wav == 0] = eps
            
        D = librosa.stft(wav,n_fft=frame_size,hop_length=frame_shift,win_length=fft_size,window=scipy.signal.hamming)
        
        magnitude  = np.abs(D)
        
        logpow = np.log10(magnitude**2)
        
        phase = np.angle(D)*-1
        
        logpows.append(logpow)
        phases.append(phase)
        
    return logpows, phases


def lps_mean_var(logpows, *args):

    if args:
        if len(args)>2:
           raise Exception("Too many arguments")        
    
    for logpow in tqdm(logpows, desc = 'Calculating Mean & Std...'):

        
        lps_mean.append(np.mean(logpow, axis=1))
        lps_std.append(logpow)
        
    return logpow



def world_decompose(wav, fs, frame_period = 5.0):

    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)

    return f0, timeaxis, sp, ap

def world_encode_spectral_envelop(sp, fs, dim = 24):

    # Get Mel-cepstral coefficients (MCEPs)

    #sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp

def world_decode_spectral_envelop(coded_sp, fs):

    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    #coded_sp = coded_sp.astype(np.float32)
    #coded_sp = np.ascontiguousarray(coded_sp)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp


def world_encode_data(wavs, fs, frame_period = 5.0, coded_dim = 24):

    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()

    for wav in tqdm(wavs, desc = 'Feature Extracting...'):
        f0, timeaxis, sp, ap = world_decompose(wav = wav, fs = fs, frame_period = frame_period)
        coded_sp = world_encode_spectral_envelop(sp = sp, fs = fs, dim = coded_dim)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)

    return f0s, timeaxes, sps, aps, coded_sps


def transpose_in_list(lst):

    transposed_lst = list()
    for array in lst:       
        transposed_lst.append(array.T)
        
    return transposed_lst


def world_decode_data(coded_sps, fs):

    decoded_sps =  list()

    for coded_sp in coded_sps:
        decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
        decoded_sps.append(decoded_sp)

    return decoded_sps


def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):

    #decoded_sp = decoded_sp.astype(np.float64)
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)

    return wav


def world_synthesis_data(f0s, decoded_sps, aps, fs, frame_period):

    wavs = list()

    for f0, decoded_sp, ap in zip(f0s, decoded_sps, aps):
        wav = world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period)
        wavs.append(wav)

    return wavs


def coded_sps_normalization_fit_transoform(coded_sps):

    coded_sps_concatenated = np.concatenate(coded_sps, axis = 1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis = 1, keepdims = True)
    coded_sps_std = np.std(coded_sps_concatenated, axis = 1, keepdims = True)

    coded_sps_normalized = list()
    for coded_sp in tqdm(coded_sps,desc='Normalizing...'):
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    
    return coded_sps_normalized, coded_sps_mean, coded_sps_std

def coded_sps_normalization_transoform(coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    
    return coded_sps_normalized

def coded_sps_normalization_inverse_transoform(normalized_coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps

def coded_sp_padding(coded_sp, multiple = 4):

    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values = 0)

    return coded_sp_padded

def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1 
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    return wav_padded


def logf0_statistics(f0s):

    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted

def wavs_to_specs(wavs, n_fft = 1024, hop_length = None):

    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)
        stfts.append(stft)

    return stfts


def wavs_to_mfccs(wavs, sr, n_fft = 1024, hop_length = None, n_mels = 128, n_mfcc = 24):

    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y = wav, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, n_mfcc = n_mfcc)
        mfccs.append(mfcc)

    return mfccs


def mfccs_normalization(mfccs):

    mfccs_concatenated = np.concatenate(mfccs, axis = 1)
    mfccs_mean = np.mean(mfccs_concatenated, axis = 1, keepdims = True)
    mfccs_std = np.std(mfccs_concatenated, axis = 1, keepdims = True)

    mfccs_normalized = list()
    for mfcc in mfccs:
        mfccs_normalized.append((mfcc - mfccs_mean) / mfccs_std)
    
    return mfccs_normalized, mfccs_mean, mfccs_std


def sample_train_data(dataset_A, dataset_B, n_frames = 128):

    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:,start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:,start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)                       

    return train_data_A, train_data_B


def sample_nonpair_data(dataset_A, dataset_B, n_frames = 128):

    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:,start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:,start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)                       

    return train_data_A, train_data_B

def sample_data(dataset_A, dataset_B,frames_per_sample=128,perm=[1,0],s_type='parallel',threshold=None):
    
    listnum_A=len(dataset_A)
    listnum_B=len(dataset_B)
    total_sample = min(listnum_A, listnum_B)

    
    dim = len(perm)    
    sample_index = perm.index(0) 
    
    if dim == 2:
        feature_index = perm.index(1)
        A_fea_dim = np.mean([data.shape[feature_index] for idx,data in enumerate(dataset_A)])
        B_fea_dim = np.mean([data.shape[feature_index] for idx,data in enumerate(dataset_B)])
        assert A_fea_dim.is_integer() and B_fea_dim.is_integer()
        A_fea_dim,B_fea_dim=int(A_fea_dim),int(B_fea_dim)
        train_data_A = np.empty([total_sample,A_fea_dim,frames_per_sample])
        train_data_B = np.empty([total_sample,B_fea_dim,frames_per_sample])
        
    elif dim == 3:
        height_index = perm.index(1)
        width_index = perm.index(2)
        channel_index = perm.index(3)
        
        for idx,data in enumerate(dataset_A):
            A_height_dim=np.mean(data.shape[height_index])
            A_width_dim=np.mean(data.shape[width_index])
            A_channel_dim=np.mean(data.shape[channel_index])
            
        for idx,data in enumerate(dataset_B):
            B_height_dim=np.mean(data.shape[height_index])
            B_width_dim=np.mean(data.shape[width_index])
            B_channel_dim=np.mean(data.shape[channel_index])
        
        assert height_dim.is_integer() and width_dim.is_integer() and channel_dim.is_integer()
        A_height_dim,A_width_dim,A_channel_dim=int(A_height_dim),int(A_width_dim),int(A_channel_dim)
        B_height_dim,B_width_dim,B_channel_dim=int(B_height_dim),int(B_width_dim),int(B_channel_dim)
        

        
        train_data_A = np.empty([total_sample,A_height_dim,A_width_dim,A_channel_dim])
        train_data_B = np.empty([total_sample,B_height_dim,B_width_dim,B_channel_dim])


    train_data_A_idx = np.arange(listnum_A)
    train_data_B_idx = np.arange(listnum_B)
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    
    
    for idx,list_idx in enumerate(zip(tqdm(train_data_A_idx[:total_sample],desc='Data Sampling...'),
                                      train_data_B_idx[:total_sample])):
        
        idx_A=list_idx[0]
        idx_B=list_idx[1]
        
        data_A=dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        
        assert frames_A_total >= frames_per_sample
        start_A = np.random.randint(frames_A_total - frames_per_sample + 1)
        end_A = start_A + frames_per_sample
        
        if s_type in ['parallel','Parallel','P','p']:
            assert listnum_A == listnum_B
            data_B=dataset_B[idx_A]                    
            frames_B_total = data_B.shape[1]
            
            assert frames_B_total >= frames_per_sample
            start_B = np.random.randint(frames_B_total - frames_per_sample + 1)          
            end_B = start_B + frames_per_sample
            
            if frames_B_total>=frames_A_total:
                train_data_A[idx,:,:]=data_A[:,start_A:end_A]
                train_data_B[idx,:,:]=data_B[:,start_A:end_A]
                
            elif frames_B_total<frames_A_total:
                train_data_A[idx,:,:]=data_A[:,start_B:end_B]
                train_data_B[idx,:,:]=data_B[:,start_B:end_B]
            
        elif s_type in ['diverse','Diverse','D','d']:
            data_B=dataset_B[idx_B]
            frames_B_total = data_B.shape[1]
            
            assert frames_B_total >= frames_per_sample
            start_B = np.random.randint(frames_B_total - frames_per_sample + 1)
            end_B = start_B + frames_per_sample
            
            train_data_A[idx,:,:]=data_A[:,start_A:end_A]
            train_data_B[idx,:,:]=data_B[:,start_B:end_B]
            
        elif s_type in ['Mix','mix','M','m']:
            
            if idx_A <= threshold: # parallel
                data_B=dataset_B[idx_A]                    
                frames_B_total = data_B.shape[1]
                
                assert frames_B_total >= frames_per_sample
                start_B = np.random.randint(frames_B_total - frames_per_sample + 1)          
                end_B = start_B + frames_per_sample
            
                if frames_B_total>=frames_A_total:
                    train_data_A[idx,:,:]=data_A[:,start_A:end_A]
                    train_data_B[idx,:,:]=data_B[:,start_A:end_A]
                
                elif frames_B_total<frames_A_total:                    
                    train_data_A[idx,:,:]=data_A[:,start_B:end_B]
                    train_data_B[idx,:,:]=data_B[:,start_B:end_B]
                
            elif idx_A > threshold: # diverse    
            
                data_B=dataset_B[idx_B]
                frames_B_total = data_B.shape[1]
            
                assert frames_B_total >= frames_per_sample
                start_B = np.random.randint(frames_B_total - frames_per_sample + 1)
                end_B = start_B + frames_per_sample
            
                train_data_A[idx,:,:]=data_A[:,start_A:end_A]
                train_data_B[idx,:,:]=data_B[:,start_B:end_B]
        
    return train_data_A, train_data_B

class S_control(object):
    
    def __init__(self, array, disp=1):
#        0:sample 1:height 2:width 3:channel        
        
        self.array_origin = array
        self.dim_origin = self.array_origin.ndim
        self.shape_origin = dict(zip(list(range(self.dim_origin)),list(self.array_origin.shape)))

        self.s_update(disp=disp)

    def s_update(self,array_new=None,disp=0):
        
        if not hasattr(self, 'array_temp'):
            self.array_temp = self.array_origin
        if type(array_new) is np.ndarray:   
            self.array_temp = array_new
            
        self.dim_temp = self.array_temp.ndim
        self.shape_temp = dict(zip(list(range(self.dim_temp)),list(self.array_temp.shape)))
            
        self.array_final = self.array_temp
        self.dim_final = self.array_final.ndim
        self.shape_final = dict(zip(list(range(self.dim_final)),list(self.array_final.shape)))
        
        if disp==1:
            current_shape = list(self.shape_final.values())
            current_dim = self.dim_final
            print('Current shape: %s'% (str(current_shape)))    
            print('Current dimension: {}\n'.format(current_dim))
        
        return self.array_final 
        
    def s_transpose(self,perm=None,disp=1):
        
        array_temp = self.array_temp
            
        if perm:
            if len(perm) == self.dim_temp:
                array_temp=array_temp.transpose(perm)
                
            elif len(perm) < self.dim_temp:
                assert max(perm) <= self.dim_origin-1
                
                shape_keys=list(self.shape_temp.keys())
                perm_temp = copy.deepcopy(perm)
                re_shape = list([-1])
                
                for idx,dim_num in enumerate((shape_keys)):
                    if dim_num in perm and idx!=0:                                       
                        re_shape.append(self.shape_temp[dim_num])   
                    elif dim_num not in perm:
                        perm_temp.insert(-1,dim_num)
                        
                array_temp=array_temp.transpose(perm_temp)
                assert array_temp.shape[-1] == self.shape_temp[shape_keys[perm[-1]]]               
                array_temp =  array_temp.reshape(re_shape)
            
        self.array_temp = array_temp
        self.s_update(disp=disp)
#            
        return self.array_final
    
    def s_reshape(self,re_shape=None,disp=1):
        
        array_temp = self.array_temp
        if re_shape:
            array_temp=array_temp.reshape(re_shape)
            
        self.array_temp = array_temp    
        self.s_update(disp=disp)
            
        return self.array_final


