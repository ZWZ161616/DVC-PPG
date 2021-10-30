from utils.files import get_files
from pathlib import Path
from typing import Union


def ljspeech(path: Union[str, Path]):
    csv_file = get_files(path, extension='.csv')
    assert len(csv_file) == 1
    text_dict = {}
    with open(csv_file[0], encoding='utf-8') as f :
        for line in f :
            split = line.split('|')
            text_dict[split[0]] = split[-1]
    return text_dict

def Yu_Mhint(path: Union[str, Path], total_num):
    csv_file = get_files(path, extension='.txt')
    assert len(csv_file) == 1
    text_dict = {}
#    with open(csv_file[0], encoding='utf-8') as f :
    with open(csv_file[0]) as f :
        for i, line in enumerate(f,1) :
#            split = line.split('\t')
            text_dict[str(i)] = line.replace('\n','')
    len_raw_ = len(text_dict)        
    if len_raw_<total_num:
        for i in range(1,total_num-len_raw_+1):
            text_dict[str(len_raw_+int(i))] = text_dict[str(i)]  
    return text_dict

def AISHELL(path: Union[str, Path], wav_files):
    csv_file = get_files(path, extension='0.txt')
    assert len(csv_file) == 1
    text_dict = {}
#    with open(csv_file[0], encoding='utf-8') as f :
    f = open(csv_file[0],"r",encoding="utf-8")  #注意此行
    data = f.read()
    data_space = data.split('\n')
    for i in data_space:
        text_dict[i.split(' ')[0]] = ''.join(i.split(' ')[1:])
    text_dict2 = {}
    wav_files2 = []
    for i in wav_files:
        try:
            text_dict2[i.stem] = text_dict[i.stem]
            wav_files2.append(i)
        except:
            0
    return text_dict2, wav_files2

def CK(path: Union[str, Path], wav_files):
    csv_file = get_files(path, extension='0.txt')
    assert len(csv_file) == 1
    text_dict = {}
#    with open(csv_file[0], encoding='utf-8') as f :
    f = open(csv_file[0],"r",encoding="utf-8")  #注意此行
    data = f.read()
    data_space = data.split('\n')
    for i in data_space:
        text_dict[i.split(' ')[0]] = ''.join(i.split(' ')[1:])
    text_dict2 = {}
    wav_files2 = []
    for i in wav_files:
        try:
            text_dict2[i.stem] = text_dict[i.stem]
            wav_files2.append(i)
        except:
            0
    return text_dict2, wav_files2