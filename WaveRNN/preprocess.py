import glob
from utils.display import *
from utils.dsp import *
from utils import hparams as hp
from multiprocessing import Pool, cpu_count
from utils.paths import Paths
import pickle
import argparse
from utils.text.recipes import ljspeech, Yu_Mhint, AISHELL
from utils.files import get_files
from pathlib import Path
import numpy as np

global hp
hp.configure('hparams.py')  # Load hparams from file


# Helper functions for argument types
def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n

def convert_file(path: Path):
    y = load_wav(path)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak
    mel = melspectrogram(y)
    if hp.voc_mode == 'RAW':
        if hp.mu_law :
            quant = encode_mu_law(y, mu=2**hp.bits)
        else:
            quant = float_2_label(y, bits=hp.bits)
    elif hp.voc_mode == 'MOL':
        quant = float_2_label(y, bits=16)
    return mel.astype(np.float32), quant.astype(np.int64)
    
def process_wav(path: Path):
    wav_id = path.stem
    m, x = convert_file(path)
    return wav_id, m.shape[-1], m, x

#def process_wav(path : Path):
#    wav_id = path.stem
#    return wav_id

def main():

    parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
    parser.add_argument('--path', '-p', help='directly point to dataset path (overrides hparams.wav_path')
    parser.add_argument('--extension', '-e', metavar='EXT', default='.wav', help='file extension to search for in dataset folder')
    parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-2, help='The number of worker threads to use for preprocessing')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()
    
#    hp.configure(args.hp_file)  # Load hparams from file
    
    if args.path is None:
        args.path = hp.wav_path
    
    extension = args.extension
    path = args.path
    # =============================================================================
    # 
    # =============================================================================
    path = r'E:\Code\Python\WaveRNN\WaveRNN_CCC\Data_raw\LDV'
    save_path = 'LDV2EGG'
    hp.tts_cleaner_names = ['transliteration_cleaners']
    
    # =============================================================================
    # 
    # =============================================================================
    wav_files = get_files(path, extension)
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id, ' ', ' ')
    paths.data = r'.\data_processed/'+save_path
    

    
    print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')
    
    if len(wav_files) == 0:
        print('Please point wav_path in hparams.py to your dataset,')
        print('or use the --path option.\n')
    
    else:
    
        if not hp.ignore_tts:
    
            # text_dict, wav_files = CK(path, wav_files)
            # assert(len(text_dict) == len(wav_files))
            if not Path(paths.data).exists():
                Path(paths.data).mkdir()
            # with open(Path(paths.data)/'text_dict.pkl', 'wb') as f:
            #         pickle.dump(text_dict, f)
        n_workers = max(1, args.num_workers)
    
        simple_table([
            ('Sample Rate', hp.sample_rate),
            ('Bit Depth', hp.bits),
            ('Mu Law', hp.mu_law),
            ('Hop Length', hp.hop_length),
            ('CPU Usage', f'{n_workers}/{cpu_count()}')
        ])
    
        pool = Pool(processes=n_workers)
        dataset = []
    # =============================================================================
    #     
    # =============================================================================
#        wav_files = wav_files
    #    item_id_length = [process_wav(i) for i in wav_files]
    #    
    #    item_id_length = pool.map(process_wav, wav_files[:28])
    
    
#        for i, item_id in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
#            dataset += [item_id]
#            bar = progbar(i, len(wav_files))
#            message = f'{bar} {i}/{len(wav_files)} '
#            stream(message)
#            
#            
#    return dataset
    #    with open(paths.data/'dataset.pkl', 'wb') as f:
    #        pickle.dump(dataset, f)
    #
    #    print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
        
    # =============================================================================
    #         
    # =============================================================================
            
        print('dataset running ...')
        if not Path(paths.data+'/mel/').exists():
            Path(paths.data+'/mel/').mkdir()
        if not Path(paths.data+'/quant/').exists():
            Path(paths.data+'/quant/').mkdir()
        
        for i, (wav_id, length, m, x) in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
            dataset += [(wav_id, length)]
            
            np.save(paths.data+'/mel/'+wav_id+'.npy', m, allow_pickle=False)
            np.save(paths.data+'/quant/'+wav_id+'.npy', x, allow_pickle=False)
#            if i%100 ==0:
#                 print(i,len(wav_files))
            bar = progbar(i, len(wav_files))
            message = f'{bar} {i}/{len(wav_files)} '
            stream(message)
        with open(paths.data+'/dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
            
        print('\n\nCompleted. Ready to run "python train_tacotron.py" or "python train_wavernn.py". \n')
        return dataset#, text_dict
    
    
    
    
if __name__ == '__main__':
    tStart = time.time()#計時開始
    dataset = main()
    tEnd = time.time()#計時結束
    print("It cost %f sec" % (tEnd - tStart))
    