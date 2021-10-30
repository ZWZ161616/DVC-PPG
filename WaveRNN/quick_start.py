
global hp
from utils import hparams as hp
hp.configure('hparams.py')  # Load hparams from file

import torch
from models.fatchord_version import WaveRNN
from utils.text.symbols import symbols
from models.tacotron import Tacotron
import argparse
from utils.text import text_to_sequence
#from utils.display import save_attention, simple_table
import zipfile, os
import glob
from utils.PINYIN import CH2PY, to_wade_giles
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from utils.display import *
from pathlib import Path
import numpy as np



# os.makedirs('quick_start/tts_weights/', exist_ok=True)
# os.makedirs('quick_start/voc_weights/', exist_ok=True)

# zip_ref = zipfile.ZipFile('pretrained/ljspeech.wavernn.mol.800k.zip', 'r')
# zip_ref.extractall('quick_start/voc_weights/')
# zip_ref.close()

# zip_ref = zipfile.ZipFile('pretrained/ljspeech.tacotron.r2.180k.zip', 'r')
# zip_ref.extractall('quick_start/tts_weights/')
# zip_ref.close()



# =============================================================================
# 
# =============================================================================
if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!', default='what is default ?')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation (lower quality)')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slower Unbatched Generation (better quality)')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py',
                        help='The file to use for the hyperparameters')
    args = parser.parse_args()
    
    # hp.configure(args.hp_file)  # Load hparams from file

    parser.set_defaults(batched=True)
    parser.set_defaults(input_text=None)
    # =============================================================================
    # add
    # =============================================================================
    args.batched = True
    
    converted_dir=r'E:\Kevin\org_gatedcnn\convert_mel\CVA-320_Ho1-288_Ho3-288_Haung-320_Augment(1+3)_time14_r6'
    converted_dir=str(converted_dir)+'\*.npy'
    list_converted=glob.glob(converted_dir)
     
    for filepath in list_converted:
        
    
    #    args.input_text = CH2PY('陳')
        import time
    #    save_name = 'm8_PYT_shuTrim0102_77400_LJ'+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        save_name = 'waveRNN_'+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    #    save_name = 'TTSdict120_76300_waveRNN_'+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    #    word = '祝我們語音轉換成功.'
    #    word = '_而且可以享受這個協會免費提供的鞋子_.'
    #    word = '_老師好_.'
        A=filepath.find('\Ho')
        AA=filepath[A+1:-4]
        print(AA)
        word = '1000k_chain_'+AA
        print(word)
        #word = '1000k_chain_Ho_3_S_15_10DTW_CCC'
        
    #    word = '_祝大家新年快樂_.'
    #    word = '國民黨總統候選人韓國瑜.'
        
    #    word = '我最近進度不  好  我想延畢 .'
    
    #    pinyin1 = CH2PY(word)
        args.input_text = CH2PY(word)
        
        load_tts_weights = r'E:\Kevin\CCC_WaveRNN\pretrained\ljspeech.tacotron.r2.180k/latest_weights.pyt'
    #    load_tts_weights = r'F:\CCC\TTS\WaveRNN-master\model_outputs\TTSdict120\TTSdict120.tacotron/taco_step76300_weights.pyt'
        load_voc_weights = r'E:\Kevin\CCC_WaveRNN\model_outputs\Haung_nodtw_80\Haung_nodtw_80.wavernn\wave_step1000K_weights.pyt'
        
        
        ### 差在數字翻譯成英文
        from utils.text import _clean_text
        
    #    text_trans = _clean_text(args.input_text, hp.tts_cleaner_names)
        print('==============================')
        print('save_name :', save_name)
        print('num_chars :', len(symbols))
    #    print('Tone      :', hp.tone)
    #    print('pyWG      :', hp.pyWG)
        print('tts_cleaner_names  = ', hp.tts_cleaner_names)
        print('text raw       :', word)
        print('text transfer  :', args.input_text)
        print('==============================')
        # =============================================================================
        # 
        # =============================================================================
    
        batched = args.batched
        input_text = args.input_text
    
        if not args.force_cpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print('Using device:', device)
    
        print('\nInitialising WaveRNN Model...\n')
    
        # Instantiate WaveRNN Model
        voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                            fc_dims=hp.voc_fc_dims,
                            bits=hp.bits,
                            pad=hp.voc_pad,
                            upsample_factors=hp.voc_upsample_factors,
                            feat_dims=hp.num_mels,
                            compute_dims=hp.voc_compute_dims,
                            res_out_dims=hp.voc_res_out_dims,
                            res_blocks=hp.voc_res_blocks,
                            hop_length=hp.hop_length,
                            sample_rate=hp.sample_rate,
                            mode='MOL').to(device)
    
        voc_model.load(load_voc_weights)
    
        print('\nInitialising Tacotron Model...\n')
    
        # Instantiate Tacotron Model
        tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
                             num_chars=len(symbols),
                             encoder_dims=hp.tts_encoder_dims,
                             decoder_dims=hp.tts_decoder_dims,
                             n_mels=hp.num_mels,
                             fft_bins=hp.num_mels,
                             postnet_dims=hp.tts_postnet_dims,
                             encoder_K=hp.tts_encoder_K,
                             lstm_dims=hp.tts_lstm_dims,
                             postnet_K=hp.tts_postnet_K,
                             num_highways=hp.tts_num_highways,
                             dropout=hp.tts_dropout,
                             stop_threshold=hp.tts_stop_threshold).to(device)
    
    
        # tts_model.load(load_tts_weights)
    
        if input_text:
            inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
        else:
            with open('sentences.txt') as f:
                inputs = [text_to_sequence(l.strip(), hp.tts_cleaner_names) for l in f]
    
        print(inputs)
        
        voc_k = voc_model.get_step() // 1000
        # tts_k = tts_model.get_step() // 1000
    
        # r = tts_model.r
    
        # simple_table([('WaveRNN', str(voc_k) + 'k'),
        #               (f'Tacotron(r={r})', str(tts_k) + 'k'),
        #               ('Generation Mode', 'Batched' if batched else 'Unbatched'),
        #               ('Target Samples', 11_000 if batched else 'N/A'),
        #               ('Overlap Samples', 550 if batched else 'N/A')])
    
        for i, x in enumerate(inputs, 1):
            # print(x)
            print(f'\n| Generating {i}/{len(inputs)}')
            
            # _, m, attention = tts_model.generate(x)
            # m = np.load(r'E:\Kevin\org_gatedcnn\convert_mel\Ho3_PPG_chain\Ho3_S_15_10.npy').T
            
    #        m = torch.tensor(m).unsqueeze(0)
            # save_attention(attention, Path(f'quick_start/{save_name}_{input_text}_att'))
             
            
            # print('=============================')
            # print('std   : ', np.std(m))
            # print('mean  : ', np.mean(m))
            # print('=============================')
    #        for mstd in [0.32]:
    #        for mstd in [0.25]:
            
    #            for mmean in [0.51]:
                
    #                m_exp = np.exp(m) 
    #                m_exp = m_exp-0.1
    #                m2 = np.log(m_exp)
    #                m=m2
    #                _, m, attention = tts_model.generate(x)
            # m = ((m-np.mean(m))/np.std(m)*mstd)+mmean
            #m = np.load('E:\Kevin\org_gatedcnn\convert_mel\DTW_Ho3_PPG\Ho_3_S_15_10DTW_CCC.npy').T
            m = np.load(filepath).T
            save_spectrogram(m, Path(f'quick_start/CVA-320_Ho1-288_Ho3-288_Haung-320_Augment(1+3)_time14_r6/{save_name}_{word}_spc'), 600)  
    #        m = (1/8)+(m/4)
    #        m = (m-np.mean(m))/8
    #        m = m-np.min(m)
            # print('m std   : ', np.std(m))
            # print('m mean  : ', np.mean(m))
            # print('=============================')
            if input_text:
    #            save_path = f'quick_start/__input_{input_text[:10]}_{tts_k}k.wav'
                save_path = f'quick_start/CVA-320_Ho1-288_Ho3-288_Haung-320_Augment(1+3)_time14_r6/{save_name}_{word}.wav'
            else:
                save_path = f'quick_start/{i}_batched{str(batched)}_{tts_k}k.wav'
    
            m = torch.tensor(m).unsqueeze(0)
    
            print('save path : ',save_path)
            voc_model.generate(m, save_path, batched, 11_000, 550, hp.mu_law)
    
        print('\n\nDone.\n')
