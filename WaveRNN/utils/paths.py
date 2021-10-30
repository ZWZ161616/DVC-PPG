import os
from pathlib import Path
import glob
from natsort import natsorted

class Paths:
    """Manages and configures the paths used by WaveRNN, Tacotron, and the data."""
    def __init__(self, data_path, voc_id, tts_id, latest_weights = '', latest_optim = ''):
        self.base = Path(__file__).parent.parent.expanduser().resolve()

        # Data Paths
        self.data = Path(data_path).expanduser().resolve()
        self.quant = self.data/'quant'
        self.mel = self.data/'mel'
        self.gta = self.data/'gta'

        # WaveRNN/Vocoder Paths
        self.voc_checkpoints = self.base/'model_outputs'/f'{voc_id}'/f'{voc_id}.wavernn'
        if not(os.path.isfile(latest_weights) and os.path.isfile(latest_optim)):
            try:
                latest_weights = natsorted(glob.glob(str(self.voc_checkpoints)+'/*_weights.pyt'))[-1]
                latest_optim  = natsorted(glob.glob(str(self.voc_checkpoints)+'/*_optim.pyt'))[-1]
            except:
                latest_weights = self.voc_checkpoints/'latest_weights.pyt'
                latest_optim = self.voc_checkpoints/'latest_optim.pyt'
        self.voc_latest_weights = Path(latest_weights)
        self.voc_latest_optim = Path(latest_optim)
        
        self.voc_output =  self.base/'model_outputs'/f'{voc_id}'/f'{voc_id}.wavernn'
        self.voc_step = self.voc_output/'step.npy'
        self.voc_log = self.voc_output/'log.txt'

        # Tactron/TTS Paths
        self.tts_checkpoints = self.base/'model_outputs'/f'{voc_id}'/f'{voc_id}.tacotron'
        
        if not(os.path.isfile(latest_weights) and os.path.isfile(latest_optim)): 
            try:
                latest_weights = natsorted(glob.glob(str(self.tts_checkpoints)+'/*_weights.pyt'))[-1]
                latest_optim  = natsorted(glob.glob(str(self.tts_checkpoints)+'/*_optim.pyt'))[-1]
            except:
                latest_weights = self.voc_checkpoints/'latest_weights.pyt'
                latest_optim = self.voc_checkpoints/'latest_optim.pyt'
        self.tts_latest_weights = Path(latest_weights)
        self.tts_latest_optim = Path(latest_optim)
        
        self.tts_output = self.base/'model_outputs'/f'{tts_id}'/f'{voc_id}.tacotron'
        self.tts_step = self.tts_output/'step.npy'
        self.tts_log = self.tts_output/'log.txt'
        self.tts_attention = self.tts_output/'attention'
        self.tts_mel_plot = self.tts_output/'mel_plots'

        self.create_paths()

    def create_paths(self):
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.quant, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.gta, exist_ok=True)
        os.makedirs(self.voc_checkpoints, exist_ok=True)
        os.makedirs(self.voc_output, exist_ok=True)
        os.makedirs(self.tts_checkpoints, exist_ok=True)
        os.makedirs(self.tts_output, exist_ok=True)
        os.makedirs(self.tts_attention, exist_ok=True)
        os.makedirs(self.tts_mel_plot, exist_ok=True)

    def get_tts_named_weights(self, name):
        """Gets the path for the weights in a named tts checkpoint."""
        return self.tts_checkpoints/f'{name}_weights.pyt'

    def get_tts_named_optim(self, name):
        """Gets the path for the optimizer state in a named tts checkpoint."""
        return self.tts_checkpoints/f'{name}_optim.pyt'

    def get_voc_named_weights(self, name):
        """Gets the path for the weights in a named voc checkpoint."""
        return self.voc_checkpoints/f'{name}_weights.pyt'

    def get_voc_named_optim(self, name):
        """Gets the path for the optimizer state in a named voc checkpoint."""
        return self.voc_checkpoints/f'{name}_optim.pyt'


