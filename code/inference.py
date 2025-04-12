import torch
import torch.nn as nn
import numpy as np
from utils import preprocess_audio
from model import load_model
from scipy.io.wavfile import write

class DeepFIREnhancer:
    def __init__(self, model_path=None, sr=16000, n_fft=128, hop_length=128, n_taps=128):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_taps = n_taps
        
        self.half_hann = np.hanning(self.n_taps)
        self.half_hann_first = self.half_hann
        self.half_hann_first[self.n_taps // 2:] = 0
        self.half_hann_second = self.half_hann
        self.half_hann_second[:self.n_taps // 2] = 0
        
        self.prev_fir_taps = None
        
        self.hidden = None
    
    def minimum_phase_conversion(self, fir_taps):
        n_fft = 512
        
        padded_filter = np.zeros(n_fft)
        padded_filter[:self.n_taps] = fir_taps
        
        fft = np.fft.rfft(padded_filter)
        log_spectrum = np.log(np.maximum(np.abs(fft), 1e-10))

        n_half = n_fft // 2
        window = np.ones(n_half + 1)
        window[1:-1] = 2.0
        
        cepstrum = np.fft.irfft(log_spectrum)
        min_phase_cepstrum = cepstrum.copy()
        min_phase_cepstrum[1:n_half] *= 2
        min_phase_cepstrum[n_half:] = 0
        
        min_phase_log_spectrum = np.fft.rfft(min_phase_cepstrum)
        min_phase_spectrum = np.exp(min_phase_log_spectrum)
        min_phase_filter = np.fft.irfft(min_phase_spectrum)[:self.n_taps]
        
        return min_phase_filter
    
    def process_frame(self, frame, use_minimum_phase=False):
        features = preprocess_audio(self.n_fft, frame)

        with torch.no_grad():
            fir_taps, self.hidden = self.model(features, self.hidden)
            fir_taps = fir_taps.squeeze().numpy()
        
        if use_minimum_phase:
            fir_taps = self.minimum_phase_conversion(fir_taps)
        
        if self.prev_fir_taps is None:
            windowed_taps = fir_taps * self.half_hann_second
            filtered_audio = np.convolve(frame, windowed_taps, mode='valid')
        else:
            filtered_prev = np.convolve(frame, self.prev_fir_taps, mode='valid')
            filtered_curr = np.convolve(frame, fir_taps, mode='valid')

            filtered_prev = filtered_prev * self.half_hann_first
            filtered_curr = filtered_curr * self.half_hann_second

            filtered_audio = filtered_prev + filtered_curr

        self.prev_fir_taps = fir_taps
        
        return filtered_audio
    
    def enhance(self, noisy_audio, use_minimum_phase=False):
        self.prev_fir_taps = None
        self.hidden1 = None
        self.hidden2 = None
        enhanced_audio = []
        for i in range(0, len(noisy_audio) - self.n_fft, self.hop_length):
            frame = noisy_audio[i:i + self.hop_length]
            if len(frame) < self.hop_length:
                frame = np.pad(frame, (0, self.hop_length - len(frame)))
            
            filtered_frame = self.process_frame(frame, use_minimum_phase)
            enhanced_audio.extend(filtered_frame)
        
        return np.array(enhanced_audio)


if __name__=="__main__":
    enhancer = DeepFIREnhancer(model_path='models/lstm_filter_dummy.pth')
    noisy_audio = np.random.randn(16000) 
    enhanced_audio = enhancer.enhance(noisy_audio, use_minimum_phase=True)
    
    write("enhanced_audio.wav", enhancer.sr, enhanced_audio.astype(np.float32))
