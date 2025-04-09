import numpy as np
import torch
import torch.nn as nn

def preprocess_audio(n_fft, audio_frame):
        """Compute FFT magnitude features with power compression"""
        analysis_window = np.hamming(n_fft)

        windowed = audio_frame * analysis_window
        fft = np.fft.fft(windowed, n=n_fft)
        magnitude = np.abs(fft)
        # Compress magnitude with power of 0.3
        compressed_mag = magnitude ** 0.3
        
        return compressed_mag


class CompressedSpectralLoss(nn.Module):
    """Implements the compressed spectral loss function described in the paper"""
    def __init__(self, alpha=0.3, beta=0.85, n_fft=128, hop_length=128):
        super(CompressedSpectralLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def forward(self, predicted, target):
        # Delay target by half of the filter taps (4 ms at 16 kHz is approximately 64 samples)
        delay = 64
        target = torch.cat([torch.zeros(target.shape[0], delay).to(target.device), target[:, :-delay]], dim=1)
        
        # Compute STFTs
        p_stft = torch.stft(predicted, n_fft=self.n_fft, hop_length=self.hop_length, 
                           window=torch.hamming_window(self.n_fft).to(predicted.device),
                           return_complex=True)
        t_stft = torch.stft(target, n_fft=self.n_fft, hop_length=self.hop_length,
                           window=torch.hamming_window(self.n_fft).to(target.device),
                           return_complex=True)
        
        # Compute magnitudes
        p_mag = torch.abs(p_stft)
        t_mag = torch.abs(t_stft)
        
        # Compress magnitudes
        p_mag_comp = p_mag ** self.alpha
        t_mag_comp = t_mag ** self.alpha
        
        # Phase component
        p_phase = p_stft / (p_mag + 1e-8)
        t_phase = t_stft / (t_mag + 1e-8)
        
        # Compute compressed complex representations
        p_comp_complex = p_mag_comp * p_phase
        t_comp_complex = t_mag_comp * t_phase
        
        # Magnitude loss
        mag_loss = torch.mean((p_mag_comp - t_mag_comp) ** 2)
        
        # Complex loss
        complex_loss = torch.mean(torch.abs(p_comp_complex - t_comp_complex) ** 2)
        
        # Combined loss
        loss = (1 - self.beta) * mag_loss + self.beta * complex_loss
        
        return loss
