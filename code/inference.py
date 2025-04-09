import torch
import torch.nn as nn
import numpy as np
from utils import preprocess_audio
from model import load_model
import argparse

class DeepFIREnhancer:
    def __init__(self, model_path=None, sr=16000, n_fft=128, hop_length=128, n_taps=128):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_taps = n_taps
        
        # Initialize model
        self.model = load_model(model_path)
        
        # Create windows
        self.half_hann = np.hanning(self.n_taps)
        self.half_hann_first = self.half_hann
        self.half_hann_first[self.n_taps // 2:] = 0
        self.half_hann_second = self.half_hann
        self.half_hann_second[:self.n_taps // 2] = 0
        
        # Buffer for previous FIR taps
        self.prev_fir_taps = None
        
        # FIFO buffer for input audio
        # self.fifo_buffer = np.zeros(self.n_fft + self.n_taps - 1)
        
        # LSTM hidden states
        self.hidden = None
    
    def minimum_phase_conversion(self, fir_taps):
        """Convert FIR filter to minimum phase using homomorphic filtering"""
        n_fft = 512  # Use a larger FFT for better accuracy
        
        # Zero-pad the filter
        padded_filter = np.zeros(n_fft)
        padded_filter[:self.n_taps] = fir_taps
        
        # Take FFT and convert to log domain
        fft = np.fft.rfft(padded_filter)
        log_spectrum = np.log(np.maximum(np.abs(fft), 1e-10))
        
        # Create cepstral window (homomorphic filter)
        n_half = n_fft // 2
        window = np.ones(n_half + 1)
        window[1:-1] = 2.0
        
        # Apply homomorphic filter to get minimum phase cepstrum
        cepstrum = np.fft.irfft(log_spectrum)
        min_phase_cepstrum = cepstrum.copy()
        min_phase_cepstrum[1:n_half] *= 2
        min_phase_cepstrum[n_half:] = 0
        
        # Convert back to time domain
        min_phase_log_spectrum = np.fft.rfft(min_phase_cepstrum)
        min_phase_spectrum = np.exp(min_phase_log_spectrum)
        min_phase_filter = np.fft.irfft(min_phase_spectrum)[:self.n_taps]
        
        return min_phase_filter
    
    def process_frame(self, frame, use_minimum_phase=False):
        """Process a single frame of audio"""
        # Update FIFO buffer with new frame
        # self.fifo_buffer = np.roll(self.fifo_buffer, -len(frame))
        # self.fifo_buffer[-len(frame):] = frame
        
        # Extract analysis window
        # analysis_frame = self.fifo_buffer[-self.n_fft:]
        
        print(f"Processing frame with shape: {frame.shape}")

        # Compute features
        features = preprocess_audio(self.n_fft, frame)
        features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # Add batch and time dimensions

        # Move features to the same device as the model
        print(f"Features shape: {features.shape}")

        # Get FIR filter from model
        with torch.no_grad():
            fir_taps, self.hidden = self.model(features, self.hidden)
            fir_taps = fir_taps.squeeze().numpy()
        
        # Convert to minimum phase if required
        if use_minimum_phase:
            fir_taps = self.minimum_phase_conversion(fir_taps)
        
        # If no previous taps, convolve directly
        if self.prev_fir_taps is None:
            # Apply half Hann window to current taps
            windowed_taps = fir_taps * self.half_hann_second
            filtered_audio = np.convolve(frame, windowed_taps, mode='valid')
        else:
            # Convolve both
            filtered_prev = np.convolve(frame, self.prev_fir_taps, mode='valid')
            filtered_curr = np.convolve(frame, fir_taps, mode='valid')

            filtered_prev = filtered_prev * self.half_hann_first
            filtered_curr = filtered_curr * self.half_hann_second

            # Cross-fade the outputs
            filtered_audio = filtered_prev + filtered_curr

        # Store current taps for next frame
        self.prev_fir_taps = fir_taps
        
        return filtered_audio
    
    def enhance(self, noisy_audio, use_minimum_phase=False):
        """Enhance a full audio signal"""
        # Reset states
        self.prev_fir_taps = None
        # self.fifo_buffer = np.zeros(self.n_fft + self.n_taps - 1)
        self.hidden1 = None
        self.hidden2 = None
        
        # Pre-fill buffer with initial samples
        # self.fifo_buffer[-len(noisy_audio[:self.n_fft]):] = noisy_audio[:self.n_fft]
        
        # Process frame by frame
        enhanced_audio = []
        for i in range(0, len(noisy_audio) - self.n_fft, self.hop_length):
            frame = noisy_audio[i:i + self.hop_length]
            if len(frame) < self.hop_length:
                # Pad the last frame if needed
                frame = np.pad(frame, (0, self.hop_length - len(frame)))
            
            filtered_frame = self.process_frame(frame, use_minimum_phase)
            enhanced_audio.extend(filtered_frame)
        
        return np.array(enhanced_audio)


if __name__=="__main__":
    # parser = argparse.ArgumentParser(description="Deep FIR Enhancer")
    # parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    
    # args = parser.parse_args()

    # Load the model
    enhancer = DeepFIREnhancer(model_path='models/lstm_filter_dummy.pth')
    # Load the noisy audio
    noisy_audio = np.random.randn(16000)  # Replace with actual audio loading
    # Enhance the audio
    enhanced_audio = enhancer.enhance(noisy_audio, use_minimum_phase=True)
    
    # Save the enhanced audio as mp3
    from scipy.io.wavfile import write
    write("enhanced_audio.wav", enhancer.sr, enhanced_audio.astype(np.float32))

