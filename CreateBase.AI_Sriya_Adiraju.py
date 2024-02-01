#!/usr/bin/env python
# coding: utf-8

# # Objective: To perform a Fast Fourier Transform (FFT) analysis on an audio track and generate a spectrogram -> to process and visualize audio data.

# In[1]:


#Notes: Fast Fourier Transform (FFT) --> 
#Implementation of Discrete Fourier Transform (DFT) --> 
#Input Signal is Discrete and periodic 
#Time domain to frequency domain


# # Data Preperation 

# In[1]:


pip install librosa numpy


# In[2]:


#importing the libraries 
import os  #read and write functions
import librosa   #for audio + music files --> loading + signal processing 
import librosa.display #spectrogram
import IPython.display as ipd  #display audio
import numpy as np #manipulate arrays and matrices 
import matplotlib.pyplot as plt #plotting visualizations


# In[21]:


classical = "Audio_Glasslink.mp3"
classical1 = "Audio_Glasslink.wav"


# In[17]:


ipd.Audio(classical)


# In[23]:


ipd.Audio(classical1)


# In[44]:


classical_signal1 , sr  = librosa.load(classical1)


# In[45]:


len(classical_signal1)


# In[15]:


pip install ffmpeg-python


# # Short term Fourier Transform

# In[81]:


print("Signal length:", len(classical_signal1))


# In[47]:


frame = 2048
overlap = 512


# In[48]:


scale_classical = librosa.stft(classical_signal1, n_fft = frame , hop_length = 512)


# In[49]:


scale_classical.shape


# In[50]:


type(scale_classical[0][0])


# # Calculating the spectrogram

# In[57]:


spectrogram = librosa.amplitude_to_db(abs(scale_classical))


# In[68]:


y_scale = np.abs(scale_classical)**2


# In[69]:


y_scale.shape


# In[70]:


type(y_scale[0][0])


# # Spectrogram 

# In[65]:


def plot_spec(Y, sr, hop_length, y_axis= "linear"):
    plt.figure(figsize=(25,10))
    librosa.display.specshow(Y,
                            sr=sr,
                            hop_length=hop_length,
                            x_axis="time",
                            y_axis=y_axis)
    plt.colorbar(format = "%+2.f")
                             


# In[71]:


plot_spec(y_scale, sr, overlap)  #(without log transformation)
                             


# In[72]:


plot_spec(spectrogram, sr, overlap)     #(with log transformation)

#Findings --> periodic and discrete !!!


# # Digital Fingerprinting Development

# In[80]:


# Extract peaks from the spectrogram using scipy's find_peaks
from scipy.signal import find_peaks
def extract_peaks(spectrogram, threshold=20):
    peaks, _ = find_peaks(spectrogram.max(axis=0), height=threshold, distance=30)
    return peaks

#Extract peaks
peaks = extract_peaks(spectrogram)

# Displaying the spectrogram with highlighted peeaks
plt.figure(figsize=(12, 8))
librosa.display.specshow(spectrogram, sr=sr, hop_length=overlap, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.scatter(peaks, np.argmax(spectrogram[:, peaks], axis=0), color='red', s=10)
plt.title('Spectrogram with Highlighted Peaks')
plt.show()

# basic fingerprint using the obtained peak positions
def construct_fingerprint(peaks):
    fingerprint = [f"{p}" for p in peaks]
    return fingerprint

#fingerprint
audio_fingerprint = construct_fingerprint(peaks)
print("Audio Fingerprint:", audio_fingerprint)


# In[82]:


from scipy.signal import find_peaks

def extract_peaks(spectrogram, threshold=20, distance=30):
    """
    Extract peaks from the spectrogram.

    Parameters:
    - spectrogram (numpy.ndarray): Input spectrogram.
    - threshold (int): Minimum height of peaks.
    - distance (int): Minimum distance between peaks.

    Returns:
    - peaks (numpy.ndarray): Indices of the peaks.
    """
    try:
        peaks, _ = find_peaks(spectrogram.max(axis=0), height=threshold, distance=distance)
        return peaks
    except Exception as e:
        print(f"Error in extract_peaks: {e}")
        return []

def display_spectrogram_with_peaks(spectrogram, peaks, sr, overlap):
    """
    Display spectrogram with highlighted peaks.

    Parameters:
    - spectrogram (numpy.ndarray): Input spectrogram.
    - peaks (numpy.ndarray): Indices of the peaks.
    - sr (int): Sample rate.
    - overlap (int): Hop size for spectrogram calculation.
    """
    try:
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(spectrogram, sr=sr, hop_length=overlap, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.scatter(peaks, np.argmax(spectrogram[:, peaks], axis=0), color='red', s=10)
        plt.title('Spectrogram with Highlighted Peaks')
        plt.show()
    except Exception as e:
        print(f"Error in display_spectrogram_with_peaks: {e}")

def construct_fingerprint(peaks):
    """
    Construct a basic fingerprint using peak positions.

    Parameters:
    - peaks (numpy.ndarray): Indices of the peaks.

    Returns:
    - fingerprint (list): List of peak positions as strings.
    """
    try:
        fingerprint = [f"{p}" for p in peaks]
        return fingerprint
    except Exception as e:
        print(f"Error in construct_fingerprint: {e}")
        return []

# Extract peaks
peaks = extract_peaks(spectrogram, threshold=20, distance=30)

# Display spectrogram with highlighted peaks
display_spectrogram_with_peaks(spectrogram, peaks, sr, overlap)

# Create audio fingerprint
audio_fingerprint = construct_fingerprint(peaks)
print("Audio Fingerprint:", audio_fingerprint)


# In[ ]:




