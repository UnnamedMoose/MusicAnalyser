# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:52:50 2023

@author: alidtke
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os
import scipy
import pygame

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

# %% Set up.
# filename = "dzaja_palmy.wav"
filename = "Uku3.wav"

# %% Read the wav file.
sampling_rate, audio = scipy.io.wavfile.read(filename)
time = np.arange(0, 1./sampling_rate*len(audio), 1./sampling_rate)

fig, ax = plt.subplots()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Signal")
try:
    ax.plot(time, audio[:, 0], label="Channel 1")
    ax.plot(time, audio[:, 1], label="Channel 2")
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.01))
except IndexError:
    audio = audio[:, np.newaxis]
    ax.plot(time, audio[:, 0], label="Channel 1")

# %% Perform FFT.

# Select a segment that's a few seconds long.
windowed_audio = audio[(time > 5.) & (time < 15.), :]

# Apply a window function to reduce spectral leakage
windowed_audio = windowed_audio[:, 0] * np.hamming(len(windowed_audio))

# Calculate the FFT
fft_result = scipy.fftpack.fft(windowed_audio)

# Calculate the frequency bins corresponding to FFT result
freq_bins = np.fft.fftfreq(len(fft_result), 1.0 / sampling_rate)

# Keep magnitude and positive frequencies only.
fft_result = np.abs(fft_result[freq_bins>0])
freq_bins = freq_bins[freq_bins>0]

# Plot.
# hearing_range = (20, 20e3)
hearing_range = (200, 2e3)  # Make the plots clearer
fft_result = fft_result[(freq_bins >= hearing_range[0]) & (freq_bins <= hearing_range[1])]
freq_bins = freq_bins[(freq_bins >= hearing_range[0]) & (freq_bins <= hearing_range[1])]
fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("FFT")
ax.set_xlim(hearing_range)
ax.plot(freq_bins, fft_result)
ylim = ax.get_ylim()
ax.set_ylim(ylim)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='both', length=0)

# %% Annotate notes.

def generate_note_frequencies(octave):
    # Define the number of semitones per octave
    semitones_per_octave = 12

    # Calculate the frequency of A4 in the target octave
    A_frequency = 440 * (2 ** octave)

    # Define a list of note names
    note_names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

    # Calculate the frequencies for all notes in the specified octave
    note_frequencies = []
    for i, note_name in enumerate(note_names):
        note_frequency = A_frequency * (2 ** (i / semitones_per_octave))
        note_frequencies.append(note_frequency)

    return dict(zip(note_names, note_frequencies))

# Example usage:
octave = 0
notes = generate_note_frequencies(octave)

for note in notes:
    # avoid clutter
    if "#" in note:
        continue
    ax.vlines(notes[note], ylim[0], ylim[1], linestyle="dashed", color="black", alpha=0.5, zorder=-1)
    ax.text(notes[note], ylim[1], note, rotation=0, ha="center")

# %% Playing of sound.
pygame.init()

# Play from data in an array.
duration = 5.  # seconds
# sound = pygame.mixer.Sound(audio[(time > 5.) & (time < 15.), :].tobytes(),
#                            sample_rate=sampling_rate, sample_size=audio.dtype.itemsize)
sound = pygame.mixer.Sound(audio[(time > 5.) & (time < 15.), :])
sound.play()
pygame.time.wait(int(duration * 1000))

# Play from a file.
# pygame.mixer.music.load(filename)
# pygame.mixer.music.play()
# while pygame.mixer.music.get_busy():
#     pass

pygame.quit()
