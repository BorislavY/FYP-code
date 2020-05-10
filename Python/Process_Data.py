import medleydb as mdb
import scipy.io.wavfile as sio
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import pandas as pd
import os
from scipy import signal
import tables


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    # We make a rolling windows of a certain size, pass it through the data, calculate the mean for each
    # step and use that value to determine if the signal needs to be cut out or not. This avoids ruining
    # the data by cutting out every sample which approaches the X axis.
    y_mean = y.rolling(window=int(rate), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# # Initialise .hdf5 files to append to
# testing_init = pd.DataFrame(columns=["raw magnitude", "raw phase", "stem magnitude", "stem phase"]).astype(object)
# training_init = pd.DataFrame(columns=["raw magnitude", "raw phase", "stem magnitude", "stem phase"]).astype(object)
# testing_init.to_hdf(r'B:\MedleyDB\testing.h5', key='1', mode='w', index=False)
# training_init.to_hdf(r'B:\MedleyDB\training.h5', key='1', mode='w', index=False)

multitracks = mdb.load_all_multitracks(dataset_version=['V1', 'V2'])
audio_files = mdb.get_files_for_instrument('acoustic guitar', multitracks)

# Counter to separate the segments of each file under different key in the hdf5 file (if appending new instruments to
# a dataset, make sure to change this value to the next free key in the hdf5, otherwise data will be overwritten.)
i = 1

for stem_item in tqdm(audio_files):

    # Initialise list to store extracted data.
    data = []

    # Converting any files that are not 44.1k to 44.1k and reading audio in
    stem_data, stem_rate = librosa.load(stem_item, sr=44100)
    mask = envelope(stem_data, stem_rate, 0.0003)
    clean_stem = stem_data[mask]

    # Remove replace STEM with RAW in the file path to get to the raw track.
    raw_item = stem_item.replace("STEMS", "RAW").replace("STEM", "RAW")
    raw_item = raw_item[:-4] + "_01.wav"
    raw_data, raw_rate = librosa.load(raw_item, sr=44100)
    clean_raw = raw_data[mask]

    # Detect onsets of the clean file
    onsets = librosa.onset.onset_detect(clean_raw, sr=raw_rate, units='samples', wait=10, pre_avg=10, post_avg=10, pre_max=10, post_max=10)

    # Calculate the shift in samples seconds * sample rate
    sample_shift = 0.030*44100
    end_sample = 0

    # Iterator to limit the number of 10s samples to 20 per track
    limiter = 1

    while True:
        # If there have already been 20 examples taken from this track, move to the next
        if limiter > 20:
            break

        # Handling the instance where there are no more onsets detected after the end of the last segment+shift.
        if end_sample+sample_shift >= np.amax(onsets):
            break

        # Find index of first element just greater than end_sample + sample_shift - next onset sample value
        # +sample shift in order to make sure there is no overlap with previous segment because of shifting back
        next_idx = next(val for idx, val in np.ndenumerate(onsets) if val > end_sample+sample_shift)

        # Move that onset sample value 30ms back to capture start of onset
        next_idx = int(next_idx - sample_shift)

        # Update end sample
        end_sample = next_idx + 441000

        # Check if there are 10s available starting from next_idx
        if end_sample > clean_raw.shape[0]:
            break

        # Take a 10s segment starting from that sample number
        raw_segment = clean_raw[next_idx:end_sample]
        stem_segment = clean_stem[next_idx:end_sample]

        # For loop to pitch shift +-4 semitones
        for semitones in range(-4, 5):
            if semitones != 0:
                # Pitch shift +-4 semitones
                raw_pitch = librosa.effects.pitch_shift(raw_segment, raw_rate, n_steps=float(semitones))
                stem_pitch = librosa.effects.pitch_shift(stem_segment, stem_rate, n_steps=float(semitones))
            else:
                # When no pitch shift is needed
                raw_pitch = raw_segment
                stem_pitch = stem_segment

            # Normalise each segment
            raw_pitch = librosa.util.normalize(raw_pitch)
            stem_pitch = librosa.util.normalize(stem_pitch)

            # Compute STFT of the raw and stem segments
            stft_raw = librosa.core.stft(raw_pitch, n_fft=2048, hop_length=1024)
            stft_stem = librosa.core.stft(stem_pitch, n_fft=2048, hop_length=1024)

            # Compute the corresponding magnitude and phase values (converting from Cartesian to polar coordinates)
            #  The amplitude is encoded as the magnitude of the complex number (sqrt(x^2+y^2))
            #  while the phase is encoded as the angle (atan2(y,x))
            raw_mag = np.sqrt(np.power(np.real(stft_raw), 2) + np.power(np.imag(stft_raw), 2))
            raw_phase = np.arctan2(np.imag(stft_raw), np.real(stft_raw))
            stem_mag = np.sqrt(np.power(np.real(stft_stem), 2) + np.power(np.imag(stft_stem), 2))
            stem_phase = np.arctan2(np.imag(stft_stem), np.real(stft_stem))

            # Storing the values in a dictionary.
            temp_dict = {"raw magnitude": raw_mag, "raw phase": raw_phase,
                         "stem magnitude": stem_mag, "stem phase": stem_phase}

            # Appending the dictionary.
            data.append(temp_dict)

        limiter += 1

    i += 1






