import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import numpy as np
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt


def read_tfrecord(serialized_example):
    feature_description = {
        'x': tf.io.FixedLenFeature((), tf.string),
        'y': tf.io.FixedLenFeature((), tf.string),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    y = tf.io.parse_tensor(example['y'], out_type=tf.float32)

    x.set_shape([None, None])
    y.set_shape([None, None])

    return x, y


mean = 0.6960867364825597
std = 4.5945316486006

example = 60

# Creating lists of file paths.
test_file_path = r'B:\FYP\Models\TFRecordsMag\data.tfrecords' + str(example)

temp_df = pd.read_hdf(r'B:\MedleyDB\dataset.h5', key=str(example))

x_mag = np.array(temp_df['raw magnitude'].tolist())
x_phase = np.array(temp_df['raw phase'].tolist())
y_mag = np.array(temp_df['stem magnitude'].tolist())
y_phase = np.array(temp_df['stem phase'].tolist())

yhat_phase = x_phase[0]

x_mag = x_mag[0]
y_mag = y_mag[0]

x_phase = x_phase[0]
y_phase = y_phase[0]

recon_stft = x_mag * (np.cos(x_phase) + np.multiply(np.sin(x_phase), 1j))
inverse_recon = librosa.core.istft(recon_stft, hop_length=1024)
librosa.output.write_wav(r'Audio\RealRaw3.wav', inverse_recon, 44100)

recon_stft = y_mag * (np.cos(y_phase) + np.multiply(np.sin(y_phase), 1j))
inverse_recon = librosa.core.istft(recon_stft, hop_length=1024)
librosa.output.write_wav(r'Audio\RealStem3.wav', inverse_recon, 44100)

test_dataset = tf.data.TFRecordDataset(test_file_path)

parsed_test_dataset = test_dataset.map(read_tfrecord).batch(1)

fig, axs = plt.subplots(1, 5)


# Model 1
model = Sequential()
model.add(layers.Bidirectional(layers.LSTM(1024, input_shape=(431, 1025), return_sequences=True), merge_mode='concat'))
model.add(layers.Bidirectional(layers.LSTM(1024, return_sequences=True), merge_mode='concat'))
model.add(layers.Dense(1025))

model.load_weights(r'B:\FYP\Models\old2\checkpoints\ModelMag2\ModelMag2-best.ckpt')

yhat = model.predict(parsed_test_dataset, steps=1)
yhat_mag = np.swapaxes(np.squeeze(yhat[:, :, 0:1025]), 0, 1)
yhat_mag = np.multiply(yhat_mag, std) + mean

librosa.display.specshow(librosa.amplitude_to_db(yhat_mag, ref=np.max), y_axis='log', x_axis='time', ax=axs[2])
axs[2].set_title('DB-LSTM_concat_2L_1024C')

recon_stft = yhat_mag * (np.cos(yhat_phase) + np.multiply(np.sin(yhat_phase), 1j))
inverse_recon = librosa.core.istft(recon_stft, hop_length=1024)
librosa.output.write_wav(r'Audio\ModelMag2_3.wav', inverse_recon, 44100)


# Model 2
model = Sequential()
model.add(layers.Bidirectional(layers.LSTM(512, input_shape=(431, 1025), return_sequences=True, dropout=0.5), merge_mode='concat'))
model.add(layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=0.5), merge_mode='concat'))
model.add(layers.Dense(1025))

model.load_weights(r'B:\FYP\Models\old3\checkpoints\ModelMag5\ModelMag5-best.ckpt')

yhat = model.predict(parsed_test_dataset, steps=1)
yhat_mag = np.swapaxes(np.squeeze(yhat[:, :, 0:1025]), 0, 1)
yhat_mag = np.multiply(yhat_mag, std) + mean

librosa.display.specshow(librosa.amplitude_to_db(yhat_mag, ref=np.max), y_axis='log', x_axis='time', ax=axs[3])
axs[3].set_title('DB-LSTM_concat_3L_512C_D')

recon_stft = yhat_mag * (np.cos(yhat_phase) + np.multiply(np.sin(yhat_phase), 1j))
inverse_recon = librosa.core.istft(recon_stft, hop_length=1024)
librosa.output.write_wav(r'Audio\ModelMag5_3.wav', inverse_recon, 44100)


# Model 3
model = Sequential()
model.add(layers.LSTM(512, input_shape=(431, 1025), return_sequences=True, dropout=0.2))
model.add(layers.LSTM(512, return_sequences=True, dropout=0.2))
model.add(layers.Dense(1025))

model.load_weights(r'B:\FYP\Models\checkpoints\ModelMag12\ModelMag12-best.ckpt')

yhat = model.predict(parsed_test_dataset, steps=1)
yhat_mag = np.swapaxes(np.squeeze(yhat[:, :, 0:1025]), 0, 1)
yhat_mag = np.multiply(yhat_mag, std) + mean

librosa.display.specshow(librosa.amplitude_to_db(yhat_mag, ref=np.max), y_axis='log', x_axis='time', ax=axs[4])
axs[4].set_title('LSTM_concat_2L_512C_D')

recon_stft = yhat_mag * (np.cos(yhat_phase) + np.multiply(np.sin(yhat_phase), 1j))
inverse_recon = librosa.core.istft(recon_stft, hop_length=1024)
librosa.output.write_wav(r'Audio\ModelMag12_3.wav', inverse_recon, 44100)


librosa.display.specshow(librosa.amplitude_to_db(x_mag, ref=np.max), y_axis='log', x_axis='time', ax=axs[0])
axs[0].set_title('Log spectrogram raw')

librosa.display.specshow(librosa.amplitude_to_db(y_mag, ref=np.max), y_axis='log', x_axis='time', ax=axs[1])
axs[1].set_title('Log spectrogram stem')

plt.show()
