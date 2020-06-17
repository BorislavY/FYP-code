import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


meanM = 0.6960867364825597
stdM = 4.5945316486006

meanP = 0.00304710628603126
stdP = 1.81513728075727

for i in tqdm(range(1, 120)):

    temp_df = pd.read_hdf(r'B:\MedleyDB\dataset.h5', key=str(i))

    x_mag = np.swapaxes(np.array(temp_df['raw magnitude'].tolist()), 1, 2)
    y_mag = np.swapaxes(np.array(temp_df['stem magnitude'].tolist()), 1, 2)
    # x_phase = np.swapaxes(np.array(temp_df['raw phase'].tolist()), 1, 2)
    # y_phase = np.swapaxes(np.array(temp_df['stem phase'].tolist()), 1, 2)

    file_path = r'B:\FYP\Models\TFRecordsMag\data.tfrecords' + str(i)

    with tf.io.TFRecordWriter(file_path) as writer:
        for example in range(1, x_mag.shape[0]):
            mag_raw = x_mag[example]
            mag_stem = y_mag[example]
            # phase_raw = x_phase[example]
            # phase_stem = y_phase[example]

            # Z score normalization of the magnitude
            mag_rawZ = np.divide(mag_raw - meanM, stdM)
            mag_stemZ = np.divide(mag_stem - meanM, stdM)

            # phase_rawZ = np.divide(phase_raw - meanP, stdP)
            # phase_stemZ = np.divide(phase_stem - meanP, stdP)

            # combined_raw = np.append(mag_rawZ, phase_rawZ, axis=1)
            # combined_stem = np.append(mag_stemZ, phase_stemZ, axis=1)

            feature = {
                'x': _bytes_feature(tf.io.serialize_tensor(mag_rawZ)),
                'y': _bytes_feature(tf.io.serialize_tensor(mag_stemZ))
            }

            # Create a Features message using tf.train.Example.
            serialized_example = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
            writer.write(serialized_example)



