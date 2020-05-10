import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential


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


# Creating lists of file paths.
test_file_paths = []

# Splitting file paths 10:1 for training and testing.
for i in range(1, 120):
    if i % 10 == 0:
        # Appending the file path.
        test_file_paths.append(r'B:\MedleyDB\TFRecords\data.tfrecords' + str(i))

test_dataset = tf.data.TFRecordDataset(test_file_paths)

parsed_test_dataset = test_dataset.map(read_tfrecord).batch(64)

model = Sequential()
model.add(layers.Bidirectional(layers.LSTM(512, input_shape=(431, 1025), return_sequences=True), merge_mode='sum'))
model.add(layers.Bidirectional(layers.LSTM(512, return_sequences=True)))
# model.add(layers.BatchNormalization(axis=2))
model.add(layers.Dense(1025))


model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))

model.load_weights(r'Models\Mag2_nobatch_001\real\cp-0005-0.078.ckpt')

loss = model.evaluate(parsed_test_dataset)

print("Model loss: " + str(loss))
