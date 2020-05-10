import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import callbacks


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
train_file_paths = []
test_file_paths = []

# Splitting file paths 10:1 for training and testing.
for i in range(1, 120):
    if i % 10 == 0:
        # Appending the file path.
        test_file_paths.append(r'TFRecordsMag/data.tfrecords' + str(i))
    else:
        # Appending the file path.
        train_file_paths.append(r'TFRecordsMag/data.tfrecords' + str(i))

test_dataset = tf.data.TFRecordDataset(test_file_paths)

parsed_test_dataset = test_dataset.map(read_tfrecord).batch(16)

train_dataset = tf.data.TFRecordDataset(train_file_paths)

parsed_train_dataset = train_dataset.map(read_tfrecord).shuffle(2000, reshuffle_each_iteration=True).batch(16).prefetch(1)

# create and fit the LSTM network
model = Sequential()
model.add(layers.Bidirectional(layers.LSTM(512, input_shape=(431, 1025), return_sequences=True, dropout=0.5), merge_mode='sum'))
model.add(layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=0.5), merge_mode='concat'))
model.add(layers.Dense(1025))

cp_callback = callbacks.ModelCheckpoint(filepath=r'checkpoints/ModelMag7/ModelMag7-best.ckpt',
                                        save_weights_only=True, monitor='val_loss', mode='min',
                                        save_best_only=True, verbose=1, save_freq='epoch')

csv_logger = callbacks.CSVLogger(r'checkpoints/ModelMag7/ModelMag7-log.csv')

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))

model.fit(parsed_train_dataset, epochs=150, verbose=2, validation_data=parsed_test_dataset,
          validation_steps=60, callbacks=[cp_callback, csv_logger])

