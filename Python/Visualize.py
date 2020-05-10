import matplotlib.pyplot as plt
import pandas as pd

fig, axs = plt.subplots(3, 3)

ax1 = 0

names = ["DB-LSTM_2L_256C", "DB-LSTM_2L_512C", "DB-LSTM_2L_1024C",
         "DB-LSTM_3L_256C", "DB-LSTM_3L_512C", "DB-LSTM_3L_1024C",
         "DB-LSTM_4L_256C", "DB-LSTM_4L_512C", "DB-LSTM_4L_1024C"]

for i in range(1, 10):

    ax2 = (i+2) % 3

    df = pd.read_csv(r'B:\FYP\Models\old\checkpoints\ModelMag%i\ModelMag%i-log.csv' % (i, i))

    # Plot training & validation loss values
    axs[ax1, ax2].plot(df['loss'])
    axs[ax1, ax2].plot(df['val_loss'])
    axs[ax1, ax2].set_title(names[i-1])
    axs[ax1, ax2].set_ylabel('Loss')
    axs[ax1, ax2].set_xlabel('Epoch')
    axs[ax1, ax2].legend(['Training loss', 'Test loss'], loc='upper right')

    if i % 3 == 0:
        ax1 += 1

plt.show()

