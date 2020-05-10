[y1,~] = audioread('RealRaw1.wav');
[y2,~] = audioread('RealStem1.wav');
[y3,~] = audioread('ModelMag2_1.wav');
[y4,~] = audioread('ModelMag5_1.wav');
[y5,fs] = audioread('ModelMag12_1.wav');

dt = 1/fs;
t = 0:dt:(length(y1)*dt)-dt;

subplot(2,6,2:3);
plot(t,y1); xlabel('Seconds'); ylabel('Amplitude'); ylim([-1 1]);
title('Raw Audio')

subplot(2,6,4:5);
plot(t,y2); xlabel('Seconds'); ylabel('Amplitude'); ylim([-1 1]);
title('Stem Audio')

subplot(2,6,7:8);
plot(t,y3); xlabel('Seconds'); ylabel('Amplitude'); ylim([-1 1]);
title('DB-LSTM_concat_2L_1024C','Interpreter','none')

subplot(2,6,9:10);
plot(t,y4); xlabel('Seconds'); ylabel('Amplitude'); ylim([-1 1]);
title('DB-LSTM_concat_3L_512C_D','Interpreter','none')

subplot(2,6,11:12);
plot(t,y5); xlabel('Seconds'); ylabel('Amplitude'); ylim([-1 1]);
title('LSTM_concat_2L_512C_D','Interpreter','none')