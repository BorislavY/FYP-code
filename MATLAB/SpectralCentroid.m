[y1,~] = audioread('RealRaw1.wav');
[y2,~] = audioread('RealStem1.wav');
[y3,~] = audioread('ModelMag2_1.wav');
[y4,~] = audioread('ModelMag5_1.wav');
[y5,fs] = audioread('ModelMag12_1.wav');

centroid1 = spectralCentroid(y1,fs);
centroid2 = spectralCentroid(y2,fs);
centroid3 = spectralCentroid(y3,fs);
centroid4 = spectralCentroid(y4,fs);
centroid5 = spectralCentroid(y5,fs);

t = linspace(0,size(y1,1)/fs,size(centroid1,1));

subplot(2,6,2:3);
plot(t,centroid1); xlabel('Seconds'); ylabel('Hz'); ylim([0 7000]);
title('Raw Audio')

subplot(2,6,4:5);
plot(t,centroid2); xlabel('Seconds'); ylabel('Hz'); ylim([0 7000]);
title('Stem Audio')

subplot(2,6,7:8);
plot(t,centroid3); xlabel('Seconds'); ylabel('Hz'); ylim([0 7000]);
title('DB-LSTM_concat_2L_1024C','Interpreter','none')

subplot(2,6,9:10);
plot(t,centroid4); xlabel('Seconds'); ylabel('Hz'); ylim([0 7000]);
title('DB-LSTM_concat_3L_512C_D','Interpreter','none')

subplot(2,6,11:12);
plot(t,centroid5); xlabel('Seconds'); ylabel('Hz'); ylim([0 7000]);
title('LSTM_concat_2L_512C_D','Interpreter','none')