[y1,~] = audioread('RealRaw1.wav');
[y2,~] = audioread('RealStem1.wav');
[y3,~] = audioread('ModelMag2_1.wav');
[y4,~] = audioread('ModelMag5_1.wav');
[y5,fs] = audioread('ModelMag12_1.wav');

A = [y1 y2 y3 y4 y5];
R = corrcoef(A);
