[y1,~] = audioread('RealRaw3.wav');
[y2,~] = audioread('RealStem3.wav');
[y3,~] = audioread('ModelMag2_3.wav');
[y4,~] = audioread('ModelMag5_3.wav');
[y5,fs] = audioread('ModelMag12_3.wav');

rms1 = rms(y1);
rms2 = rms(y2);
rms3 = rms(y3);
rms4 = rms(y4);
rms5 = rms(y5);

crest1 = max(abs(y1))/rms1;
crest2 = max(abs(y2))/rms2;
crest3 = max(abs(y3))/rms3;
crest4 = max(abs(y4))/rms4;
crest5 = max(abs(y5))/rms5;

ZCR1 = ZCR(y1);
ZCR2 = ZCR(y2);
ZCR3 = ZCR(y3);
ZCR4 = ZCR(y4);
ZCR5 = ZCR(y5);



