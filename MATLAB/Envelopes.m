[y1,~] = audioread('RealRaw1.wav');
[y2,~] = audioread('RealStem1.wav');
[y3,~] = audioread('ModelMag2_1.wav');
[y4,~] = audioread('ModelMag5_1.wav');
[y5,fs] = audioread('ModelMag12_1.wav');

dt = 1/fs;
t = 0:dt:(length(y1)*dt)-dt;

[upy1,loy1] = envelope(y1, 1024, 'peak');
[upy2,loy2] = envelope(y2, 1024, 'peak');
[upy3,loy3] = envelope(y3, 1024, 'peak');
[upy4,loy4] = envelope(y4, 1024, 'peak');
[upy5,loy5] = envelope(y5, 1024, 'peak');

subplot(1,3,1);
plot(t,upy1,t,loy1,'color','blue','linewidth',1.5);
hold on;
plot(t,upy2,t,loy2,'color','red','linewidth',1.5); 
xlabel('Seconds'); ylabel('Amplitude envelope'); ylim([-1 1]); title('Raw Audio')

subplot(1,3,2);
plot(t,upy1,'color','blue','linewidth',1.5);
hold on;
plot(t,upy2-0.2,'color','red','linewidth',1.5); 
xlabel('Seconds'); ylabel('Amplitude envelope'); ylim([-1 1]); title('Raw Audio')

subplot(1,3,3);
plot(t,y3); xlabel('Seconds'); ylabel('Amplitude envelope'); ylim([-1 1]);
title('DB-LSTM_concat_2L_1024C','Interpreter','none')
