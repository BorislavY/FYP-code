[y, Fs] = audioread("impulse.wav");

L = length(y);      % Signal length

n = 2^nextpow2(L);

Y = fft(y,n);

f = Fs*(0:(n/2))/n;
P = abs(Y/n);

plot(f,P(1:n/2+1)) 
title('Gaussian Pulse in Frequency Domain')
xlabel('Frequency (f)')
ylabel('|P(f)|')