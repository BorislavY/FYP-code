a1 = miraudio('RealRaw3.wav');
a2 = miraudio('RealStem3.wav');
a3 = miraudio('ModelMag2_3.wav');
a4 = miraudio('ModelMag5_3.wav');
a5 = miraudio('ModelMag12_3.wav');

a1_rolloff = mirrolloff(a1);
a2_rolloff = mirrolloff(a2);
a3_rolloff = mirrolloff(a3);
a4_rolloff = mirrolloff(a4);
a5_rolloff = mirrolloff(a5);

a1_brightness = mirbrightness(a1);
a2_brightness = mirbrightness(a2);
a3_brightness = mirbrightness(a3);
a4_brightness = mirbrightness(a4);
a5_brightness = mirbrightness(a5);

a1_inharmonicity = mirinharmonicity(a1);
a2_inharmonicity = mirinharmonicity(a2);
a3_inharmonicity = mirinharmonicity(a3);
a4_inharmonicity = mirinharmonicity(a4);
a5_inharmonicity = mirinharmonicity(a5);

a1_centroid = mircentroid(a1);
a2_centroid = mircentroid(a2);
a3_centroid = mircentroid(a3);
a4_centroid = mircentroid(a4);
a5_centroid = mircentroid(a5);

a1_spread = mirspread(a1);
a2_spread = mirspread(a2);
a3_spread = mirspread(a3);
a4_spread = mirspread(a4);
a5_spread = mirspread(a5);
%If the input is an audio waveform, a file name, or the ‘Folder’ keyword,
%the spread is computed on the spectrum (spectral spread).

a1_skewness = mirskewness(a1);
a2_skewness = mirskewness(a2);
a3_skewness = mirskewness(a3);
a4_skewness = mirskewness(a4);
a5_skewness = mirskewness(a5);

%{
a5_rolloff
a5_brightness
a5_inharmonicity
a5_centroid
a5_spread
a5_skewness
%}