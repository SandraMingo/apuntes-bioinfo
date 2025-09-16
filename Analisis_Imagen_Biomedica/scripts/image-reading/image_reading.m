
img = niftiread('C:\Users\Sandra\Documents\GitHub\apuntes_bioinfo\Analisis_Imagen_Biomedica\scripts\image_reading\Template_LPS_sl2_corrected.nii.gz');

% Rotate the image 90 degrees counterclockwise
J = imrotate(img, -90, 'bilinear', 'crop');

% Display the original and rotated images
figure;
imshowpair(img, J, 'montage');
title('Original Image (Left) and Rotated Image (Right)');

%Regular display
% Adjust the display settings for better visualization
colormap(gray);
axis on;
imshow(J, [2,12])

% Apply Gaussian smoothing filters with different standard deviations
Iblur1 = imgaussfilt(J, 2); % Standard deviation = 2
Iblur2 = imgaussfilt(J, 4); % Standard deviation = 4
Iblur3 = imgaussfilt(J, 8); % Standard deviation = 8

% Display the original and blurred images
figure;
subplot(2, 2, 1);
imshow(J, [2,12]);
title('Original Image');

subplot(2, 2, 2);
imshow(Iblur1, [2,12]);
title('Smoothed Image, \sigma = 2');

subplot(2, 2, 3);
imshow(Iblur2, [2,12]);
title('Smoothed Image, \sigma = 4');

subplot(2, 2, 4);
imshow(Iblur3, [2,12]);
title('Smoothed Image, \sigma = 8');

% Adding noise to an image

% Add Gaussian noise
J_norm = J/12;
J_gaussian = imnoise(J_norm, 'gaussian'); % Default mean 0, variance 0.01
J_gaussian_scaled = J_gaussian * 12; % Scale back to uint8 range

% Add salt and pepper noise
J_salt_pepper = imnoise(J_norm, 'salt & pepper', 0.02); % 2% noise density
J_salt_pepper_scaled = J_salt_pepper * 12; % Scale back to uint8 range

% Add Poisson noise
J_poisson = imnoise(J_norm, 'poisson'); % Poisson noise based on image data
J_poisson_scaled = J_poisson * 12; % Scale back to uint8 range

% Display the original and noisy images
figure;
subplot(2, 2, 1);
imshow(J, [2,12]);
title('Original Image');

subplot(2, 2, 2);
imshow(J_gaussian_scaled, [2,12]);
title('Gaussian Noise');

subplot(2, 2, 3);
imshow(J_salt_pepper_scaled, [2,12]);
title('Salt & Pepper Noise');

subplot(2, 2, 4);
imshow(J_poisson_scaled, [2,12]);
title('Poisson Noise');


% now I apply Gaussian filter to previous images
% Apply Gaussian filter with sigma = 2

J_gaussian_filtered = imgaussfilt(J_gaussian_scaled, 2);
J_salt_pepper_filtered = imgaussfilt(J_salt_pepper_scaled, 2);
J_poisson_filtered = imgaussfilt(J_poisson_scaled, 2);

% Display the original and filtered images for gaussian
figure;

subplot(1, 3, 1);
imshow(J, []);
title('Original Image');

subplot(1, 3, 2);
imshow(J_gaussian_scaled, []);
title('Gaussian Noisy Image');

subplot(1, 3, 3);
imshow(J_gaussian_filtered, []);
title('Filtered Gaussian Noise');

% Display the original and filtered images for salt and pepper
figure;

subplot(1, 3, 1);
imshow(J, []);
title('Original Image');

subplot(1, 3, 2);
imshow(J_salt_pepper_scaled, []);
title('Salt & Pepper Noisy Image');

subplot(1, 3, 3);
imshow(J_salt_pepper_filtered, []);
title('Filtered Salt & Pepper Noise');

% Display the original and filtered images for poisson
figure;

subplot(1, 3, 1);
imshow(J, []);
title('Original Image');

subplot(1, 3, 2);
imshow(J_poisson_scaled, []);
title('Poisson Noisy Image');

subplot(1, 3, 3);
imshow(J_poisson_filtered, []);
title('Filtered Poisson Noise');