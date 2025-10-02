%% EJ 1
clear; close all; clc;

% Leer imagen en escala de grises
ima = imread('MRI_gray.jpg');

% Aplicar función de umbralización
[mascara, umbral] = UmbralizaGlobal(ima);

% Generar imagen enmascarada
ima_mask = uint8(mascara) .* ima;

% ---- Calcular energías (convertir a double solo aquí) ----
E    = sum(double(ima(:)).^2);
E_m  = sum(double(mascara(:)).^2);
E_im = sum(double(ima_mask(:)).^2);

% Mostrar resultados
figure;

subplot(2,3,1);
imshow(ima);
title(sprintf('Imagen original, E = %.2e', E));

subplot(2,3,4);
imhist(ima);   % se queda en uint8 → eje 0–255
xlim([0 255]);
title('Histograma original');

subplot(2,3,2);
imshow(mascara);
title(sprintf('Máscara binaria (Umbral = %.1f), E = %.2e', umbral, E_m));

subplot(2,3,5);
imhist(mascara);   % máscara lógica/double → eje 0–1
xlim([0 1]);
title('Histograma máscara');

subplot(2,3,3);
imshow(ima_mask);
title(sprintf('Imagen enmascarada (Umbral = %.1f), E = %.2e', umbral, E_im));

subplot(2,3,6);
imhist(ima_mask);   % enmascarada uint8 → eje 0–255
xlim([0 255]);
title('Histograma imagen enmascarada');

%% EJ 2
% Umbralización global por Otsu (minimizando varianza intra-clase)
% Comparación con graythresh() de Matlab

clear; close all; clc;

% Leer imagen en escala de grises
ima = imread('MRI_gray.jpg');

% Aplicar Otsu intra (implementación propia)
[mascara, umbral] = UmbralizaOtsuIntra(ima);

% Generar imagen enmascarada
ima_mask = uint8(mascara) .* ima;

% ---- Calcular energías (en double para consistencia) ----
E    = sum(double(ima(:)).^2);
E_m  = sum(double(mascara(:)).^2);
E_im = sum(double(ima_mask(:)).^2);

% ---- Comparar con Otsu Matlab (interclase) ----
th_matlab = graythresh(ima);        % devuelve en [0,1]
umbral_matlab = th_matlab * 255;

fprintf('Umbral propio (intra): %.1f\n', umbral);
fprintf('Umbral Matlab (inter): %.1f\n', umbral_matlab);

% ---- Mostrar resultados en figura 2x3 ----
figure;

% Columna 1: imagen original y su histograma
subplot(2,3,1);
imshow(ima);
title(sprintf('Imagen original, E = %.2e', E));

subplot(2,3,4);
imhist(ima);
xlim([0 255]);
hold on;
y = ylim;
title('Histograma original');

% Columna 2: máscara y su histograma (0–1)
subplot(2,3,2);
imshow(mascara);
title(sprintf('Máscara binaria (U = %.1f), E = %.2e', umbral, E_m));

subplot(2,3,5);
imhist(mascara);
xlim([0 1]);
title('Histograma máscara');

% Columna 3: imagen enmascarada y su histograma (0–255)
subplot(2,3,3);
imshow(ima_mask);
title(sprintf('Imagen enmascarada (U = %.1f), E = %.2e', umbral, E_im));

subplot(2,3,6);
imhist(ima_mask);
xlim([0 255]);
title('Histograma imagen enmascarada');
