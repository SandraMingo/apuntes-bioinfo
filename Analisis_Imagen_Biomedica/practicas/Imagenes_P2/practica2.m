% PRÁCTICA 2

% Ej 1

% ---- 1. Cargar la imagen ----
[img, map] = imread('skin_gray_bc_560.tif');

% Verificar si es indexada
if ~isempty(map)
    disp('La imagen es indexada y viene acompañada de una VLT.');
    L = size(map,1); % niveles de la VLT
else
    disp('La imagen NO es indexada.');
    L = double(max(img(:))) + 1; % niveles
end
fprintf('Numero de niveles: %d\n', L);

% ---- 2. Valores de c ----
C = [8, 32, 64, 128];

% ---- 3. Crear figura organizada ----
figure('Position',[100 100 1200 600]);

for ci = 1:length(C)
    c = C(ci);

    % Aproximación 1: modificar pixeles directamente
    img1 = min(double(img) + c, L-1);
    E1 = sum(img1(:).^2); % energía
    subplot(4,length(C),ci);
    imshow(uint8(img1));
    title(sprintf('p2p transf c= %d E= %.2e',c,E1));

    % Aproximación 2: modificar valores con LUT
    T = (0:L-1);
    T2 = min(T + c, L-1);
    img2 = T2(double(img)+1);
    E2 = sum(img2(:).^2);
    subplot(4,length(C),ci+length(C));
    imshow(uint8(img2));
    title(sprintf('val transf c= %d E= %.2e',c,E2));

    % Aproximación 3: modificar VLT
    if ~isempty(map)
        map2 = min(map + c/L, 1); % desplazar colormap
        img3 = ind2rgb(img, map2);
    else
        img3 = ind2rgb(img, gray(L)); % fallback
    end
    E3 = sum(img(:).^2); % energía se mantiene (valores originales)
    subplot(4,length(C),ci+2*length(C));
    imshow(img3);
    title(sprintf('VLT transf c= %d E= %.2e',c,E3));

    % ---- 4. Función de transformación ----
    Tc = min(T + c, L-1);
    subplot(4,length(C),ci+3*length(C));
    plot(T, Tc, 'LineWidth',1.5);
    title(sprintf('funcion s1 : c=%d',c));
    xlim([0 L+50]); ylim([0 L+50]);
end

% Ej. 2

% ---- Imágenes ----
imgs = {'Skin_gray_bw_560.tif', 'Skin_gray_bw_1120.tif'};
C = [8, 32, 64, 128];

for i = 1:length(imgs)
    % Cargar imagen
    img = imread(imgs{i});
    L = double(max(img(:))) + 1;

    % Crear figura para esta imagen
    figure('Name', sprintf('Resultados para %s', imgs{i}), ...
           'Position',[100 100 1200 600]);

    % ---- Procesar para cada c ----
    for ci = 1:length(C)
        c = C(ci);

        % Transformación (opción 1: modificar píxeles directamente)
        imgT = min(double(img) + c, L-1);

        % ---- Mostrar imagen ---- (fila superior)
        subplot(2, length(C), ci);
        imshow(uint8(imgT));
        title(sprintf('Imagen %s, c=%d', imgs{i}, c));

        % ---- Mostrar histograma ---- (fila inferior)
        subplot(2, length(C), ci + length(C));
        imhist(uint8(imgT));
        axis tight;
        title(sprintf('Histograma c=%d', c));
    end
end

%% Ej. 3
[img, map] = imread('skin_gray_bc_560.tif');
imhist(uint8(img));