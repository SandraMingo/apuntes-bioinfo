%%Práctica 1
% Ejercicio 1

%n=[0:0.05:4-0.05]; 
%m=[0:0.05:4-0.05];

%[N,M]=meshgrid(n,m); % Generación de las matrices N y M
%f=cos(2*pi*N)+sin(6*pi*M); % Definición de la función: x->N, y->M 

%imshow(f,[-2 2],'InitialMagnification',100); % Muestra la función f como una imagen. 

%imshow(f,[min(min(f)) max(max(f))], 'InitialMagnification',100); 
%subplot(1,2,1), imshow(f,[-2 2]);
%subplot(1,2,2), imshow(f,[-1 1]); 

%subplot(1,2,1), imshow(f,[-2 2]); colorbar
%subplot(1,2,2), imshow(f,[-1 1]); colorbar

% 1.1
n=[0:1/256:1-1/256]; 
m=[0:1/256:1-1/256];

[N,M]=meshgrid(n,m);
f1=4*N/5
f2=M/2
f3=2*N/5+M/4

subplot(1,3,1), imshow(f1,[min(min(f1)) max(max(f1))], 'InitialMagnification',100);
subplot(1,3,2), imshow(f2,[min(min(f2)) max(max(f2))], 'InitialMagnification',100);
subplot(1,3,3), imshow(f3,[min(min(f3)) max(max(f3))], 'InitialMagnification',100);

%1.2
n=[0:1/256:1-1/256]; 
m=[0:1/256:1-1/256];

[N,M]=meshgrid(n,m);
f1=cos(2*pi*N)
f2=cos(4*pi*N)
f3=cos(8*pi*N)
f4=cos(16*pi*N)

subplot(1,4,1), imshow(f1,[min(min(f1)) max(max(f1))], 'InitialMagnification',100);
subplot(1,4,2), imshow(f2,[min(min(f2)) max(max(f2))], 'InitialMagnification',100);
subplot(1,4,3), imshow(f3,[min(min(f3)) max(max(f3))], 'InitialMagnification',100);
subplot(1,4,4), imshow(f4,[min(min(f4)) max(max(f4))], 'InitialMagnification',100);

%1.3
n=[0:1/256:1-1/256]; 
m=[0:1/256:1-1/256];

[N,M]=meshgrid(n,m);
f1=cos(2*pi*N) + sin(8*pi*M)
f2=cos(2*pi*N) + sin(16*pi*M)

subplot(1,2,1), imshow(f1,[min(min(f1)) max(max(f1))], 'InitialMagnification',100);
subplot(1,2,2), imshow(f2,[min(min(f2)) max(max(f2))], 'InitialMagnification',100);

%1.4
n=[0:1/256:1-1/256]; 
m=[0:1/256:1-1/256];

[N,M]=meshgrid(n,m);
f1=cos(4*pi*N + 4*pi*M)
f2=cos(4*pi*N + 8*pi*M)

subplot(1,2,1), imshow(f1,[min(min(f1)) max(max(f1))], 'InitialMagnification',100);
subplot(1,2,2), imshow(f2,[min(min(f2)) max(max(f2))], 'InitialMagnification',100);

% Ejercicio 2
% 2.1
n=[0:1/256:1-1/256]; 
m=[0:1/256:1-1/256];

[N,M]=meshgrid(n,m);
f=32*cos(2*pi*N)+16*sin(4*pi*M)

min_f=min(min(f)); max_f=max(max(f)); % Obtengo los extremos
step_f=(max_f-min_f)/256; % Intervalo que corresponde a cada nivel
ima=uint8(round((f-min_f)/step_f)); % Desplazo y escalo 

figure;
%rango óptimo
subplot(2,3,1), imshow(f,[min_f max_f]), colormap(gray);

%rangos sucesivamente más pequeños
range_min = min_f; 
range_max = max_f;
for k = 2:5
    range_min = range_min/2;   % mitad del negro
    range_max = range_max/2;   % mitad del blanco
    subplot(2,3,k), imshow(f,[range_min range_max]), colormap(gray);
end

%sin rango
subplot(2,3,6), imshow(f), colormap(gray);

%2.2
n=[0:1/256:1-1/256]; 
m=[0:1/256:1-1/256];

[N,M]=meshgrid(n,m);
f=32*cos(2*pi*N + 3*pi*M)+16*sin(4*pi*M)

min_f=min(min(f)); max_f=max(max(f)); % Obtengo los extremos
step_f=(max_f-min_f)/256; % Intervalo que corresponde a cada nivel
ima=uint8(round((f-min_f)/step_f)); % Desplazo y escalo 

figure;
subplot(2,3,1), imshow(ima, gray(256));
subplot(2,3,2), imshow(ima,turbo(256));
subplot(2,3,3), imshow(ima, hot(256));
subplot(2,3,4), imshow(ima, jet(256));
subplot(2,3,5), imshow(ima, cool(256));
subplot(2,3,6), imshow(ima, bone(256));

% Ejercicio 3

% MRI pseudo-color (indexada)
[ima1,map1] = imread('MRI_pseudo_colored.jpg');
figure, subplot(2,2,1)
if isempty(map1)
    imshow(ima1)
else
    imshow(ima1,map1)
end
title('MRI pseudo-colored')

% CT abdomen
[ima2,map2] = imread('CT_abdomen.jpg');
subplot(2,2,2)
if isempty(map2)
    imshow(ima2)
else
    imshow(ima2,map2)
end
title('CT abdomen')

% Skin.tif
[ima3,map3] = imread('Skin.tif');
subplot(2,2,3)
if isempty(map3)
    imshow(ima3)
else
    imshow(ima3,map3)
end
title('Skin.tif')

% Xray_th.tif
[ima4,map4] = imread('Xray_th.tif');
subplot(2,2,4)
if isempty(map4)
    imshow(ima4)
else
    imshow(ima4,map4)
end
title('Xray_th.tif')

% 3.2