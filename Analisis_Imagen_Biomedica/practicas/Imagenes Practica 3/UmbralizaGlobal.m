function [mascara, umbral] = UmbralizaGlobal(ima)
% UMBRALIZAGLOBAL - Umbralización iterativa global de una imagen en escala de grises
%
% Sintaxis:
%   [mascara, umbral] = UmbralizaGlobal(ima)
%
% Entradas:
%   ima - Imagen en escala de grises (uint8)
%
% Salidas:
%   mascara - Imagen binaria lógica (1 = píxel sobre umbral, 0 = píxel bajo umbral)
%   umbral - Valor de umbral calculado

    % Asegurarse de que sea un solo canal
    if size(ima,3) ~= 1
        error('La imagen debe ser en escala de grises (un solo canal).');
    end

    % Convertir a double para cálculos
    ima = double(ima);

    % Valor inicial del umbral (media global)
    umbral = mean(ima(:));

    % Parámetros de iteración
    diferencia = Inf;
    tol = 0.5; % tolerancia de convergencia

    while diferencia > tol
        % Separar en dos grupos
        G1 = ima(ima > umbral);
        G2 = ima(ima <= umbral);

        % Calcular medias
        m1 = mean(G1(:));
        m2 = mean(G2(:));

        % Nuevo umbral
        nuevo_umbral = (m1 + m2) / 2;

        % Ver diferencia
        diferencia = abs(nuevo_umbral - umbral);

        % Actualizar
        umbral = nuevo_umbral;
    end

    % Generar máscara binaria
    mascara = ima > umbral;

    % Convertir máscara a tipo lógico
    mascara = logical(mascara);

end
