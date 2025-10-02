function [mascara, umbral] = UmbralizaOtsuIntra(ima)
% UMBRALIZAOTSUINTRA - Umbralización por método de Otsu (minimizando varianza intra-clase)
%
% Sintaxis:
%   [mascara, umbral] = UmbralizaOtsuIntra(ima)
%
% Entradas:
%   ima - Imagen en escala de grises (uint8)
%
% Salidas:
%   mascara - Imagen binaria lógica (1 = píxel sobre umbral, 0 = píxel bajo umbral)
%   umbral  - Umbral calculado (en rango [0,255])

    % Asegurarse de que sea imagen en escala de grises
    if size(ima,3) ~= 1
        error('La imagen debe ser en escala de grises (un solo canal).');
    end
    
    % Calcular histograma normalizado
    counts = imhist(ima);  
    p = counts / sum(counts);  % probabilidad de cada nivel [0-255]

    % Variables iniciales
    var_intra = zeros(256,1);

    % Recorrer todos los posibles umbrales
    for t = 1:256
        % Probabilidades de cada clase
        w0 = sum(p(1:t));
        w1 = sum(p(t+1:end));

        if w0 > 0 && w1 > 0
            % Medias de cada clase
            mu0 = sum((0:t-1)'.*p(1:t)) / w0;
            mu1 = sum((t:255)'.*p(t+1:end)) / w1;

            % Varianzas intra-clase
            sigma0 = sum(((0:t-1)'-mu0).^2 .* p(1:t)) / w0;
            sigma1 = sum(((t:255)'-mu1).^2 .* p(t+1:end)) / w1;

            % Varianza intra-clase ponderada
            var_intra(t) = w0*sigma0 + w1*sigma1;
        else
            var_intra(t) = Inf; % umbrales inválidos
        end
    end

    % Seleccionar umbral que minimiza la varianza intra-clase
    [~, umbral] = min(var_intra);
    umbral = umbral - 1; % porque niveles van de 0 a 255

    % Generar máscara binaria
    mascara = ima > umbral;
    mascara = logical(mascara);

end
