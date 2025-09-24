function ima_res=imfilter_binomial(ima,orden)

triang = cell(1,orden);

triang{1} = [1 1];

for i=2:orden
    v_ant = triang{i-1};
    v=ones(1, i+1);
    v(1)=1;
    v(end)=1;
    for j=2:length(v)-1
        v(j)=v_ant(j-1) + v_ant(j);
    end
    triang{i}=v;
end
    
v= triang{orden};

% Mascara del filtro
v=double(v);
W = v'*v;

% Normalizacion
C = sum(sum(W));
W = W * (1/C);

% Filtrado
ima_res = imfilter(ima,W);