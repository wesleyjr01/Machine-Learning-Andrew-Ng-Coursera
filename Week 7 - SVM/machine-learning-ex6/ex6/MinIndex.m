function [R C] = MinIndex(X)
%Funcao recebe uma Matrix X quadrada NxN, e retorna em 
%R o indice da linha e em C o indice da coluna do menor elemento
%da matrix X.


Jval=X; %Matriz de valores.
Jval_ = zeros(length(Jval),1);


[M I] = min(Jval);%I vai ser um vetor com o indice do valor minimo de cada coluna de Jval
for i=1:size(Jval,2) %para cada coluna
    Jval_(i)=Jval(I(i),i);%Jval_(i) recebe o valor maximo de cada coluna
end
[M_ I_] = min(Jval_);%I_ diz qual o indice da coluna com o menor valor

[M_2 I_2] = min(Jval(:,I_)); %I_2 recebe o indice da linha com menor custo para coluna I_
R=I_2;
C=I_;



% =========================================================================

end
