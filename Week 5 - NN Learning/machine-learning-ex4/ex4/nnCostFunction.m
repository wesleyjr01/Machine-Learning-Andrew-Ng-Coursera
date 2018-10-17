function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                               
% nn_params = big vector of unrolled theta parameters.

%grad should be returned as a unrolled vector of partial derivatives.

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

         
% You need to return the following variables correctly 
J = 0;
regul = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Setup some useful variables
m = size(X, 1);
n = size(X, 2);
X = [ones(m,1),X];



% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%% ==Transformar y num vetor de '0's e '1's, na variavel y_
y_ = zeros(m,num_labels);
for i=1:m
    for k=1:num_labels
        if y(i)==k
            y_(i,k)=1;
        end
    end
end
%y_(i) - (1Xnum_labels)

%% ==FeedForward // Backprop // Custo(J)
for i=1:m %para cada caso de treinamento
    %ForwardProp
    a1 = X(i,:)';%[401x1]
    z2 = Theta1*a1; %calcula z(k+1)=Theta(k)*a(k) para o i-esimo exemplo, (25x401)x(401x1)
    a2 = sigmoid(z2); %a(k+1) = sigmoid(z(k+1)), k+1 e a camada que recebe esses valores, (25x1)  
    a2 = [1;a2]; %adiciona o bias nesta camada [26x1]
            
    z3 = Theta2*a2; %[10x26]x[26x1]  
    a3 = sigmoid(z3); %camada de saida (10x1)
    
    %Backprop
    delta3 = a3-y_(i,:)'; %[10x1]
    delta2 = Theta2'*delta3;%[26x10]x[10x1] = [26x1]
    delta2 = delta2(2:end).*sigmoidGradient(z2);%[25x1]

    Theta1_grad = Theta1_grad + delta2*a1'; %[25x1]x[1x401] = [25x401]
    Theta2_grad = Theta2_grad + delta3*a2'; %[10x1]x[1x26] - [10x26]

    %Funcao Custo(J) Nao Regularizada
    J = J + (1/m)*((-y_(i,:)*log(a3)-(1-y_(i,:))*log(1-a3))); %J nao-Regularizado escalar
    %y_(i,:) is a [1x10] vector, a3 is a[10x1] vector.Portanto ja estamos calculando J para
    %todos as K saidas em cada i de treinamento.
end
Theta1_grad = (1/m)*Theta1_grad;
Theta2_grad = (1/m)*Theta2_grad;


%% ==Regularizacao do Gradiente
%Regularizando Grad de Theta1
for i=1:size(Theta1_grad,1) %todas as linhas
    for j=2:size(Theta1_grad,2) %todas as colunas, menos a coluna de bias
        Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m)*Theta1(i,j);
    end
end
%Regularizando Grad de Theta2
for i=1:size(Theta2_grad,1) %todas as linhas
    for j=2:size(Theta2_grad,2) %todas as colunas, menos a coluna de bias
        Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m)*Theta2(i,j);
    end
end


%% ==Regularizacao da funcao Custo(J), leva em conta todos os pesos de Theta1 e Theta2,
%menos os termos bias
Theta1_ = Theta1;
Theta2_ = Theta2;
Theta1_(:,1)=0; %Os termos Bias não contribuem na regularizacao
Theta2_(:,1)=0; %Os termos Bias não contribuem na regularizacao

thetaVector = [Theta1_(:);Theta2_(:)]; %unroll all elements and put them into a big vector
regul =thetaVector'*thetaVector;
J = J + (lambda/(2*m))*regul;

%Implementacao da alternativa da Regularizacao usando for's
% for k=1:size(Theta1_,1)%para cada linha de Theta1_
%    regul = regul + Theta1_(k,:)*Theta1_(k,:)'; %[1x401]x[401x1] k vezes.
% end
% 
% for l=1:size(Theta2_,1)%para cada linha de Theta2_   
%     regul= regul + Theta2_(l,:)*Theta2_(l,:)';%[1x26]x[26x1] l vezes.
% end
% J = J + (lambda/(2*m))*regul;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
