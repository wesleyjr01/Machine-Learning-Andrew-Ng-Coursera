%%%
clc;clear all;close all;

theta = [1;0.5;3;0.1];
X = [1;2;1.5;0.5];

h = theta'*X %Vectorized Implementation of J, Monovariable Problem

%Para sanar as duvidas para implementar J multivariavel vetorialmente,
%Reassistir a aula Vectorization da Week2