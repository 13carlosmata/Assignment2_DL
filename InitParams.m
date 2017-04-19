function [W,b] = InitParams(d,m,K)
mean=0;
std=0.001;
%mean 0 standard variation 0.01
W1mat=zeros(m,d); W1 = mean + std.*randn(m,d);
W2mat=zeros(K,m); W2 = mean + std.*randn(K,m);
b1mat=zeros(m,1); b1 = mean + std.*randn(m,1);
b2mat=zeros(K,1); b2 = mean + std.*randn(K,1);

W = {W1,W2};
b = {b1,b2};
end

