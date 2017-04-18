function [W1,W2,b1,b2] = InitParams(d,m,K)
mean=0;
std=0.001;
%mean 0 standard variation 0.01
W1mat=zeros(m,d); W1mat = mean + std.*randn(m,d); W1=num2cell(W1mat);
W2mat=zeros(K,m); W2mat = mean + std.*randn(K,m); W2=num2cell(W2mat);
b1mat=zeros(m,1); b1mat = mean + std.*randn(m,1); b1=num2cell(b1mat);
b2mat=zeros(K,1); b2mat = mean + std.*randn(K,1); b2=num2cell(b2mat); 
end

