function [P,h,s1] = EvaluateClassifier(X, W, b)
K=10;
P=[];
h=[];
s1=[];
n=size(X,2);
for i=1:n
    S1=W{1}*X(:,i)+b{1};
    H=max(0,S1);
    s=W{2}*H+b{2};
    P1=softmax(s);
    P=[P,P1];
    h=[h,H];
end
