function [P,h,s1] = EvaluateClassifier(X, W1, W2, b1, b2)
K=10;
P=[];
n=size(X,2);
for i=1:n
    s1=W1*X(:,i)+b1;
    h=max(0,s1);
    s=W2*h+b2;
    P1=softmax(s);
    P=[P,P1];
end
