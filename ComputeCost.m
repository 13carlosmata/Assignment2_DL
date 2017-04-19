function [J,J1] = ComputeCost(X,Y,W,b,lambda)
n=size(Y,2);
lcross=0;
tr1=(Y)';
[P,h,s1] = EvaluateClassifier(X,W,b);
for i=1:n
  lcross=-log(tr1(i,:)*P(:,i))+lcross;
end
J1=lcross/n;
sumW1=sum(W{1}.^2);
sumW2=sum(W{2}.^2);
J2=lambda*sum(sumW1)*sum(sumW2);
J=J1+J2;
end