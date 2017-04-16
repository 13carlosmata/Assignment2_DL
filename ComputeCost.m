function J = ComputeCost(P,Y,W1,W2,lambda)
n=size(Y,2);
lcross=0;
tr1=(Y)';
for i=1:n
  lcross=-log(tr1(i,:)*P(:,i))+lcross;
end
J1=lcross/n;
sumW1=sum(W1.^2);
sumW2=sum(W2.^2);
J2=lambda*sum(sumW1)*sum(sumW2);
J=J1+J2;
end