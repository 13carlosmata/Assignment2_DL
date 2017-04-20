function [LW,Lb,JW,Jb] = ComputeGradients(X, Y, P, W, b, h, s1, lambda)
n=size(X,2);
LW{1}=zeros(size(W{1}));
LW{2}=zeros(size(W{2}));
Lb{1}=zeros(size(b{1}));
Lb{2}=zeros(size(b{2}));
for i=1:n
    %individuales
    Pi=P(:,i);
    Yt=(Y(:,i))';
    Xt=(X(:,i))';
    %ops
    g=-Yt/(Yt*Pi) * (diag(Pi)-Pi*(Pi)');
    Lb{2}=Lb{2}+g';
    LW{2}=LW{2}+((g')*h(:,i)');
    
    g=g*W{2};
    g=g*diag(s1(:,i)>0);
    Lb{1}=Lb{1}+g';
    LW{1}=LW{1}+((g')*Xt);
end

LW{1}=LW{1}/n;
LW{2}=LW{2}/n;
Lb{1}=Lb{1}/n;
Lb{2}=Lb{2}/n;
%For regularization
JW{1}=LW{1}+2*lambda*W{1};
JW{2}=LW{2}+2*lambda*W{2};
Jb{1}=Lb{1};
Jb{2}=Lb{2};
end
