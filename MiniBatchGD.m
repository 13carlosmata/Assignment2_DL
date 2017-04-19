function [W1star,W2star,b1star,b2star,JK] = MiniBatchGD(X, Y, GD, W1,W2,b1,b2, lambda)
N=size(X,2);
JK=[];
decay_rate=0.95;
v_W1=zeros(size(W1)); v_W2=zeros(size(W2));
v_b1=zeros(size(b1)); v_b2=zeros(size(b2));
rho=[0.5,0.9,0.99];
for i=1:GD.n_epochs
%     fprintf('i = %d\n', i);
    GD.n_epochs;
    N/GD.n_batch;
    for j=1:N/GD.n_batch
%         fprintf('j = %d\n', j)
        %Composicion de los batches - del pdf
        j_start = (j-1)*GD.n_batch + 1;
        j_end = j*GD.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        % calif del batch
        [P,h,s1] = EvaluateClassifier(Xbatch, W1, W2, b1, b2);
        [LW1,LW2,Lb1,Lb2,JW1,JW2,Jb1,Jb2] = ComputeGradients(Xbatch,Ybatch, P, W1, W2, h, s1, lambda);
        W1 = W1 - GD.eta*LW1; b1 = b1 - GD.eta*Lb1;
        W2 = W2 - GD.eta*LW2; b2 = b2 - GD.eta*Lb2;
        
        %Adding the momentum
        
        v_W1=rho(2)*W1+GD.eta*JW1;
        v_W2=rho(2)*W2+GD.eta*JW2;
        v_b1=rho(2)*b1+GD.eta*Jb1;
        v_b2=rho(2)*b2+GD.eta*Jb2;
    end
    [J,J1] = ComputeCost(X,Y,W1,W2,b1,b2,lambda);
%     J = ComputeCost(X, Y, W, b, lambda);
    JK = [JK;J];
    W1star=v_W1; b1star=v_b1; W2star=v_W2; b2star=v_b2;
    GD.eta=GD.eta*decay_rate;
end
end
