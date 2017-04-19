function [Wstar,bstar,JK] = MiniBatchGD(X, Y, GD, W,b, lambda)
N=size(X,2)
JK=[];
decay_rate=0.95;
v_W1=zeros(size(W{1})); v_W2=zeros(size(W{2}));
v_b1=zeros(size(b{1})); v_b2=zeros(size(b{2}));

rho=[0.5,0.9,0.99];
for i=1:GD.n_epochs
     fprintf('i = %d\n', i);
    for j=1:N/GD.n_batch
%          fprintf('j = %d\n', j)
        %Composicion de los batches - del pdf
        j_start = (j-1)*GD.n_batch + 1;
        j_end = j*GD.n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, j_start:j_end);
        Ybatch = Y(:, j_start:j_end);
        % calif del batch
        
        [P,h,s1] = EvaluateClassifier(Xbatch, W, b);
%         [P,h,s1] = EvaluateClassifier(Xbatch, W1, W2, b1, b2);
        [LW,Lb,JW,Jb] = ComputeGradients(Xbatch, Ybatch, P, W,b, h, s1, lambda);

%         [LW1,LW2,Lb1,Lb2,JW1,JW2,Jb1,Jb2] = ComputeGradients(Xbatch,Ybatch, P, W1, W2, h, s1, lambda);
        W{1} = W{1} - GD.eta*LW{1}; b{1} = b{1} - GD.eta*Lb{1};
        W{2} = W{2} - GD.eta*LW{2}; b{2} = b{2} - GD.eta*Lb{2};
        
        %Adding the momentum
        
        v_W1=rho(2)*v_W1+GD.eta*JW{1};
        v_W2=rho(2)*v_W2+GD.eta*JW{2};
        v_b1=rho(2)*v_b1+GD.eta*Jb{1};
        v_b2=rho(2)*v_b2+GD.eta*Jb{2};
        
        W{1}=W{1}-v_W1;
        W{2}=W{2}-v_W2;
        b{1}=b{1}-v_b1;
        b{2}=b{2}-v_b2;
        W={W{1},W{2}};
        b={b{1},b{2}};
    end
%     [J,J1] = ComputeCost(X,Y,W1,W2,b1,b2,lambda);
    [J,J1] = ComputeCost(X,Y,W,b,lambda);
    JK = [JK;J];
    W1star=W{1}; b1star=b{1}; W2star=W{2}; b2star=b{2};
    GD.eta=GD.eta*decay_rate;
    
end
Wstar={W1star,W2star};
bstar={b1star,b2star};
end

