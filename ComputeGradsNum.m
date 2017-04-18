function [grad_b, grad_W] = ComputeGradsNum(X, Y, W1,W2,b1,b2,lambda,h)
W=W1;
b=b1;
grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);


[c, ~] = ComputeCost(X, Y, cell2mat(W1),cell2mat(W2),cell2mat(b1),cell2mat(b2), lambda);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, cell2mat(W1),cell2mat(W2),cell2mat(b_try),cell2mat(b2), lambda);
        grad_b{j}(i) = (c2-c) / h;
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        [c2, ~] = ComputeCost(X, Y, cell2mat(W_try),cell2mat(W2),cell2mat(b1),cell2mat(b2), lambda);
        
        grad_W{j}(i) = (c2-c) / h;
    end
end