function acc = ComputeAccuracy(X, Y, P)
suma = 0
for i=1:size(X,2)
    [value,index] = max(P(:,i));
    if Y(index,i) == 1
        suma = suma + 1;
    end
end
acc=suma/size(X,2)*100;



