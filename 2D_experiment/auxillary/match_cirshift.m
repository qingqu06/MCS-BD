function [ A_hat,Index ] = match_cirshift( A0, A_hat );

%A_hat, preconditioned inverse kernel 
%A0, underlying kernel

[m,n] = size(A_hat);

Index = [1 1];
maximum = 0;

for I = 1:m
    for J = 1:n
        A_shift = circshift(A_hat,[I J]);
        cor2 = trace(abs(A0)' * abs(A_shift));
        if cor2 > maximum
            Index = [I,J];
            maximum = cor2;
        end
    end
end
A_hat = circshift(A_hat,Index);

end

