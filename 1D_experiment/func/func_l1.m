classdef func_l1 < func_simple
    properties
        Y      % an n by p observation matrix
    end
    
    methods
        function f = func_l1(Y)
            f.Y = Y;
        end
        
        % 0th and 1st order oracle
        function [fval, grad] = oracle(f, q)
            [n,p] = size(f.Y);
    
            % Return function value f(x) = sum_{i=1}^p ||C_{y_i}q||_1
            z = multi_conv(f.Y, q, 0);
            fval = 1/(n*p) * norm(z(:),1);
            
            if nargout <= 1; return; end
            
            % compute the Riemannian subgradient
            grad = 1/(n*p) * sum( multi_conv( reversal(f.Y), ...
                sign(multi_conv(f.Y, q, 0)), 1) , 2);
            grad = orth_proj(q, grad);
        end
        
        
    end
end





