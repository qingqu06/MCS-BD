classdef func_l4 < func_simple
    properties
        Y      % an n by p observation matrix
    end
    
    methods
        function f = func_l4(Y)
            f.Y = Y;
        end
        
        % 0th and 1st order oracle
        function [fval, grad] = oracle(f, q)
            [n,p] = size(f.Y);
            
            % Return function value f(x) = sum_{i=1}^p ||C_{y_i}q||_4^4
            z = multi_conv(f.Y, q, 0);
            fval = - 1/(n*p) * sum(z(:).^4);
            
            if nargout <= 1; return; end
            
            % compute the Riemannian gradient
            tmp = multi_conv(f.Y, q, 0);
            grad = - 4/(n*p) * sum( multi_conv( reversal(f.Y), tmp.^3, 1) , 2);
            grad =  orth_proj(q, grad);
        end
        
        
    end
end





