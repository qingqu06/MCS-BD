classdef func_l4_2D < func_simple
    properties
        Y      % an n by p observation matrix
    end
    
    methods
        function f = func_l4_2D(Y)
            f.Y = Y;
        end
        
        % 0th and 1st order oracle
        function [fval, grad] = oracle(f, Z)
            [n(1),n(2),p] = size(f.Y);
            
            % Return function value f(x) = sum_{i=1}^p ||C_{y_i}q||_4^4
            C = multi_conv_2D(f.Y, Z);
            fval = - 1/(n(1)*n(2)*p) * sum(C(:).^4);
            
            if nargout <= 1; return; end
            
            % compute the Riemannian gradient
            grad = - 4/(n(1)*n(2)*p) * ...
                sum( multi_conv_2D( f.Y, C.^3, 'adj-left'), 3);
%             grad = orth_proj(Z, grad);
        end
        
        
    end
end





