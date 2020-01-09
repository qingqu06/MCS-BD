classdef func_huber_2D_joint < func_simple
    properties
        Y      % an n by p observation matrix
        mu     % smoothing parameter
    end
    
    methods
        function f = func_huber_2D_joint(Y, mu)
            f.Y  = Y;
            f.mu = mu;
        end
        
        % 0th and 1st order oracle
        function [fval, grad] = oracle(f, Z)
            [n(1),n(2),p] = size(f.Y);
    
            % Return function value f(x) = sum_{i=1}^p ||C_{y_i}q||_1
            C = multi_conv_2D(f.Y, Z);
            C_joint = sqrt(sum(C.^2,3)); 
            
            fval = 1/(n(1)*n(2)*p) * huber(C_joint, f.mu);
            
            if nargout <= 1; return; end
            
            % compute the gradient
            
            grad = 1/(n(1)*n(2)*p) * ...
                sum( multi_conv_2D( f.Y, huber_grad(C, f.mu), 'adj-left') , 3);
%             grad = orth_proj(Z, grad);
            
        end
        
        
    end
end

% evaluate the function value of huber
function f_val = huber(z, mu)

h = abs(z) .* (abs(z)>=mu) + (mu/2 + z.^2/2/mu) .* (abs(z)<mu);
f_val = sum(h(:));

end

% evaluate the 1st order deriative of huber
function h = huber_grad(z, mu)

h = sign(z) .* (abs(z)>=mu) + (z/mu) .* (abs(z)<mu);

end

