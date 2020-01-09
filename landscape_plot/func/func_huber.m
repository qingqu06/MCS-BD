classdef func_huber < func_simple
    properties
        Y      % an n by p observation matrix
        mu     % smoothing parameter
    end
    
    methods
        function f = func_huber(Y, mu)
            f.Y  = Y;
            f.mu = mu;
        end
        
        % 0th and 1st order oracle
        function [fval, grad] = oracle(f, q)
            [n,p] = size(f.Y);
            
            % Return function value f(x)
            z = multi_conv(f.Y, q, 0);
            fval = 1/(n*p) * huber(z, f.mu);
            
            if nargout <= 1; return; end
            
            % compute the Riemannian gradient
            
            grad = 1/(n*p) * sum( multi_conv( reversal(f.Y), ...
                huber_grad(z, f.mu), 1), 2);
            grad = orth_proj(q, grad);
            
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

