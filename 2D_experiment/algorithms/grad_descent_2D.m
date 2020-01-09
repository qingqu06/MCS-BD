function [Z, F_val, Err] = grad_descent_2D(f, opts)
%   f:    a function object that implements [fval, grad] = f.oracle(x)
Z = opts.Z_init;

tau = opts.tau; %preset stepsize
linesearch_tau = 0.1;

F_val = zeros(opts.MaxIter,1);
Err = zeros(opts.MaxIter,1);

tau_stack = [];

Numinitzed = 1;
iter = 1;
while iter <= opts.MaxIter
    
    [fval, Grad] = f.oracle(Z);
    
    Grad = orth_proj(Z, Grad); % compute Riemannian gradient
    
    if norm(Grad(:)) > 1e2
        Grad = 1e2 * Grad / norm( Grad(:) );
    end
    
    % print result
    F_val(iter) = fval;
    
    if(opts.isprint)
        fprintf('iter = %d, f_val = %f, err = %f ...\n',...
            iter, F_val(iter), Err(iter));
    end
    
    % take a Riemannian gradient step
    if(opts.islinesearch)
        [Z_new,linesearch_tau] = linesearch_2D(f, Z, fval, Grad, linesearch_tau);
        tau_stack = [tau_stack linesearch_tau];
    else
        Z_new = normalize(Z - tau * Grad);
    end
    
    
    
    if(opts.truth)
        Z_p = real(ifft2(  V.* fft2(Z_new)));
        Err(iter) = dist_2D(opts.A_0, Z_p);
        if(Err(iter) <= opts.tol)
            return;
        end
    end
    
    Z = Z_new;
    
    if(opts.truth)
        %re-initialization
        if iter == opts.MaxIter
            if Numinitzed < opts.NumReinit
                iter = 0;
                Z = normc(randn(size(Z)) );
                Numinitzed = Numinitzed +1;
            end
        end
    end
    
    iter = iter + 1;
end
% figure; plot(F_val)
% figure;semilogy(diff_q)

end