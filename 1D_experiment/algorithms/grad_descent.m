function [q, F_val, Err,dist2a] = grad_descent(f, opts)
%   f:    a function object that implements [fval, grad] = f.oracle(x)
q = opts.q_init;

tau = opts.tau; %preset stepsize
linesearch_tau = 0.1;

F_val = zeros(opts.MaxIter,1);
Err = zeros(opts.MaxIter,1);
dist2a = zeros(opts.MaxIter,1);

tau_stack = [];

Numinitzed = 1; 
iter = 1;
while iter <= opts.MaxIter
    
    [fval, grad] = f.oracle(q);
    
    if norm(grad) > 1e2
        grad = 1e2*grad/norm(grad);
    end
    % print result
    F_val(iter) = fval;
    precond_q = real(ifft(  opts.precond .* fft(q)));
    a_bar = normc(precond_q); a_bar = normc(real(ifft(1./fft(a_bar))));
    dist2a(iter) = dist_a(opts.a_0,a_bar); 
     
    
    if(opts.isprint)
        fprintf('iter = %d, f_val = %f, err = %f ...\n',...
            iter, F_val(iter), Err(iter));
    end
    
    % take a Riemannian gradient step
    if(opts.islinesearch)
        [q_new,linesearch_tau] = linesearch_new(f, q, fval, grad,linesearch_tau);
        tau_stack = [tau_stack linesearch_tau];
    else
        q_new = normc(q - tau * grad);
    end
    
 
    q = q_new;
    
    
    %re-initialization 
    if iter == opts.MaxIter
       if Numinitzed < opts.NumReinit
           iter = 0;
           q = normc(randn(size(q)) );
           Numinitzed = Numinitzed +1;
       end
    end

    iter = iter + 1;
end
% figure; plot(F_val)
% figure;semilogy(tau_stack)

end