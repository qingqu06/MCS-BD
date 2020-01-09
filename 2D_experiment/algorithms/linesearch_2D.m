function [Z_new,tau] = linesearch_2D( f, Z, fval, Grad, tau)
% line search parameters
beta  = 0.8;
eta = 1e-3;
tau = 2*tau; % initial stepsize
tau_threshold = 1e-12;

Grad_norm = sum(Grad(:).^2);
Z_new = normalize( Z - tau * Grad );


while( f.oracle(Z_new) >  fval - eta * tau * Grad_norm && tau>tau_threshold )
    tau = tau * beta;
    Z_new = normalize(Z - tau*Grad);
end


end

