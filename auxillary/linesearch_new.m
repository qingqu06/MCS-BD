function [q_new,tau] = linesearch_new( f, q, fval, grad,tau)
% line search parameters
beta  = 0.8;
eta = 1e-3;
tau = 2*tau; % initial stepsize
tau_threshold = 1e-15;

grad_norm = sum(grad.^2);
q_new = normc(q - tau*grad);


while( f.oracle(q_new) >  fval - eta * tau * grad_norm && tau>tau_threshold )
    tau = tau * beta;
    q_new = normc(q - tau*grad);
end


end

