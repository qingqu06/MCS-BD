function [ Z ] = rounding_2D( f, U, opts)

%rounding using projected subgradient descent
% Z = size(Y_p);
% Obj = @(h)sum(sum( abs(fft(ifft(Y_p).*repmat(ifft(h),1,p))) ));
Z = U; %initialization

tau = 1; rho = 0.85;  tau_threshold = 1e-15; tol = 0.1;
funcval = [];

for iter = 1 : opts.MaxIter
    
    [fval, Grad] = f.oracle(Z);
    
    % update: projected subgradient
    Z_new = Z - tau * orth_proj(U, Grad);
    
    fval_new = f.oracle(Z_new);
    
    if fval < fval_new && tau > tau_threshold
        tau = tau * rho;
    elseif fval - tol >  fval_new && tau > tau_threshold
        Z = Z_new; %update
%         funcval = [funcval obj_new];
    else
        Z = Z_new; %update 
        return;
    end
end
% plot(funcval)

end

