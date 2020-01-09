function [ q, dist2a ] = rounding( r,Y_p,opts);

%rounding using projected subgradient descent
[n,p] = size(Y_p);
Obj = @(h) 1/(n*p) *sum(sum( abs(fft(ifft(Y_p).*repmat(ifft(h),1,p))) ));
q_0 = r; %initialization
q = q_0;
mu = 1; rho = 0.85;  mu_threshold = 1e-15; 
dist2a = [];
iter = 1;
while iter <= 1e2 
    grad = 1/(n*p) * sum( multi_conv( reversal(Y_p),sign(multi_conv(Y_p, q, 0)), 1) , 2);
    %temporary update
    q_new = q - mu*grad;
    %projection
    q_new = q_new - (r'*q_new -1) /norm(q_new,'fro')^2 *r;
    obj_old = Obj(q); obj_new = Obj(q_new);
    grad_norm = norm(grad)^2; 
    alpha = 1e-3;
    if obj_old <= obj_new && mu>mu_threshold
        mu = mu*rho;
    elseif obj_old  >  obj_new && mu>mu_threshold
        q = q_new; %update
        iter = iter +1;
        if  nargin > 2
            precond_q = real(ifft(  opts.precond .* fft(q)));
            a_bar = normc(precond_q); a_bar = normc(real(ifft(1./fft(a_bar))));
            dist2a = [dist2a dist_a(opts.a_0,a_bar)];
        end
    elseif mu < mu_threshold
        q = q_new; %update 
        return;
    end
  
end

end


