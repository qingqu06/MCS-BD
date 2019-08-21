%%
%==========================================================================
%This Demo solves the 2D multi-chanel blind deconvolution problem in the 
%following paper:
%
%``A Nonconvex Approach for Exact and Efficient Multichannel Sparse Blind
% Deconvolution'',
%Qing Qu, Xiao Li, Zhihui Zhu.
%
%where it aims to recovers both the channel $A$ and sparse signals 
%${X_i}_{i=1}^p$ from the measurements $Y_i$. These three terms are 
%linked by the gemerative model:
%
%    $$ Y_i = A * X_i, i = 1,...,p $$
%where '*' represents 2D circulant convolution, the same applies to the sequel. 
%
%This Demo solves the following optimization problem using Riemannian
%gradient descent
%
%    $$ min_{Z} 1/n^2p sum_{i=1}^p H_mu(\overline_{Y}_i*Z), s.t. \|Z\|_F = 1 $$
%where $\overline_{Y}_i$ denotes the preconditioned version of $Y_i$, $H_mu$
%represents huber function with parameter $mu$. Suppose the algorithm 
%returns the optimal solution of the above problem: $Z^\star$, then the
%target kernel and the sparse signals are recovered by 
%
% $$ A^\star = F^{-1}( 1./ ( F(V * Z^\star) ) ), X_i^\star = (Y_i*V)*Z^\star $$
%where $F(.)$ represents fourier tranform.
%==========================================================================

%%
clc;close all;clear all;

folder = fileparts(which(mfilename)); 
addpath(genpath(folder));

%% setting parameters for the problem
n = [10,10]; % size of the kernel
p = 100; % number of samples 
gen_opts.x_type = 'bernoulli-gaussian'; % sparsity pattern
gen_opts.noise_level = 0; % no noise
gen_opts.theta = 0.2; % sparsity level
mu = 1e-2; % the smoothing parameter for huber

%% setting parameter for the algorithm
opts.islinesearch = true;
opts.isprint = false;
opts.tol = 1e-10;
opts.tau = 1e-2;
opts.MaxIter = 2e2;
opts.rounding = true;
opts.NumReinit = 1;
opts.truth = false;
opts.rounding = false;

Z_init = randn(n); 
opts.Z_init = Z_init / norm( Z_init(:) );

loss_type = 'huber'; % choose l1, huber, l4

%% generate and preprocess the data
[ Y, A_0, X_0] = gen_data_2D(n, p, gen_opts); % generate the data

% precondition of the data
V = (1/(n(1)*n(2)*p) * sum( abs(fft2(Y)).^2,3) ).^(-1/2); 
Y_p =  ifft2( bsxfun(@times, fft2(Y), V) );
opts.V = V;
opts.A_0 = A_0;

switch lower(loss_type)
    case 'l1'
        f = func_l1_2D(Y_p);
    case 'huber'
        f = func_huber_2D(Y_p, mu);
    case 'l4'
        f = func_l4_2D(Y_p);
end

%% solving the problem

% phase-1: gradient descent
[Z_r, F_val, Err] = grad_descent_2D(f, opts);
precond_Z = real(ifft2(  V.* fft2(Z_r)));
dist_2D(A_0,precond_Z)



% phase-2: rounding
if opts.rounding
    opts_r.MaxIter = 2e2;
    R = Z_r;
    f = func_l1_2D(Y_p);
    Z = rounding_2D(f, R, opts_r);
    precond_Z = real(ifft2(  V.* fft2(Z)));
    dist_2D(A_0,precond_Z)
else
    Z = Z_r;
end


%% show the results
precond_Z = real(ifft2(  V.* fft2(Z)));
figure
imagesc(abs(cconvfft2(precond_Z, A_0)))
colormap(gray(255));
cbh = colorbar; set(cbh,'YTick',[0,1])
