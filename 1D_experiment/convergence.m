%==========================================================================
%This Demo solves the 1D multi-chanel blind deconvolution problem in the 
%following paper:
%
%``A Nonconvex Approach for Exact and Efficient Multichannel Sparse Blind
% Deconvolution'',
%Qing Qu, Xiao Li, Zhihui Zhu.
%
%where it aims to recovers both the channel $a$ and sparse signals 
%${x_i}_{i=1}^p$ from the measurements $y_i$. These three terms are 
%linked by the gemerative model:
%
%    $$ y_i = a * x_i, i = 1,...,p $$
%where '*' represents circulant convolution. 
%
%This Demo solves the following optimization problem using Riemannian
%gradient descent
%
%    $$ min_{q} 1/np sum_{i=1}^p H_mu(C_{y_i}Pq),  s.t. \|q\| = 1 $$
%where $C_{y_i}$ denotes the circulant matrix of vector $y_i$, $H_mu$
%represents huber function with parameter $mu$, and $P$ indicates
%preconditioning matrix introduce in our paper. Suppose the algorithm 
%returns the optimal solution of the above problem: $q^r$, we apply a
%refining rounding processing which solves 
%    
%    $$ min_{q} 1/np sum_{i=1}^p \|C_{y_i}Pq\|_1,  s.t. q_r^Tq = 1 $$
%Suppose the optimal solution of the above problem is given by $q^\star$, 
%then the kernel and the sparse signals are recovered by 
%
% $$ a^\star = F^{-1}( 1./ ( F(Pq^\star) ) ), x_i^\star = C_{y_i}Pq^\star $$
%where $F(.)$ represents fourier tranform.
%==========================================================================




close all;clear;

folder = fileparts(which(mfilename)); 
addpath(genpath(folder));
% randn('seed',2019);
% rand('seed',2019);


% setup parameters

% experiment parameters
gen_opts.x_type = 'bernoulli-gaussian';
p = 50;
n = 200;
gen_opts.theta = .25; % sparsity
gen_opts.noise_level = 0;


% algorithm parameters
opts.islinesearch = true;
opts.isprint = false;
opts.tol = 1e-12;
opts.tau = 1e-3;
opts.MaxIter = 1e2;
opts.NumReinit = 1;


alg = { 'l1', 'huber-0.5','huber-0.05','huber-0.005','l4' };


% generate the data
[ Y, a_0, X_0] = gen_data(n, p, gen_opts);

% generate the initialization and precondition matrix

precond = sqrt(gen_opts.theta *n*p) * ...
    (sum(abs(ifft(Y)).^2 , 2)).^(-1/2); % preconditioning vector

% Y_p = fft( ifft(Y) .* repmat(precond,1,p) ); % preconditioned Y
Y_p =  fft( bsxfun(@times, ifft(Y), precond) );

opts.precond = precond;
opts.a_0 = a_0;

% random initialization
opts.q_init = normc(randn(n,1) );

dist2a = zeros(opts.MaxIter,length(alg));
Err2a = zeros(opts.MaxIter,length(alg));

mu = .1/n; %parameter for huber
for k = 1:length(alg)
    switch lower(alg{k})
        
        case 'l1'
            opts.MaxIter = 2e2;
            f = func_l1(Y_p);
            [r_l1, F_val1,Err,error_l1] = grad_descent( f, opts);
            
        case 'huber-1'
            opts.MaxIter = 1e2;
            mu = 1;
            f = func_huber(Y_p,mu);
            [r_huber_1, F_val2,Err,dist2a(:,k)] = grad_descent( f, opts);
            
        case 'huber-0.5'
            opts.MaxIter = 1e2;
            mu = 5e-1;
            f = func_huber(Y_p,mu);
            [r_huber_2, F_val2,Err,dist2a(:,k)] = grad_descent( f, opts);
        case 'huber-0.05'
            opts.MaxIter = 1e2;
            mu = 5e-2;
            f = func_huber(Y_p,mu);
            [r_huber_3, F_val2,Err,dist2a(:,k)] = grad_descent( f, opts);
        case 'huber-0.005'
            opts.MaxIter = 1e2;
            mu = 5e-3;
            f = func_huber(Y_p,mu);
            [r_huber_4, F_val2,Err,dist2a(:,k)] = grad_descent( f, opts);
            
        case 'l4'
            opts.MaxIter = 1e2;
            f = func_l4(Y_p);
            [r_l4, F_val3,Err,dist2a(:,k)] = grad_descent( f, opts);
            
    end
end

%refining process
for k = 1:length(alg)
    switch lower(alg{k})
        
        case 'l1'       
            
        case 'huber-1'
            [q_huber_1,Err2a(:,k)] = rounding( r_huber_1,Y_p,opts); 
            
        case 'huber-0.5'
            [q_huber_2,Err2a(:,k)] = rounding( r_huber_2,Y_p,opts);
            
        case 'huber-0.05'
            [q_huber_3,Err2a(:,k)] = rounding( r_huber_3,Y_p,opts);
            
        case 'huber-0.005'
            [q_huber_4,Err2a(:,k)] = rounding( r_huber_4,Y_p,opts);
            
        case 'l4'
            [q_l4,Err2a(:,k)] = rounding( r_l4,Y_p,opts);
            
    end
end


%over all error
Error = [dist2a ; Err2a];
Error(:,1) = error_l1;



Markers = {'-+','-o','-*','-x','-v'};
% Colors = {'r','g','b','c','k'};
legen ={'$\ell^1$-loss','Huber-loss, $\mu=5\times 10^{-1}$','Huber-loss, $\mu=5\times 10^{-2}$',...
    'Huber-loss,$\mu=5\times 10^{-3}$','$\ell^4$-loss'};

T = [1:2*opts.MaxIter];
T_s = [1:10:2*opts.MaxIter];

figure;
hold on;
for k = 1:length(alg)
    plot(T,log(Error(:,k)),Markers{k},'LineWidth',2.5,...
        'MarkerIndices', 1:10:length(T),'MarkerSize',8);
end

g = legend(legen);
set(g,'FontSize',16); set(g,'Interpreter','latex');

grid on;
set(gca, 'FontName', 'Times New Roman','FontSize',14);

xlabel('Iteration Number','Interpreter','latex','FontSize',16);
ylabel('$\log ( \min \{||${\boldmath$a$}$_\star-${\boldmath$a$}$ ||\;,||${\boldmath$a$}$_\star + ${\boldmath$a$}$  || \} )$',...
    'Interpreter','latex','FontSize',16);
box on;

ylim([-25,1]);
