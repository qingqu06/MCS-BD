clc;close all; clear all;

addpath(genpath(pwd));

% rng(1,'twister'); % fix the seed for random number generation

% parameter settings
isprint = true;

gen_opts.x_type = 'bernoulli-gaussian';
gen_opts.noise_level = 0; % no noise
gen_opts.theta = 0.2; % sparsity level

% tVec = 0:.01:1;
% thetaVec = 0:.01:2*pi;
% numPts = length(tVec) * length(thetaVec);

mu = 0.01;% smoothing parameters
p = 100;% number of samples
n = 3;



% generate random data

[ Y, a, X] = gen_data( n, p, gen_opts);

q1 = real(ifft((fft(a)).^(-1))); q1 = q1/norm(q1);
q2 = circshift(q1,1);
q3 = circshift(q1,2);

f_l1    = func_l1(Y);
f_huber = func_huber(Y,mu);
f_l4    = func_l4(Y);

ss = 1;
% precondition matrix

precond = sqrt(gen_opts.theta *n*p) * ...
    (sum(abs(ifft(Y)).^2 , 2)).^(-1/2); % preconditioning vector

precond_i = precond.^(-1);

q1_p = ifft( precond.^(-1) .* fft(q1) ); q1_p = q1_p /norm(q1_p);
q2_p = circshift(q1_p,1);
q3_p = circshift(q1_p,2);

Y_p = fft( ifft(Y) .* repmat(precond,1,p) ); % preconditioned Y

f_l1_p    = func_l1(Y_p);
f_huber_p = func_huber(Y_p,mu);
f_l4_p     = func_l4(Y_p);


% generate spherical coordinate
U = eye(n);

R = [0:.01:.75, .75:.005:.95, .95:.0005:.99, .99:.0001:1];
T = 0:.005:(2*pi+.05);

rm = max(R);

X = R' * cos(T);
Y = R' * sin(T);
Z = sqrt(max(1 - X.^2 - Y.^2,0));

X = [X; X];
Y = [Y; Y];
Z = [Z; -Z];

[x_1, x_2] = size(X);

fnVals   = zeros([x_1, x_2, 3]); % record function value without preconditioning
fnVals_p = zeros([x_1, x_2, 3]); % record function value with preconditioning

% record function value

for i = 1 : x_1
    for j = 1 : x_2
        
        % print itermediate steps
        if(isprint == true)
%             fprintf('L_x1 = %d, x1 = %d, L_x2 = %d, x2 = %d...\n',...
%                 x_1, i, x_2, j);
        end
        
        q = [X(i,j); Y(i,j); Z(i,j)];
        
        % evaluate l1, huber, l4 loss value
        fnVals(i,j,1) = f_l1.oracle(q);
        fnVals(i,j,2) = f_huber.oracle(q);
        fnVals(i,j,3) = f_l4.oracle(q);
        
        
        % evaluate preconditioned l1, huber, l4 loss value
        fnVals_p(i,j,1) = f_l1_p.oracle(q);
        fnVals_p(i,j,2) = f_huber_p.oracle(q);
        fnVals_p(i,j,3) = f_l4_p.oracle(q);
        
    end
end

% normalize the function value
for t = 1:3
    tmp_min = fnVals(:,:,t);
    fnVals(:,:,t) = fnVals(:,:,t) - min(tmp_min(:));
    tmp_max = fnVals(:,:,t);
    fnVals(:,:,t) = fnVals(:,:,t) / max(tmp_max(:));
    
    tmp_min = fnVals_p(:,:,t);
    fnVals_p(:,:,t) = fnVals_p(:,:,t) - min(tmp_min(:));
    tmp_max = fnVals_p(:,:,t);
    fnVals_p(:,:,t) = fnVals_p(:,:,t) / max(tmp_max(:));
    
end

% save('data.mat','q1','q2','q3','q1_p','q2_p','q3_p','X','Y','Z','fnVals','fnVals_p');

% plot the landscape over 3D sphere
r = 1.005;
Marker = 25;
text = {'l1','huber','l4'};
text_p = {'l1-p','huber-p','l4-p'};

for t = 1:3
    figure;
    surf(X,Y,Z,fnVals(:,:,t),'EdgeAlpha',0);
    axis off; axis equal;
    title(text{t});
    
    hold on;
    plot3(r*q1(1),r*q1(2),r*q1(3),'r.','MarkerSize',Marker);
    plot3(r*q2(1),r*q2(2),r*q2(3),'r.','MarkerSize',Marker);
    plot3(r*q3(1),r*q3(2),r*q3(3),'r.','MarkerSize',Marker);
    plot3(-r*q1(1),-r*q1(2),-r*q1(3),'r.','MarkerSize',Marker);
    plot3(-r*q2(1),-r*q2(2),-r*q2(3),'r.','MarkerSize',Marker);
    plot3(-r*q3(1),-r*q3(2),-r*q3(3),'r.','MarkerSize',Marker);
    
    figure;
    surf(X,Y,Z,fnVals_p(:,:,t),'EdgeAlpha',0);
    axis off; axis equal;
    title(text_p{t});
    
    hold on;
    
    plot3(r*q1_p(1),r*q1_p(2),r*q1_p(3),'r.','MarkerSize',Marker);
    plot3(r*q2_p(1),r*q2_p(2),r*q2_p(3),'r.','MarkerSize',Marker);
    plot3(r*q3_p(1),r*q3_p(2),r*q3_p(3),'r.','MarkerSize',Marker);
    plot3(-r*q1_p(1),-r*q1_p(2),-r*q1_p(3),'r.','MarkerSize',Marker);
    plot3(-r*q2_p(1),-r*q2_p(2),-r*q2_p(3),'r.','MarkerSize',Marker);
    plot3(-r*q3_p(1),-r*q3_p(2),-r*q3_p(3),'r.','MarkerSize',Marker);
    
end


% figure(1);
%
% r = 1.005;
% Marker = 15;
%
% hold on;
% surf(X,Y,Z,F_val,'EdgeAlpha',0);
% axis off; axis equal;
%
% plot3(r*u1'*a0,r*u2'*a0,r*u3'*a0,'r.','MarkerSize',Marker);
% plot3(r*u1'*a1,r*u2'*a1,r*u3'*a1,'r.','MarkerSize',Marker);
% plot3(r*u1'*a2,r*u2'*a2,r*u3'*a2,'r.','MarkerSize',Marker);
% plot3(-r*u1'*a0,-r*u2'*a0,-r*u3'*a0,'r.','MarkerSize',Marker);
% plot3(-r*u1'*a1,-r*u2'*a1,-r*u3'*a1,'r.','MarkerSize',Marker);
% plot3(-r*u1'*a2,-r*u2'*a2,-r*u3'*a2,'r.','MarkerSize',Marker);






