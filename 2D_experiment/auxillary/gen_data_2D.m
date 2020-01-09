% the function to generate simulated data
% y_i = conv(a, x_i) + noise


function [ Y, A, X] = gen_data_2D(n, p, opts)

X = zeros([n,p]);
Y = zeros([n,p]);

% generate the kernel
A = randn(n); A = A / norm(A,'fro');

% generate the data
switch lower(opts.x_type)
    case 'bernoulli-gaussian'
        X = randn([n,p]) .* (rand([n,p])<=opts.theta);
    case 'bernoulli-radmacher'
        X =  (rand([n,p])<=opts.theta) .* (double(rand([n,p])<0.5) -0.5)*2 ;
end

Y = multi_conv_2D( A, X);

% for k = 1:p
%     switch lower(opts.x_type)
%         case 'bernoulli-gaussian'
%             X(:,k) = randn(n,1) .* (rand(n,1)<=opts.theta);
%         case 'bernoulli-radmacher'
%             X(:,k) =  (rand(m,1)<=opts.theta) .* (double(rand(m,1)<0.5) -0.5)*2 ;
%     end
%     
%     Y(:,k) = cconv(a, X(:,k),n);
% end

Y = Y + opts.noise_level * randn([n,p]);

end








