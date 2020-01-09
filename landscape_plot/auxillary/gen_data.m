% the function to generate simulated data
% y_i = conv(a, x_i) + noise


function [ Y, a, X] = gen_data(n, p, opts)

X = zeros(n,p);
Y = zeros(n,p);

% generate the kernel
a = normc(randn(n,1));

% generate the data
switch lower(opts.x_type)
    case 'bernoulli-gaussian'
        X = randn(n,p) .* (rand(n,p)<=opts.theta);
    case 'bernoulli-radmacher'
        X =  (rand(n,p)<=opts.theta) .* (double(rand(n,p)<0.5) -0.5)*2 ;
end

Y = multi_conv( X, a, 0);

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

Y = Y + opts.noise_level * randn(n,p);

end