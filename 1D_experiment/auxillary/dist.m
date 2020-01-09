function err = dist(a_0, a)

% % normalization 
% a_0 = normc(a_0); a = normc(a);

n = length(a_0);
% compute correlations
cor = cconv(a_0,a,n);
err = 1 - max(abs(cor))/norm(cor,'fro') ;

end