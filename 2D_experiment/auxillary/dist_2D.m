function err = dist_2D(A_0, A)

% % normalization 
% a_0 = normc(a_0); a = normc(a);

% n = length(a_0);
% compute correlations
% cor = cconv(A_0,A);
cor = cconvfft2(A_0,A);
err = 1 - max(abs(cor(:)))/norm(cor,'fro') ;

end