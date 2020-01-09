function err = dist_a(a_0, a)

% % normalization 
 a_0 = normc(a_0); a = normc(a);

n = length(a_0);
% compute correlations
A = a_0;
for i =1:n-1
A = [A circshift(a_0,i)];
end
A = [A -A];

err = min(sqrt(sum((repmat(a,1,2*n) - A).^2)));

end