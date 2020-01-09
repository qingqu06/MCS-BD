% projection of V onto orthogonal complement of U
% P = V -  innerprod(U,V)*U / ||U||_F^2 ;

function P = orth_proj(U,V)
 
norm_U_2 = sum(U(:).^2);

P = V - sum(U(:).*V(:)) * U / norm_U_2;

end