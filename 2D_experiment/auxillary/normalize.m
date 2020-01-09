% normalization of a vector Z

function Z_n = normalize(Z)

norm_Z = norm(Z(:));
Z_n = Z / norm_Z;

end