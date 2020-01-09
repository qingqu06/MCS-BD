% projection of v to the orthgonal complement of u
% z = v - u* (u'*v)
%

function z = orth_proj(u,v)
   u = u / norm(u);
   z = v - u* (u'*v);
end