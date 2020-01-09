% A = {A_1,A_2, ..., A_p}
% C = {C_1,C_2, ..., C_p}
%
% if flag = 0, B is a matrix
% C_i = conv( A_i, B)
%
% if flag = 1, B is a tensor of the same size of A
% C_i = conv( A_i, B_i)
%
function C = multi_conv_2D(A, B, varargin)

numvararg = numel(varargin);

if numvararg > 1
    error('Too many input arguments.');
end

[~,~,p] = size(A);


A_hat = fft2(A); B_hat = fft2(B);

% if numvararg >= 1 && ~isempty(varargin{1})
%     if strcmp(varargin{1}, 'multi-left')
%         A_hat = repmat( A_hat, 1, 1, p);
%     elseif strcmp(varargin{1}, 'multi-right')
%         B_hat = repmat( B_hat, 1, 1, p);
%     else
%         
%     end
% end

if numvararg >= 1 && ~isempty(varargin{1})
    
    if strcmp(varargin{1}, 'adj-left')
        A_hat = conj(A_hat);
    elseif strcmp(varargin{1}, 'adj-right')
        B_hat = conj(B_hat);
    elseif strcmp(varargin{1}, 'adj-both')
        A_hat = conj(A_hat);
        B_hat = conj(B_hat);
    else
        
    end
end

C = real( ifft2( bsxfun(@times, A_hat, B_hat) ) );
% C = real( ifft2( A_hat .* B_hat ) );
    
end

