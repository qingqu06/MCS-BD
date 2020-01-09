% Y = [y_1, y_2, ..., y_p]
% C = [c_1, c_2, ..., c_p]
%
% if flag = 0, Q is a vector
% c_i = conv( y_i, Q)
%
% if flag = 1, Q is a matrix of the same size of Y
% c_i = conv( y_i, q_i)
%
function C = multi_conv(Y, Q, flag)
[~,p] = size(Y);
if(flag == 0)
    C = real(ifft( fft(Y) .* repmat(fft(Q),1,p) ));
end

if(flag == 1)
    C = real( ifft( fft(Y) .* fft(Q) ) );
end

end