% reversal operator reverses each column of Y
% for each column of Y_R,
% y_ri = [y_i1, y_im, y_i(m-1), ..., y_i2]'

function Y_R = reversal(Y)

    Y_R = [Y(1,:);Y(end:-1:2,:)];
    
end