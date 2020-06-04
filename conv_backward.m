function [dX,dW,db] = conv_backward(dout, X_col, W, Sx, stride, padding)

[n_filter,d_filter,h_filter,w_filter] = size(W);
db = sum(dout,[1 3 4]);
db = reshape(db,n_filter,[]);

dout = reshape(permute(dout,[2 3 4 1]),n_filter,[]);
dW = dout * X_col';
dW = permute(reshape(dW,[n_filter,h_filter,w_filter,d_filter]),[1 4 2 3]);

W_reshape = reshape(permute(W,[1 3 4 2]),n_filter,[]);
dX_col = W_reshape' * dout;

%% do col2im_indices
dX = col2im_indices(dX_col, Sx, h_filter, w_filter, padding, stride);


end
