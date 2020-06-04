
function [out,X_col] = conv_forward(X,W,b,stride, padding)

[n_x,~,h_x,w_x] = size(X);
[n_filter,~,h_filter,w_filter] = size(W);

h_out = floor((h_x - h_filter + 2 * padding) / stride) + 1;
w_out = floor((w_x - w_filter + 2 * padding) / stride) + 1;

% start forward
X_col = im2col_indices(X, h_filter, w_filter, padding, stride);
W_col = reshape(permute(W,[1 3 4 2]),n_filter,[]);

out = W_col * X_col + repmat(b,[1 size(X_col,2)]);
out = reshape(out,n_filter, h_out, w_out, n_x);
out = permute(out,[4,1,2,3]);

end
