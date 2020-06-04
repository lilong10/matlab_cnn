function [out,max_idx] = maxpool_forward(X,r, padding, stride)

[n,d,h,w] = size(X);
h_filter = r;
w_filter = r;
h_out = floor((h  - h_filter + 2 * padding) / stride) + 1;
w_out = floor((w  - w_filter + 2 * padding) / stride) + 1;

%% forward 
X = reshape(X,n*d,1,h,w);
X_col = im2col_indices(X, h_filter, h_filter, padding, stride);

[out, max_idx] = max(X_col,[],1);
out = reshape(out, h_out,w_out,n,d);
out = permute(out, [3 4 1 2]);


end
