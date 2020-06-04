function X_col = im2col_indices(X, h_filter, w_filter, padding, stride)
% An implementation of im2col based on some fancy indexing
[n_x, d_x, h_x, w_x] = size(X);

out_height = floor((h_x + 2 * padding - h_filter) / stride + 1);
out_width = floor((w_x + 2 * padding - w_filter) / stride + 1);

[colSub, rowSub] = meshgrid(1:stride:h_x+2*padding-(h_filter-1),1:stride:w_x+2*padding-(w_filter-1));
idx = sub2ind([h_x+2*padding-(h_filter-1),w_x+2*padding-(w_filter-1)], rowSub, colSub);

% padd x
x_padded = zeros(n_x,d_x,h_x+2*padding, w_x+2*padding);
x_padded(:,:,padding+1:end-padding,padding+1:end-padding) = X;

% img2col
N = out_height*out_width;
M = h_filter*w_filter;
X_col = zeros(h_filter*w_filter*d_x, n_x*N);

for i = 1:n_x % number of samples 2nd direction
    for j = 1:d_x % dimension first direction
        p = im2col(squeeze(x_padded(i,j,:,:)),[h_filter,w_filter]);
        X_col((j-1)*M+1:j*M,(i-1)*N+1:i*N) = p(:,idx);
    end
end

end
