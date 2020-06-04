function dX = maxpool_backward(dout,max_idx,Sx, r,padding, stride)
n = Sx(1); d = Sx(2); h = Sx(3); w = Sx(4);

dX_col = zeros(r*r, numel(dout));
dout_flat = reshape(permute(dout,[3 4 1 2]),[],1);
idx = sub2ind([r*r length(max_idx)], max_idx, 1:length(max_idx));
dX_col(idx) = dout_flat;
 
dX = col2im_indices(dX_col, [n*d,1,h,w], r, r, padding, stride);
dX = reshape(dX,n,d,h,w);


end
