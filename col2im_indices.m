function  dX = col2im_indices(dX_col, Sx, h_filter, w_filter, padding, stride)

n_x = Sx(1); d_x = Sx(2); h_x = Sx(3); w_x = Sx(4);
d_filter = d_x;
dX = zeros(n_x, d_x, h_x+2*padding, w_x+2*padding);

out_height = floor((h_x + 2 * padding - h_filter) / stride + 1);
out_width = floor((w_x + 2 * padding - w_filter) / stride + 1);
N = out_height*out_width;
h2l = ceil(h_filter/2-1); h2r = h_filter - h2l-1;
w2l = ceil(w_filter/2-1); w2r = w_filter - w2l-1;
ii = w2l+1;
for i = 1:out_width
    jj = h2l+1;
    for j = 1:out_height    
        k = (i-1)*out_height+j;
        p =   dX_col(:,k:N:end);  % same position for all samples
        p = permute(reshape(p,h_filter,w_filter,d_filter,[]),[4 3 1 2]);
        dX(:,:,jj-h2l:jj+h2r, ii-w2l:ii+w2r) =  dX(:,:,jj-h2l:jj+h2r, ii-w2l:ii+w2r) + p; 
        jj = jj + stride;
    end
    ii = ii + stride;
end
dX = dX(:,:,padding+1:end-padding,padding+1:end-padding);


end
