function img = AdjNUFFT_GPU5D(param,kdata_cs,img_size)
FT = param.E;
[RO1,LPE,LCH_PCA,tt,cc] = size(kdata_cs);
kdata_cs = reshape(kdata_cs,[RO1*LPE,LCH_PCA,tt,cc]);
img = single(zeros([img_size,tt,cc]));
for i = 1:tt
    for j = 1:cc
        kdata = kdata_cs(:,:,i,j);
        img(:,:,:,i,j) = FT{i}{j}'*(kdata);
    end
end