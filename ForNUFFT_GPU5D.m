function ksp = ForNUFFT_GPU5D(param,img)
FT = param.E;
[RO1,LPE,LCH_PCA,tt,cc] = size(param.y);
ksp = single(zeros(RO1*LPE,LCH_PCA,tt,cc));
for i = 1:tt
    for j = 1:cc
        single_img = img(:,:,:,i,j);
        ksp(:,:,i,j) = FT{i}{j}*single_img;
    end
end
ksp = reshape(ksp,[RO1,LPE,LCH_PCA,tt,cc]);