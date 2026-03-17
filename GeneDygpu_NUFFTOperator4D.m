function Dygpu_NUFFTOperator = GeneDygpu_NUFFTOperator4D(k_cs,w_cs,sens1,sw)
[~,RO1,LPE,tt,cc] = size(k_cs);
img_size = size(sens1);
img_size = img_size(1:3);
k_cs = reshape(k_cs,[3,RO1*LPE,tt,cc]);
w_cs = reshape(w_cs,[RO1,LPE,tt,cc]);
Dygpu_NUFFTOperator = cell(tt,cc);
for i = 1:tt
    for j = 1:cc
        k = k_cs(:,:,i,j);
        w = w_cs(:,:,i,j);
        Dygpu_NUFFTOperator{i}{j} = gpuNUFFT(k,w,2,3,sw,img_size,sens1,true);
    end
end