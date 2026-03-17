function [recon_nufft,recon1_CS]=LiverRecon_4D(kdata1_,kdata_SI,sens1,k_bart,w_bart,Res_Signal,Card_Sig,reconParam)
idxGroup = reconParam.idxGroup;
nframe = reconParam.nframe;
ncardiac = reconParam.ncardiac;
cardIdx = reconParam.cardIdx;
seqParam = reconParam.seqParam;
prin = reconParam.prin;
img_size = size(sens1(:,:,:,1));
kdata_bart = kdata1_;
%% Data sorting based on SI signal
[nx,~,nc] = size(kdata_bart);
spk = seqParam.spk;
seg = seqParam.seg;
kdata_bart = reshape(kdata_bart,nx,spk,seg,nc);
k_bart = reshape(k_bart,3,nx,spk,seg);
w_bart = reshape(w_bart,nx,spk,seg);
nline = floor(seg/nframe);
nlineC = floor(floor(seg/nframe)/ncardiac);
kdata_cs = zeros(nx,spk,nlineC,nc,nframe,ncardiac);
k_bart_u = zeros(3,nx,spk,nlineC,nframe,ncardiac);
w_bart_u = zeros(nx,spk,nlineC,nframe,ncardiac);
[~,indexR] = sort(Res_Signal,'descend');
kdata_bart=kdata_bart(:,:,indexR,:);
k_bart=k_bart(:,:,:,indexR);
w_bart=w_bart(:,:,indexR);
for ii = 1 : nframe    
    kdatauTmp = kdata_bart(:,:,(ii-1)*nline+1:ii*nline,:);
    tuTmp = k_bart(:,:,:,(ii-1)*nline+1:ii*nline);  
    wuTmp = w_bart(:,:,(ii-1)*nline+1:ii*nline);
    
    CardiTmp = Card_Sig(indexR((ii-1)*nline+1:ii*nline)) ;
    [~,indexC] = sort(CardiTmp,'descend'); 
    kdatauTmp = kdatauTmp(:,:,indexC,:);
    tuTmp = tuTmp(:,:,:,indexC);
    wuTmp = wuTmp(:,:,indexC);
    for jj = 1:ncardiac
        kdata_cs(:,:,:,:,ii,jj) = kdatauTmp(:,:,(jj-1)*nlineC+1:jj*nlineC,:);
        k_bart_u(:,:,:,:,ii,jj) = tuTmp(:,:,:,(jj-1)*nlineC+1:jj*nlineC);  
        w_bart_u(:,:,:,ii,jj) = wuTmp(:,:,(jj-1)*nlineC+1:jj*nlineC);
    end
end
kdata_cs = reshape(kdata_cs,nx,spk*nlineC,nc,nframe,ncardiac);
k_bart = reshape(k_bart_u,3,nx,spk*nlineC,nframe,ncardiac);
w_bart = reshape(w_bart_u,nx,spk*nlineC,nframe,ncardiac);
kdata_cs = permute(kdata_cs,[7,1,2,3,6,4,5]);   
k_bart = permute(k_bart,[1,2,3,7,6,4,5]);       
w_bart = permute(w_bart,[7,1,2,6,5,3,4]);      
w_bart = w_bart/max(abs(w_bart(:)));          
w_cs2 = sqrt(w_bart);
kdata_cs = squeeze(kdata_cs.*w_cs2);
k_cs = squeeze(k_bart/max(k_bart(:))/2);
w_cs = squeeze(w_bart);
SpkIdx = 1:round(size(k_cs,3));
k_cs = k_cs(:,:,SpkIdx,:,:);
w_cs = w_cs(:,SpkIdx,:,:);
kdata_cs = kdata_cs(:,SpkIdx,:,:,:);
FOVRatio = img_size/max(img_size(:));
k_cs(1,:,:,:,:) = k_cs(1,:,:,:,:)/FOVRatio(1);
k_cs(2,:,:,:,:) = k_cs(2,:,:,:,:)/FOVRatio(2);
k_cs(3,:,:,:,:) = k_cs(3,:,:,:,:)/FOVRatio(3);
%% Prepare for gpuNUFFT operator
param.E = GeneDygpu_NUFFTOperator4D(single(k_cs),single(w_cs),single(sens1),8);
param.y = single(kdata_cs);
FT = param.E;
[RO1,LPE,LCH_PCA,tt,cc] = size(kdata_cs);
kdata_cs = reshape(kdata_cs,[RO1*LPE,LCH_PCA,tt,cc]);
recon_nufft = single(zeros([img_size,tt,cc]));
for i = 1:tt
    for j = 1:cc
        kdata = kdata_cs(:,:,i,j);
        recon_nufft(:,:,:,i,j) = FT{i}{j}'*(kdata);
    end
end
%% Low-rank subspace model estimation
[nx,~,nc] = size(kdata_SI);
spk = seqParam.spk;
seg = seqParam.seg;
kdata_SI = reshape(kdata_SI,nx,seg,nc);
nline = floor(seg/nframe);
nlineC = floor(floor(seg/nframe)/ncardiac);
kdata_SIu = zeros(nx,nlineC,nc,nframe,ncardiac);
[~,indexR] = sort(Res_Signal,'descend');
kdata_SI=kdata_SI(:,indexR,:);
for ii = 1 : nframe   
    kdatauTmp = kdata_SI(:,(ii-1)*nline+1:ii*nline,:);
    CardiTmp = Card_Sig(indexR((ii-1)*nline+1:ii*nline)) ;
    [~,indexC] = sort(CardiTmp,'descend');
    kdatauTmp = kdatauTmp(:,indexC,:);
    for jj = 1:ncardiac
        kdata_SIu(:,:,:,ii,jj) = kdatauTmp(:,(jj-1)*nlineC+1:jj*nlineC,:);
    end
end
kdata_SIu = reshape(kdata_SIu,nx,nlineC,nc,nframe,ncardiac);
kdata_SIu = kdata_SIu(1:2:end,:,:,:,:);
[nx,ntviews,nc,nframe,ncardi] = size(kdata_SIu);
ZIP =abs(ifftshift(ifft(ifftshift(kdata_SIu,1),nx,1),1));
absZIP = abs(ZIP);
for cc = 1:ncardi
    for rr = 1:nframe
        for ii=1:nc
            for jj=1:ntviews
                maxprof=max(absZIP(:,jj,ii,rr,cc));
                minprof=min(absZIP(:,jj,ii,rr,cc));
                ZIP(:,jj,ii,rr,cc)=(ZIP(:,jj,ii,rr,cc)-minprof)./(maxprof-minprof);
            end
        end
    end
end
ZIP = squeeze(mean(ZIP,2));
m = reshape(ZIP,nx*nc,nframe*ncardi);
[U,S,V] = svd(double(m),'econ');
Uk = U(:,1:prin)*S(1:prin,1:prin);
Vk = V(:,1:prin);Vk = Vk';                           %temporal basis function
%% XD-GRASP based 4D Reconstruction
lambdaS  = 0.02;
lambdaTR = 0.02;
lambdaTC = 0;
x = recon_nufft;
scale = 1;
param.F = @(z)ForNUFFT_GPU5D(param,z)*scale;
param.Ft = @(z)AdjNUFFT_GPU5D(param,z,[size(x,1),size(x,2),size(x,3)])*scale;
scale = sqrt(1/(eps+abs(mean(vec(param.Ft(param.F(ones(size(x)))))))));
param.F = @(z)ForNUFFT_GPU5D(param,z)*scale;
param.Ft = @(z)AdjNUFFT_GPU5D(param,z,[size(x,1),size(x,2),size(x,3)])*scale;
clear x
recon1_CS = param.Ft(param.y);
img_size = size(recon1_CS);
nframe = size(recon1_CS,4);
ncardiac = size(recon1_CS,5);
m = reshape(recon1_CS,img_size(1)*img_size(2)*img_size(3),nframe*ncardiac);
[U,S,V] = svd(double(m),'econ');
UL = U(:,1:prin)*S(1:prin,1:prin);% coefficient
VL = V(:,1:prin);VL = VL';%temporal basis function
param.TV_TempWeightTR = max(abs(recon1_CS(:)))*lambdaTR;
param.TV_TempWeightS = max(abs(recon1_CS(:)))*lambdaS;
param.TV_TempTR = TV_TempR();
param.TV_TempWeightTC = max(abs(recon1_CS(:)))*lambdaTC;
param.TV_TempTC = TV_TempC();
param.TV_TempS  = TV_TempS();
param.nite = 30;
param.display = 1;
tic
Uk = UL;
Uk_gpu = gpuArray(single(Uk));
Vk_gpu = gpuArray(single(Vk));
for n = 1 : 3
    n
    Uk_gpu = CSL1NlCg_LowRank4D(recon1_CS,Uk_gpu,Vk_gpu,param);
    recon1_CS = gather(Uk_gpu*Vk_gpu);
    recon1_CS = reshape(recon1_CS,img_size(1),img_size(2),img_size(3),nframe,ncardiac);
end
toc
end