function x = CSL1NlCg_LowRank4D(recon_cs,Uk,Vk,param)
% f(x) = ||E*x - y||^2 + L1Weight * ||W*x||_1TVWeight * ||TV*x||_1 + TV_TempWeight * ||TV_Temp*x||_1 

% starting point
x = Uk;
clear Uk;

% line search parameters
maxlsiter = 10 ;
gradToll = 1e-8 ;
param.l1Smooth = 1e-15;	
alpha = 0.01;  
beta = 0.6;
t0 = 1 ; 
k = 0;
% compute g0  = grad(f(x))
g0 = grad(x,Vk,recon_cs,param);
dx = -g0;

% iterations
while(1)

    % backtracking line-search
	f0 = objective(x,Vk,recon_cs,dx,0,param);
	t = t0;
    f1 = objective(x,Vk,recon_cs,dx,t,param);
	lsiter = 0;
    
    %line search
	while (f1 > f0 - alpha*t*abs(g0(:)'*dx(:))) && (lsiter < maxlsiter)
		lsiter = lsiter + 1;
		t = t * beta;
		f1 = objective(x,Vk,recon_cs,dx,t,param);
	end

% 	if lsiter == maxlsiter
% 		disp('Error - line search ...');
% 		return;
% 	end

	% control the number of line searches by adapting the initial step search
	if lsiter > 2, t0 = t0 * beta; end 
	if lsiter < 1, t0 = t0 / beta; end

    % update x
	x = (x + t*dx);
    
	% print some numbers for debug purposes	
    if param.display
        fprintf('%d   , obj: %f, L-S: %d\n', k,f1,lsiter);
    end
    k = k + 1;
	
    %imshow
    if 0 
        p=round(110*2/3);
        figure, imshow(fliplr(abs(squeeze(gather(x(:,p,:,8))))'),[])
    end
    
	% stopping criteria (to be improved)
	if (k >= param.nite) || (norm(dx(:)) < gradToll), break; end

    
    %conjugate gradient calculation
	g1 = grad(x,Vk,recon_cs,param);
	bk = g1(:)'*g1(:)/(g0(:)'*g0(:)+eps);
	g0 = g1;
	dx =  - g1 + bk*dx;
	

end
return;

function res = objective(x,Vk,recon_cs,dx,t,param) %**********************************
imgsize = size(recon_cs);
nframe = size(recon_cs,4);
ncardiac = size(recon_cs,5);
prin = size(x,2);
m = (x+t*dx)*Vk;
m = reshape(m,imgsize(1),imgsize(2),imgsize(3),nframe,ncardiac);

m_cpu = gather(m);
% L2-norm part
w = gpuArray(single(param.F (m_cpu))) - param.y;
L2Obj = w(:)' * w(:);clear w;

% TV part along time
if param.TV_TempWeightTR
    w = param.TV_TempTR * (m); 
    TV_TempObjTR = sum((w(:).*conj(w(:))+param.l1Smooth).^(1/2));
else
    TV_TempObjTR = 0;
end
clear w;

% TV part along time
if param.TV_TempWeightTC
    w = param.TV_TempTC * (m); 
    TV_TempObjTC = sum((w(:).*conj(w(:))+param.l1Smooth).^(1/2));
else
    TV_TempObjTC = 0;
end  
clear w;

% TV part along spatial
if param.TV_TempWeightS
    w = reshape(x,imgsize(1),imgsize(2),imgsize(3),prin);
    w = param.TV_TempS * (w); 
    TV_TempObjS = sum((w(:).*conj(w(:))+param.l1Smooth).^(1/2));
else
    TV_TempObjS = 0;
end 
clear w;

res = L2Obj + param.TV_TempWeightTR*TV_TempObjTR + param.TV_TempWeightTC*TV_TempObjTC + param.TV_TempWeightS*TV_TempObjS;


function g = grad(x,Vk,recon_cs,param)%***********************************************
imgsize = size(recon_cs);
nframe = size(recon_cs,4);
ncardiac = size(recon_cs,5);
prin = size(x,2);
m = x*Vk;
m = reshape(m,imgsize(1),imgsize(2),imgsize(3),nframe,ncardiac);

m_cpu = gather(m);
% L2-norm part
GradTmp = 2*(gpuArray(single(param.Ft(param.F(m_cpu)-param.y))));
GradTmp = reshape(GradTmp,imgsize(1)*imgsize(2)*imgsize(3),nframe*ncardiac);
GradTmp = GradTmp * (Vk');
g = GradTmp;

% TV part along time
if param.TV_TempWeightTR
    GradTmp = param.TV_TempTR * m;
    GradTmp = param.TV_TempTR'*(GradTmp.*(GradTmp.*conj(GradTmp)+param.l1Smooth).^(-0.5));
    GradTmp = reshape(GradTmp,imgsize(1)*imgsize(2)*imgsize(3),nframe*ncardiac);
    GradTmp = GradTmp * (Vk');
else
    GradTmp = 0;
end
g = g+param.TV_TempWeightTR*GradTmp;

% TV part along time
if param.TV_TempWeightTC    
    GradTmp = param.TV_TempTC * m;
    GradTmp = param.TV_TempTC'*(GradTmp.*(GradTmp.*conj(GradTmp)+param.l1Smooth).^(-0.5));
    GradTmp = reshape(GradTmp,imgsize(1)*imgsize(2)*imgsize(3),nframe*ncardiac);
    GradTmp = GradTmp * (Vk');
else
    GradTmp = 0;
end
g = g+param.TV_TempWeightTC*GradTmp;

% TV part along spatial
if param.TV_TempWeightS
    GradTmp = reshape(x,imgsize(1),imgsize(2),imgsize(3),prin);
    GradTmp = param.TV_TempS * GradTmp;
    GradTmp = param.TV_TempS'*(GradTmp.*(GradTmp.*conj(GradTmp)+param.l1Smooth).^(-0.5));
    GradTmp = reshape(GradTmp,imgsize(1)*imgsize(2)*imgsize(3),prin);
%     TV_TempGradS = TV_TempGradS * (Vk');
else
    GradTmp = 0;
end
g = g+param.TV_TempWeightS*GradTmp;

% g = GradTmp + param.TV_TempWeightTR*GradTmp + param.TV_TempWeightTC*GradTmp + param.TV_TempWeightS*GradTmp;