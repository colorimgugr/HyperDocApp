
%register all images in VNIR cube using the transform estimated from bands
%60 of VNIR and 10 of SWIR
name='pleito_39c';
%Load cube VNIR
load(strcat(name,'.mat'));
%%
im_VNIR=cube.DataCube(:,:,60);
wl_VNIR=wl;
cube_VNIR=cube;
clear cube;
%%
%Load cube SWIR
load(strcat(name,' (1).mat'));
%%
im_SWIR=cube.DataCube(:,:,10);
clear cube;
%%
%Register using App and save results in variable RegResults

fixedRefObj = imref2d(size(im_SWIR));
movingRefObj = imref2d(size(im_VNIR));
tform= RegResults.Transformation;
for k=1:121
    cubereg(:,:,k) = imwarp(cube_VNIR.DataCube(:,:,k), movingRefObj, tform, 'OutputView', fixedRefObj, 'SmoothEdges', true);
end

cube=hypercube(cubereg,wl_VNIR,cube_VNIR.Metadata);
im_reg=cube.DataCube(:,:,60);
figure, imshowpair(im_SWIR,im_reg,'falsecolor');
namesave=strcat(name,'_reg.mat');
save(namesave,'cube','fixedRefObj','movingRefObj','tform');
