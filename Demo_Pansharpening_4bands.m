%% Before running this demo, you need to compile MatConvNet on your device 

%% to make sure you have the MEX functions "vl_nnconv" and "vl_nnrelu" that 

%% are needed when running 'MSDCNN_Matconvnet.m'.

%% You may also compile Caffe and MatCaffe on your device

%% to make sure you have the MEX function "caffe.Net" that is needed to load the trained caffe models.


clear;close all;

run vl_setupnn; 

scale = 4;


%% Loading test images and pre-processing them for fusion:

im_gt = imread('./testdata/QB-MS.tif');
im_gt = RSgenerate(im2double(im_gt),0,0);

im_gt = modcrop(im_gt,scale);

[hei,wid,channels] = size(im_gt);

im_pan = imread('./testdata/QB-PAN.tif');
im_pan = RSgenerate(im2double(im_pan),0,0);

im_pan = im_pan(1:hei*4,1:wid*4);

im_input = imresize(imresize(im_gt,1/scale,'bicubic'),scale,'bicubic');

im_input(:,:,channels+1) = imresize(im_pan,1/scale,'bicubic');


%% If you have trained a new model with your own training data,  
%% Loading with MatCaffe and note that the image sizes must be respectively set in the 'MSDPNN_mat.prototxt':

net1 = caffe.Net('MSDCNN_mat.prototxt','My_MSDCNNfor4bands_iter_242100.caffemodel','test');
tic;
im_fused = net1.forward({single(im_input)});
im_fused = double(im_fused{1});
toc;
%% Without MatCaffe:

save_conv_and_deconv_filters('My_MSDCNNfor4bands',242100,'MSDCNN_mat.prototxt','My_MSDCNNfor4bands.mat');

load('My_MSDCNNfor4bands.mat');
tic;

im_fused = double(MSDCNN_Matconvnet(im_input,weight,bias));

toc;



%% You may also use the currently provided models, by simply replace the 'My_MSDCNNfor4bands' above
%% with 'MSDCNNfor4bands'£º
%% With MatCaffe:
net1 = caffe.Net('MSDCNN_mat.prototxt','MSDCNNfor4bands_iter_242100.caffemodel','test');
tic;
im_fused = net1.forward({single(im_input)});
im_fused = double(im_fused{1});
toc;
%% Without MatCaffe:
save_conv_and_deconv_filters('MSDCNNfor4bands',242100,'MSDCNN_mat.prototxt','MSDCNNfor4bands.mat');

load('MSDCNNfor4bands.mat');
tic;

im_fused = double(MSDCNN_Matconvnet(im_input,weight,bias));

toc;





%% Computing quality indices:

im_fused = min(max(im_fused,0),1);

q2n_MSDCNN = q2n(im_gt,im_fused,16,16);

ergas_MSDCNN = ERGAS(im_gt,im_fused,4);

sam_MSDCNN = SAM(im_gt,im_fused);


%% Visualizing fused images:

imwrite(uint16(RSgenerate(im_fused(:,:,[3 2 1]),1,1)*65535),'My_MSDNNfor4bands.tif');

imwrite(uint16(RSgenerate(im_gt(:,:,[3 2 1]),1,1)*65535),'Ground truth.tif');

fprintf(['Q4 of MSDCNN fusion:',num2str(q2n_MSDCNN),'\n']); 

fprintf(['ERGAS of MSDCNN fusion:',num2str(ergas_MSDCNN),'\n']);

fprintf(['SAM of MSDCNN fusion:',num2str(sam_MSDCNN),'\n']);