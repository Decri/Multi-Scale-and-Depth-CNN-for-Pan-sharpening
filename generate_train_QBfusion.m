%% settings
folder1 = 'Train_MSFusion\Quickbird\MUL';folder2 = 'Train_MSFusion\Quickbird\PAN';
savepath = 'train_QBfusion.h5';
size_input = 31;
size_label = 31;
scale = 4;
stride = 14;

%% initialization
data = zeros(size_input, size_input, 5, 1);
label = zeros(size_label, size_label, 4, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
 filepath1 = dir(fullfile(folder1,'*.tif'));filepath2 = dir(fullfile(folder2,'*.tif'));   
 for i = 1 : length(filepath1)    
    im_mul = imread(fullfile(folder1,filepath1(i).name));im_pan = imread(fullfile(folder2,filepath2(i).name));
    im_mul = modcrop(im_mul,40);[hei,wid,channels] = size(im_mul);im_pan = im_pan(1:hei*4,1:wid*4);
    hei = hei/10;wid = wid/10;
    for subx = 1:10
        for suby = 1:10
            subim_mul = im_mul((subx-1)*hei+1:subx*hei,(suby-1)*wid+1:suby*wid,:);
            subim_pan = im_pan((subx-1)*hei*4+1:subx*hei*4,(suby-1)*wid*4+1:suby*wid*4);
    subim_mul = RSgenerate(im2double(subim_mul),0,0);
    im_label = subim_mul;
    subim_pan = RSgenerate(im2double(subim_pan),0,0);
    im_input = im_label;
    subim_pan = imresize(subim_pan,[hei,wid],'bicubic');
    for channel = 1:channels
    im_input(:,:,channel) = imresize(imresize(im_label(:,:,channel),1/scale,'bicubic'),[hei,wid],'bicubic');
    end
    im_input(:,:,channels+1) = subim_pan;
    %% extract sub images
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1,:);
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,:);
            count=count+1;
            data(:, :, :, count) = subim_input;
            label(:, :, :, count) = subim_label;
        end
    end
        end
    end
end

order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);