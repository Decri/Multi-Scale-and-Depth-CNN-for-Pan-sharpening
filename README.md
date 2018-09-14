# Multi-Scale-and-Depth-CNN-for-Pan-sharpening

（1）This matlab code includes the implementation of two deep convolutional networks for fusion of MS and Pan images：
MSDCNN: Q. Yuan, Y. Wei, X. Meng, et al, A Multiscale and Multidepth Convolutional Neural Network for Remote Sensing Imagery Pan-Sharpening[J]. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2018, 11(3):978-989.

DRPNN: Wei Y, Yuan Q, Shen H, et al. Boosting the Accuracy of Multispectral Image Pansharpening by Learning a Deep Residual Network[J]. IEEE Geoscience & Remote Sensing Letters, 2017, 14(10):1795-1799.

（2）Before running the code you need to at least compile MatConvNet on your device. Caffe and MatCaffe are also recommended if you want to train your own model.
*The currently given model only supports fusion of MS image with 4 bands and PAN image with 1 band, such as QuickBird, IKONOS and Pleiades. 

（3）If you simply want to use the models trained by the author to fuse your data:
a. Set your matlab path to the './Pansharpening' as where the code was put.
b. Put your MS and Pan images in ./Pansharpening/testdata and use their filenames to replace 'QB_MS' and 'QB_PAN'in the file 'Demo_Pansharpening_4bands'.
c. Select your mode in the 'Demo_Pansharpening_4bands' as described and run the demo.

（4）If you want to train your own model：
a. Put your MS and PAN training imageS in ./Pansharpening/Train_MSFusion as the given files.
b. Run the 'generate_train_QBfusion'. Note that the savepath of the .h5 file should be a full path.
c. Set the full path of 'train_QBfusion.h5' in the 'train_QBfusion.txt'.
d. Set the full path of 'train_QBfusion.txt' in the 'MSDCNN_net.prototxt'. The arg 'snapshot_prefix' should also be set as a full path.
e. Locate your current path in CMD, then run the command 'caffe train -solver solver.prototxt'.
f. Get the trained model in the ./snapshots.
