function im_h = MSDCNN_Matconvnet(im_input,weight,bias)
im_input = single(im_input);
    convfea_shallow = vl_nnrelu(vl_nnconv(im_input,weight{1},bias{1},'Pad',(size(weight{1},1)-1)/2));
    convfea_shallow = vl_nnrelu(vl_nnconv(convfea_shallow,weight{2},bias{2},'Pad',(size(weight{2},1)-1)/2));
    convfea_shallow = vl_nnconv(convfea_shallow,weight{3},bias{3},'Pad',(size(weight{3},1)-1)/2);
    
    convfea_deep_stage1_in = vl_nnrelu(vl_nnconv(im_input,weight{4},bias{4},'Pad',(size(weight{4},1)-1)/2));
    
    convfea_deep_stage1(:,:,1:20) = vl_nnrelu(vl_nnconv(convfea_deep_stage1_in,weight{5},bias{5},'Pad',(size(weight{5},1)-1)/2));
    convfea_deep_stage1(:,:,21:40) = vl_nnrelu(vl_nnconv(convfea_deep_stage1_in,weight{6},bias{6},'Pad',(size(weight{6},1)-1)/2));
    convfea_deep_stage1(:,:,41:60) = vl_nnrelu(vl_nnconv(convfea_deep_stage1_in,weight{7},bias{7},'Pad',(size(weight{7},1)-1)/2));
    convfea_deep_stage1 = convfea_deep_stage1_in + convfea_deep_stage1;
    
    convfea_deep_stage2_in = vl_nnrelu(vl_nnconv(convfea_deep_stage1,weight{8},bias{8},'Pad',(size(weight{8},1)-1)/2));
    
    convfea_deep_stage2(:,:,1:10) = vl_nnrelu(vl_nnconv(convfea_deep_stage2_in,weight{9},bias{9},'Pad',(size(weight{9},1)-1)/2));
    convfea_deep_stage2(:,:,11:20) = vl_nnrelu(vl_nnconv(convfea_deep_stage2_in,weight{10},bias{10},'Pad',(size(weight{10},1)-1)/2));
    convfea_deep_stage2(:,:,21:30) = vl_nnrelu(vl_nnconv(convfea_deep_stage2_in,weight{11},bias{11},'Pad',(size(weight{11},1)-1)/2));
    convfea_deep_stage2 = convfea_deep_stage2_in + convfea_deep_stage2; 
    convfea_deep = vl_nnconv(convfea_deep_stage2,weight{12},bias{12},'Pad',(size(weight{12},1)-1)/2);
    
    im_h = convfea_deep+convfea_shallow;

end

