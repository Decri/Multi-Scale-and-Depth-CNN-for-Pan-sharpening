function im_fused = DRPNN_Matconvnet(im_input,weight,bias)
im_input = single(im_input);
layer_num = size(weight,2);
    convfea = vl_nnconv(im_input,weight{1},bias{1},'Pad',(size(weight{1},1)-1)/2);
    for i = 2:layer_num-1
        convfea = vl_nnrelu(convfea);
        convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',(size(weight{i},1)-1)/2);
    end
    im_fused = vl_nnconv(vl_nnrelu(convfea+im_input),weight{layer_num},bias{layer_num},'Pad',(size(weight{layer_num},1)-1)/2);
end