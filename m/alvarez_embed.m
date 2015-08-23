function [ ] = alvarez_embed( grayscale_container_name, binary_watermark_name, watermarked_image_name)
%ALVAREZ_EMBED

clc;

tic;

configure;

grayscale_container = [BMP_DIR_PATH,grayscale_container_name];
binary_watermark = [BMP_DIR_PATH,binary_watermark_name];
watermarked_image = [BMP_DIR_PATH,watermarked_image_name];

cmd = [PYTHON_PATH,' ',SCRIPT_PATH,'alvarez_embedding_method.py --mode embed --grayscale_container',' ',grayscale_container,' ','--binary_watermark',' ',binary_watermark,' ','--watermarked_image',' ',watermarked_image];

system(cmd);

clear all;

disp('Operation finished');

toc;

end