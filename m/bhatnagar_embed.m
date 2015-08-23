function [ ] = bhatnagar_embed( grayscale_container_name, grayscale_watermark_name, watermarked_image_name, alpha)
%BHATNAGAR_EMBED

clc;

tic;

configure;

grayscale_container = [BMP_DIR_PATH,grayscale_container_name];
grayscale_watermark = [BMP_DIR_PATH,grayscale_watermark_name];
watermarked_image = [BMP_DIR_PATH,watermarked_image_name];
alpha = num2str(alpha);

cmd = [PYTHON_PATH,' ',SCRIPT_PATH,'bhatnagar_embedding_method.py --mode embed --grayscale_container',' ',grayscale_container,' ','--grayscale_watermark',' ',grayscale_watermark,' ','--watermarked_image',' ',watermarked_image,' ','--alpha',' ',alpha];

system(cmd);

clear all;

disp('Operation finished');

toc;

end