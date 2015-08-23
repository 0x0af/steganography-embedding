function [ ] = bhatnagar_extract( watermarked_image_name, grayscale_container_name, grayscale_watermark_name, extracted_coarsest_watermark_name, extracted_finest_watermark_name, watermark_size, alpha)
%BHATNAGAR_EXTRACT

clc;

tic;

configure;

grayscale_container = [BMP_DIR_PATH,grayscale_container_name];
grayscale_watermark = [BMP_DIR_PATH,grayscale_watermark_name];
watermarked_image = [BMP_DIR_PATH,watermarked_image_name];
extracted_coarsest_watermark = [BMP_DIR_PATH,extracted_coarsest_watermark_name];
extracted_finest_watermark = [BMP_DIR_PATH,extracted_finest_watermark_name];
watermark_size = int2str(watermark_size);
alpha = num2str(alpha);

cmd = [PYTHON_PATH,' ',SCRIPT_PATH,'bhatnagar_embedding_method.py --mode extract --grayscale_container',' ',grayscale_container,' ','--grayscale_watermark',' ',grayscale_watermark,' ','--watermarked_image',' ',watermarked_image,' ','--alpha',' ',alpha,' ','--watermark_size',' ',watermark_size,' ','--extracted_coarsest_watermark',' ',extracted_coarsest_watermark,' ','--extracted_finest_watermark',' ',extracted_finest_watermark];

system(cmd);

clear all;

disp('Operation finished');

toc;

end