function [ ] = alvarez_extract( grayscale_container_name, watermarked_image_name, extracted_binary_watermark_name, watermark_size)
%ALVAREZ_EXTRACT

clc;

tic;

configure;

grayscale_container = [BMP_DIR_PATH,grayscale_container_name];
watermarked_image = [BMP_DIR_PATH,watermarked_image_name];
extracted_binary_watermark = [BMP_DIR_PATH,extracted_binary_watermark_name];
watermark_size = int2str(watermark_size);

cmd = [PYTHON_PATH,' ',SCRIPT_PATH,'alvarez_embedding_method.py --mode extract --grayscale_container',' ',grayscale_container,' ','--extracted_binary_watermark',' ', extracted_binary_watermark,' ','--watermarked_image',' ',watermarked_image,' ','--watermark_size',' ',watermark_size];

system(cmd);

clear all;

disp('Operation finished');

toc;

end