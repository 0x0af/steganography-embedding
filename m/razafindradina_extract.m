function [ ] = razafindradina_extract( watermarked_image_name, grayscale_container_name, same_size_grayscale_watermark_name, extracted_watermark_name, alpha)
%RAZAFINDRADINA_EXTRACT

clc;

tic;

configure;

grayscale_container = [BMP_DIR_PATH,grayscale_container_name];
same_size_grayscale_watermark = [BMP_DIR_PATH,same_size_grayscale_watermark_name];
watermarked_image = [BMP_DIR_PATH,watermarked_image_name];
extracted_watermark = [BMP_DIR_PATH,extracted_watermark_name];
alpha = num2str(alpha);

cmd = [PYTHON_PATH,' ',SCRIPT_PATH,'razafindradina_embedding_method.py --mode extract --grayscale_container',' ',grayscale_container,' ','--same_size_grayscale_watermark',' ',same_size_grayscale_watermark,' ','--watermarked_image',' ',watermarked_image,' ','--alpha',' ',alpha,' ','--extracted_watermark',' ', extracted_watermark];

system(cmd);

clear all;

disp('Operation finished');

toc;

end