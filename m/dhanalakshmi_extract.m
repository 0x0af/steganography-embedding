function [ ] = dhanalakshmi_extract( watermarked_image_name, grayscale_container_name, grayscale_primary_watermark_name, grayscale_secondary_watermark_name,extracted_primary_watermark_name, extracted_secondary_watermark_name, alpha1, alpha2)
%DHANALAKSHMI_EXTRACT

clc;

tic;

configure;

grayscale_container = [BMP_DIR_PATH,grayscale_container_name];
grayscale_primary_watermark = [BMP_DIR_PATH,grayscale_primary_watermark_name];
grayscale_secondary_watermark = [BMP_DIR_PATH,grayscale_secondary_watermark_name];
watermarked_image = [BMP_DIR_PATH,watermarked_image_name];
alpha1 = num2str(alpha1);
alpha2 = num2str(alpha2);
extracted_primary_watermark = [BMP_DIR_PATH,extracted_primary_watermark_name];
extracted_secondary_watermark = [BMP_DIR_PATH,extracted_secondary_watermark_name];

cmd = [PYTHON_PATH,' ',SCRIPT_PATH,'dhanalakshmi_embedding_method.py --mode extract --grayscale_container',' ',grayscale_container,' ','--grayscale_primary_watermark',' ',grayscale_primary_watermark,' ','--grayscale_secondary_watermark',' ',grayscale_secondary_watermark,' ','--watermarked_image',' ',watermarked_image,' ','--alpha1',' ',alpha1,' ','--alpha2',' ',alpha2,' ','--extracted_primary_watermark',' ',extracted_primary_watermark,' ','--extracted_secondary_watermark',' ',extracted_secondary_watermark];

system(cmd);

clear all;

disp('Operation finished');

toc;

end