% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Jan 30

%% 3(1)
flower = read_img_rgb('images/Flower.raw', 512, 768);
frost_flower_5 = rgb_wrapper(@frost_transform, flower, 2);
figure
imshow(frost_flower_5)
frost_flower_7 = rgb_wrapper(@frost_transform, flower, 3);
figure
imshow(frost_flower_7)

%% 3(2)
flower_noisy = read_img_rgb('images/Flower_noisy.raw', 512, 768);
figure
imshow(flower_noisy)
frost_flower_noisy_5 = rgb_wrapper(@frost_transform, flower_noisy, 2);
figure
imshow(frost_flower_noisy_5)
frost_flower_noisy_7 = rgb_wrapper(@frost_transform, flower_noisy, 3);
figure
imshow(frost_flower_noisy_7)

%% 3(3)
% a
flower_gray = read_img_gray('images/Flower_gray.raw', 512, 768);
figure
imshow(flower_gray)
flower_gray_noisy = read_img_gray('images/Flower_gray_noisy.raw', 512, 768);
figure
imshow(flower_gray_noisy)
psnr_noisy = PSNR(flower_gray, flower_gray_noisy);
bilateral_flower_pre = bilateral_filter_transform(flower_gray_noisy, 1, 1.2, 40);
figure
imshow(bilateral_flower_pre)
psnr_bilateral_pre = PSNR(flower_gray, bilateral_flower_pre);
frost_flower_pre = frost_transform(bilateral_flower_pre, 3);
figure
imshow(frost_flower_pre)
psnr_frost_pre = PSNR(flower_gray, frost_flower_pre);
% b
frost_flower_post = frost_transform(flower_gray_noisy, 3);
figure
imshow(frost_flower_post)
psnr_frost_post = PSNR(flower_gray, frost_flower_post);
bilateral_flower_post = bilateral_filter_transform(frost_flower_post, 1, 1.2, 40);
figure
imshow(bilateral_flower_post)
psnr_bilateral_post = PSNR(flower_gray, bilateral_flower_post);

%% 3(1) helper
function img = read_img_rgb(file, row, col)
    fr = fopen(file,'rb');
    if (fr == -1)
        error('Can not open output image file. Press CTRL-C to exit \n');
    end

    img = zeros(row, col, 3);
    img = uint8(img);
    temp=fread(fr, 'uint8=>uint8');
    
    for i=0:(row-1)
        for j=0:(col-1)
            img(i+1,j+1,1) = temp(i*col*3+j*3+1);
            img(i+1,j+1,2) = temp(i*col*3+j*3+2);
            img(i+1,j+1,3) = temp(i*col*3+j*3+3);
        end
    end

    fclose(fr);
end

function padded_img = zero_pad(img_matrix, pad_width)
    [nrows, ncols] = size(img_matrix);
    padded_img = zeros(nrows+pad_width*2, ncols+pad_width*2);
    
    padded_img((pad_width+1):(pad_width+nrows),(pad_width+1):(pad_width+ncols))...
        = img_matrix(:,:);
end

function frost_filter = frost_filter_of_size(sz)
    frost_filter = zeros((2*sz+1));
    frost_filter(randi(2*sz+1),randi(2*sz+1)) = 1;
end

function sum = convolute(ori_matrix, filter)
    [nrows, ncols] = size(ori_matrix);
    sum = 0;
    for i=1:nrows
        for j=1:ncols
            sum = sum + ori_matrix(i,j) * filter(i,j);
        end
    end
    sum = uint8(sum);
end

function new_img = frost_transform(img, sz)
    padded_img = zero_pad(img, sz);
    [nrows, ncols] = size(img);
    new_img = zeros(nrows, ncols);
    new_img = uint8(new_img);
    
    for i=1:nrows
        for j=1:ncols
            frost_filter = frost_filter_of_size(sz);
            orig_matrix = padded_img(i:(i+2*sz),j:(j+2*sz));
            new_img(i,j) = convolute(orig_matrix, frost_filter);
        end
    end
end

function result = rgb_wrapper(func, img, varargin)
    nrows = size(img, 1);
    ncols = size(img, 2);
    result = zeros(nrows, ncols, 3);
    result = uint8(result);
    
    result(:,:,1) = func(img(:,:,1), varargin{:});
    result(:,:,2) = func(img(:,:,2), varargin{:});
    result(:,:,3) = func(img(:,:,3), varargin{:});
end

%% 3(3) helper
function img = read_img_gray(file, row, col)
    fr = fopen(file,'rb');
    if (fr == -1)
        error('Can not open output image file. Press CTRL-C to exit \n');
    end

    temp=fread(fr, 'uint8=>uint8');
    img=reshape(temp, [col row]);
    img=img';

    fclose(fr);
end

function sum = bilateral_convolute(matrix, sigma_c, sigma_s)
    [nrows, ncols] = size(matrix);
    center_i = floor(nrows/2) + 1;
    center_j = floor(ncols/2) + 1;
    center_v = matrix(center_i, center_j);

    wsum = 0;
    vsum = 0;
    for i=1:nrows
        for j=1:ncols
            distance = (i - center_i)^2 + (j - center_j)^2;
            vdifference = (matrix(i,j) - center_v)^2;
            weight = exp(-distance/(2*sigma_c^2) - ...
                            vdifference/(2*sigma_s^2));
            wsum = wsum + weight;
            vsum = vsum + weight * matrix(i,j);
        end
    end
    
    sum = vsum / wsum;
end

function new_img = bilateral_filter_transform(img, sz, sigma_c, sigma_s)
    padded_img = zero_pad(img, sz);
    [nrows, ncols] = size(img);
    new_img = zeros(nrows, ncols);
    new_img = uint8(new_img);
    
    for i=1:nrows
        for j=1:ncols
            orig_matrix = padded_img(i:(i+2*sz),j:(j+2*sz));
            new_img(i,j) = bilateral_convolute(orig_matrix, sigma_c, sigma_s);
        end
    end
end

function mse = MSE(img1, img2)
    [nrows, ncols] = size(img1);
    npx = nrows * ncols;
    
    sum = 0;
    for i=1:nrows
        for j=1:ncols
           sum = sum + (double(img1(i,j))-double(img2(i,j)))^2;
        end
    end
    
    mse = sum / npx;
end

function psnr = PSNR(img1, img2)
    mse = MSE(img1, img2);
    
    % 255^2 = 65025
    psnr = 10 * log10(65025/double(mse));
end