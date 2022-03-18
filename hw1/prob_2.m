% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Jan 30

%% 2(a) analysis
flower = read_img_gray('images/Flower_gray.raw', 512, 768);
figure
imshow(flower)
flower_noisy = read_img_gray('images/Flower_gray_noisy.raw', 512, 768);
figure
imshow(flower_noisy)
psnr_noisy = PSNR(flower, flower_noisy);
noise = double(flower_noisy) - double(flower);
figure
histogram(noise(:,:))

%% 2(a)
mean_filter_3by3 = mean_filter_of_size(1);
mean_3by3_flower = filter_transform(flower_noisy, mean_filter_3by3);
figure
imshow(mean_3by3_flower)
psnr_mean_3by3 = PSNR(flower, mean_3by3_flower);
mean_filter_5by5 = mean_filter_of_size(2);
mean_5by5_flower = filter_transform(flower_noisy, mean_filter_5by5);
figure
imshow(mean_5by5_flower)
psnr_mean_5by5 = PSNR(flower, mean_5by5_flower);
gaussian_filter_3by3 = gaussian_filter_of_size(1, 1.5);
gaussian_3by3_flower = filter_transform(flower_noisy, gaussian_filter_3by3);
figure
imshow(gaussian_3by3_flower)
psnr_gaussian_3by3 = PSNR(flower, gaussian_3by3_flower);
gaussian_filter_5by5 = gaussian_filter_of_size(2, 1.5);
gaussian_5by5_flower = filter_transform(flower_noisy, gaussian_filter_5by5);
figure
imshow(gaussian_5by5_flower)
psnr_gaussian_5by5 = PSNR(flower, gaussian_5by5_flower);

%% 2(a) exploration - mean filter parameters
% 1
% 2 for rgb
% mean_filter_results = test_mean_filter_parameters(1:5, flower, flower_noisy);

%% 2(a) exploration - gaussian filter parameters
% 1 0.75
% 2 1.25 for rgb
% gaussian_filter_results = test_gaussian_filter_parameters(1:5, 0.5:0.25:1.5, ...
%                             flower, flower_noisy);

%% 2(b)
bilateral_flower = bilateral_filter_transform(flower_noisy, 1, 1.2, 40);
figure
imshow(bilateral_flower)
psnr_bilateral = PSNR(flower, bilateral_flower);

%% 2(b) exploration - bilateral filter parameters
% 1 1.2 40
% 2 1.5 140 for rgb
% bilateral_filter_results = test_bilateral_filter_parameters(1, 0.8:0.1:1.2, ...
%                             30:10:70, flower, flower_noisy);

%% 2(c)
% 21 3
nlm_flower_try1 = nlm_filter_transform(flower_noisy);
figure
imshow(nlm_flower_try1)
psnr_nlm_try1 = PSNR(flower, nlm_flower_try1);
nlm_flower_try2 = nlm_filter_transform(flower_noisy, 31);
figure
imshow(nlm_flower_try2)
psnr_nlm_try2 = PSNR(flower, nlm_flower_try2);
nlm_flower_try3 = nlm_filter_transform(flower_noisy, 21, 3);
figure
imshow(nlm_flower_try3)
psnr_nlm_try3 = PSNR(flower, nlm_flower_try3);
nlm_flower_try4 = nlm_filter_transform(flower_noisy, 21, 5, 10);
figure
imshow(nlm_flower_try4)
psnr_nlm_try4 = PSNR(flower, nlm_flower_try4);

%% 2(d)
% analysis
flower_rgb = read_img_rgb('images/Flower.raw', 512, 768);
figure
imshow(flower_rgb)
flower_rgb_noisy = read_img_rgb('images/Flower_noisy.raw', 512, 768);
figure
imshow(flower_rgb_noisy)
psnr_rgb_noisy = PSNR_rgb(flower_rgb, flower_rgb_noisy);
noise_rgb = double(flower_rgb_noisy) - double(flower_rgb);
figure
histogram(noise_rgb(:,:,1))
% median filter
median_flower_rgb = rgb_wrapper(@median_filter_transform, flower_rgb_noisy, 1, true);
figure
imshow(median_flower_rgb)
psnr_median_rgb = PSNR_rgb(flower_rgb, median_flower_rgb);
% mean filter
mean_flower_rgb = rgb_wrapper(@filter_transform, median_flower_rgb, mean_filter_3by3);
figure
imshow(mean_flower_rgb)
psnr_mean_rgb = PSNR_rgb(flower_rgb, mean_flower_rgb);
% gaussian filter
% gaussian_flower_rgb = rgb_wrapper(@filter_transform, median_flower_rgb, gaussian_filter_5by5);
% figure
% imshow(gaussian_flower_rgb)
% psnr_gaussian_rgb = PSNR_rgb(flower_rgb, gaussian_flower_rgb);
% bilateral filter
% bilateral_flower_rgb = rgb_wrapper(@bilateral_filter_transform, median_flower_rgb, 2, 1.5, 140);
% figure
% imshow(bilateral_flower_rgb)
% psnr_bilateral_rgb = PSNR_rgb(flower_rgb, bilateral_flower_rgb);
% nlm filter
nlm_flower_rgb = rgb_wrapper(@nlm_filter_transform, median_flower_rgb);
figure
imshow(nlm_flower_rgb)
psnr_nlm_rgb = PSNR_rgb(flower_rgb, nlm_flower_rgb);

%% 2(a) helper
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

function padded_img = zero_pad(img_matrix, pad_width)
    [nrows, ncols] = size(img_matrix);
    padded_img = zeros(nrows+pad_width*2, ncols+pad_width*2);
    
    padded_img((pad_width+1):(pad_width+nrows),(pad_width+1):(pad_width+ncols))...
        = img_matrix(:,:);
end

% size n means filter of size (2n+1)-by-(2n+1)
function mean_filter = mean_filter_of_size(sz)
    % not using matrix operators as required
    mean_filter = ones(2*sz+1);
    normalization_factor = (2*sz+1)^2;
    for i=1:(2*sz+1)
        for j=1:(2*sz+1)
            mean_filter(i,j) = mean_filter(i,j) / normalization_factor;
        end
    end
end

function gaussian_filter = gaussian_filter_of_size(sz, sigma)
    gaussian_filter = zeros(2*sz+1);
    
    sum = 0;
    for i=1:(2*sz+1)
        for j=1:(2*sz+1)
            xy_sq = (i-1-sz)^2 + (j-1-sz)^2;
            raw_weight = exp(-(xy_sq/(2*sigma^2))) / (2*pi*sigma^2);
            gaussian_filter(i,j) = raw_weight;
            sum = sum + raw_weight;
        end
    end
    
    for i=1:(2*sz+1)
        for j=1:(2*sz+1)
            gaussian_filter(i,j) = gaussian_filter(i,j) / sum;
        end
    end
end

function new_img = filter_transform(img, filter)
    sz = (size(filter,1)-1) / 2;
    
    padded_img = zero_pad(img, sz);
    [nrows, ncols] = size(img);
    new_img = zeros(nrows, ncols);
    new_img = uint8(new_img);
    
    for i=1:nrows
        for j=1:ncols
            orig_matrix = padded_img(i:(i+2*sz),j:(j+2*sz));
            new_img(i,j) = convolute(orig_matrix, filter);
        end
    end
end

function results = test_mean_filter_parameters(filter_sizes, orig_img, noisy_img)
    len = length(filter_sizes);
    results = zeros(len, 1);
    
    for i=1:len
        mean_filter = mean_filter_of_size(filter_sizes(i));
        mean_img = filter_transform(noisy_img, mean_filter);
        psnr_mean = PSNR(orig_img, mean_img);
        results(i) = psnr_mean;
    end
end

function results = test_gaussian_filter_parameters(filter_sizes, sigmas, orig_img, noisy_img)
    sz_len = length(filter_sizes);
    sig_len = length(sigmas);
    results = zeros(sz_len, sig_len);
    
    for i=1:sz_len
        for j=1:sig_len
            gaussian_filter = gaussian_filter_of_size(filter_sizes(i),sigmas(j));
            gaussian_img = filter_transform(noisy_img, gaussian_filter);
            psnr_gaussian = PSNR(orig_img, gaussian_img);
            results(i,j) = psnr_gaussian;
        end
    end
end

%% 2(b) helper
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

function results = test_bilateral_filter_parameters(filter_sizes, sigma_cs, ...
                    sigma_ss,orig_img, noisy_img)
    sz_len = length(filter_sizes);
    sig_c_len = length(sigma_cs);
    sig_s_len = length(sigma_ss);
    results = zeros(sz_len, sig_c_len, sig_s_len);
    
    for i=1:sz_len
        for j=1:sig_c_len
            for k=1:sig_s_len
                bilateral_img = bilateral_filter_transform(noisy_img, ...
                                    filter_sizes(i), ... 
                                    sigma_cs(j), ... 
                                    sigma_ss(k));
                psnr_bilateral = PSNR(orig_img, bilateral_img);
                results(i,j,k) = psnr_bilateral;
            end
        end
    end
end

%% 2(c) helper
function new_img = nlm_filter_transform(img, search_wd_sz, ...
                                        comp_wd_sz, degree_smoothing)
    [nrows, ncols] = size(img);
    new_img = zeros(nrows, ncols);
    new_img = uint8(new_img);
                                    
    if nargin == 1
        new_img(:,:) = imnlmfilt(img(:,:));
    elseif nargin == 2
        new_img(:,:) = imnlmfilt(img(:,:), 'SearchWindowSize', search_wd_sz);
    elseif nargin == 3
        new_img(:,:) = imnlmfilt(img(:,:), 'SearchWindowSize', search_wd_sz,...
            'ComparisonWindowSize', comp_wd_sz);
    elseif nargin == 4
        new_img(:,:) = imnlmfilt(img(:,:), 'SearchWindowSize', search_wd_sz,...
            'ComparisonWindowSize', comp_wd_sz,...
            'DegreeOfSmoothing', degree_smoothing);
    end
end

%% 2(d) helper
function new_img = median_filter_transform(img, sz, only_extreme)
    if nargin == 2
        only_extreme = false;
    end

    padded_img = zero_pad(img, sz);
    [nrows, ncols] = size(img);
    new_img = zeros(nrows, ncols);
    new_img = uint8(new_img);
    
    for i=1:nrows
        for j=1:ncols
            if only_extreme
                if img(i,j)==0 || img(i,j)==255
                    orig_matrix = padded_img(i:(i+2*sz),j:(j+2*sz));
                    new_img(i,j) = median(orig_matrix(:));
                else
                    new_img(i,j) = img(i,j);
                end
            else
                orig_matrix = padded_img(i:(i+2*sz),j:(j+2*sz));
                new_img(i,j) = median(orig_matrix(:));
            end
        end
    end
end

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

function psnr = PSNR_rgb(img1, img2)
    psnr_r = PSNR(img1(:,:,1), img2(:,:,1));
    psnr_g = PSNR(img1(:,:,2), img2(:,:,2));
    psnr_b = PSNR(img1(:,:,3), img2(:,:,3));
    
    psnr = (psnr_r+psnr_g+psnr_b) / 3;
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