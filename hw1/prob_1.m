% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Jan 30

%% 1(a)
house_img = read_img_gray('images/House.raw', 512, 768);
padded_house_img = odd_mirror_pad(house_img);
demosaic_house_img = bilinear(padded_house_img);
figure
imshow(demosaic_house_img)

%% 1(b)
% (1)
hat_img = read_img_gray('images/Hat.raw', 256, 256);
hat_hist_bins = histogram_bins(hat_img);
figure
bar(0:255,hat_hist_bins)
title('original image histogram')
% (2)
new_hat_img_transfer_func = transfer_func_equalization(hat_img);
%   enhanced image
figure
imshow(new_hat_img_transfer_func)
%   transfer function
lut_sum_bins = cumulative_hist_bins(hat_img);
hat_img_lut = scaled_histogram_bins(lut_sum_bins);
figure
plot(0:255, hat_img_lut)
title('transfer function')
xlabel('old intensity')
ylabel('new intensity')
% (3)
new_hat_img_bucket_fill = bucket_fill_equalization(hat_img);
%   enhanced image
figure
imshow(new_hat_img_bucket_fill)
%   old cumulative histogram
old_cumulative_hist_bins = cumulative_hist_bins(hat_img);
figure
bar(0:255,old_cumulative_hist_bins)
title('original image cumulative histogram')
%   new cumulative histogram
new_cumulative_hist_bins = cumulative_hist_bins(new_hat_img_bucket_fill);
figure
bar(0:255,new_cumulative_hist_bins)
title('new image cumulative histogram')

%% 1(c)
fog_img = read_img_rgb('images/Taj_Mahal.raw', 400, 600);
yuv_pre_sharp = rgb_yuv(fog_img);
%   transfer function equalization
yuv_post_cdf = transform_lum_merge(@transfer_func_equalization, yuv_pre_sharp);
rgb_post_cdf = yuv_rgb(yuv_post_cdf);
figure
imshow(rgb_post_cdf)
%   bucket fill equalization
yuv_post_bin = transform_lum_merge(@bucket_fill_equalization, yuv_pre_sharp);
rgb_post_bin = yuv_rgb(yuv_post_bin);
figure
imshow(rgb_post_bin)
%   CLAHE
yuv_post_clahe_try1 = transform_adapthisteq_merge(yuv_pre_sharp, [8,8], 0.01);
rgb_post_clahe_try1 = yuv_rgb(yuv_post_clahe_try1);
figure
imshow(rgb_post_clahe_try1)
yuv_post_clahe_try2 = transform_adapthisteq_merge(yuv_pre_sharp, [4,4], 0.01);
rgb_post_clahe_try2 = yuv_rgb(yuv_post_clahe_try2);
figure
imshow(rgb_post_clahe_try2)
yuv_post_clahe_try3 = transform_adapthisteq_merge(yuv_pre_sharp, [16,16], 0.01);
rgb_post_clahe_try3 = yuv_rgb(yuv_post_clahe_try3);
figure
imshow(rgb_post_clahe_try3)

%% 1(a) helper
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

function padded_img = odd_mirror_pad(img_matrix)
    [nrows, ncols] = size(img_matrix);
    padded_img = zeros(nrows+2, ncols+2);
    % top and bottom rows
    padded_img(1,2:ncols+1) = img_matrix(2,:);
    padded_img(end,2:ncols+1) = img_matrix(end-1,:);
    % leftmost and rightmost cols
    padded_img(2:nrows+1,1) = img_matrix(:,2);
    padded_img(2:nrows+1,end) = img_matrix(:,end-1);
    % four corners
    padded_img(1,1) = img_matrix(2,2);
    padded_img(1,end) = img_matrix(2,end-1);
    padded_img(end,1) = img_matrix(end-1,2);
    padded_img(end,end) = img_matrix(end-1,end-1);
    
    padded_img(2:end-1,2:end-1) = img_matrix(:,:);
end

function demosaic_img = bilinear(padded_img)
    vertical_filter = [0 0.5 0; 0 0 0; 0 0.5 0];
    horizontal_filter = [0 0 0; 0.5 0 0.5; 0 0 0];
    cross_filter = [0 0.25 0; 0.25 0 0.25; 0 0.25 0];
    diagonal_filter = [0.25 0 0.25; 0 0 0; 0.25 0 0.25];
    
    [nrows, ncols] = size(padded_img);
    nrows = nrows - 2;
    ncols = ncols - 2;
    demosaic_img = zeros(nrows, ncols, 3);
    demosaic_img = uint8(demosaic_img);
    for i=1:nrows
        for j=1:ncols
            orig_matrix = padded_img(i:i+2,j:j+2);
            % green cell, horizontal filter for red, vertical filter for blue
            if mod(i,2) == 1 && mod(j,2) == 1
                demosaic_img(i,j,1) = convolute(orig_matrix,horizontal_filter);
                demosaic_img(i,j,2) = padded_img(i+1,j+1);
                demosaic_img(i,j,3) = convolute(orig_matrix,vertical_filter);
            % green cell, vertical filter for red, horizontal filter for blue
            elseif mod(i,2) == 0 && mod(j,2) == 0
                demosaic_img(i,j,1) = convolute(orig_matrix,vertical_filter);
                demosaic_img(i,j,2) = padded_img(i+1,j+1);
                demosaic_img(i,j,3) = convolute(orig_matrix,horizontal_filter);
            % red cell, cross filter for green, diagonal filter for blue
            elseif mod(i,2) == 1 && mod(j,2) == 0
                demosaic_img(i,j,1) = padded_img(i+1,j+1);
                demosaic_img(i,j,2) = convolute(orig_matrix,cross_filter);
                demosaic_img(i,j,3) = convolute(orig_matrix,diagonal_filter);
            % blue cell, diagonal filter for red, cross filter for green
            else
                demosaic_img(i,j,1) = convolute(orig_matrix,diagonal_filter);
                demosaic_img(i,j,2) = convolute(orig_matrix,cross_filter);
                demosaic_img(i,j,3) = padded_img(i+1,j+1);
            end
        end
    end
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

%% 1(b) helper
function bins = histogram_bins(matrix)
    [nrows, ncols] = size(matrix);
    bins = zeros(256,1);
    
    for i=1:nrows
        for j=1:ncols
%             bin_index = max(ceil(double(matrix(i,j))/interval), 1);
            bins(matrix(i,j)+1) = bins(matrix(i,j)+1) + 1;
        end
    end
end

function sum_bins = sum_histogram_bins(histogram_bins)
    sum_bins = zeros(256,1);
    
    sum = 0;
    for i=1:256
        sum = sum + histogram_bins(i);
        sum_bins(i) = sum;
    end
end

function sum_bins = cumulative_hist_bins(matrix)
    bins = histogram_bins(matrix);
    sum_bins = sum_histogram_bins(bins);
end

function lut = scaled_histogram_bins(sum_bins)
    num_pixels = sum_bins(end);
    lut = zeros(256,1);
    lut = uint8(lut);
    
    for i=1:256
        lut(i) = uint8(255 * sum_bins(i)/num_pixels);
    end
end

function new_image = lut_lookup(old_img, lut)
    [nrows, ncols] = size(old_img);
    new_image = zeros(nrows, ncols);
    new_image = uint8(new_image);
    
    for i=1:nrows
        for j=1:ncols
%             lut_index = max(ceil(len*double(old_img(i,j))/255), 1);
            new_image(i,j) = lut(old_img(i,j)+1);
        end
    end
end

function new_image = transfer_func_equalization(old_img)
    bins = histogram_bins(old_img);
    sum_bins = sum_histogram_bins(bins);
    lut = scaled_histogram_bins(sum_bins);
    new_image = lut_lookup(old_img, lut);
end

function bins = pixels_by_intensities(matrix)
    [nrows, ncols] = size(matrix);
    bins = cell(256,1);
    
    % randomly access pixels to avoid bias
    %   optional; if this goes against the rule for using only basic
    %   functions then simply iterate the image the normal way
    for i=randperm(nrows)
        for j=randperm(ncols)
            bin_index = uint8(matrix(i,j)+1);
            bins{bin_index}(:,end+1) = [i j];
        end
    end
end

function new_image = bin_assignment(matrix, bins)
    [nrows, ncols] = size(matrix);
    npx = nrows * ncols;
    npx_per_bin = floor(double(npx) / 256);
    bin_counter = 0;
    bin_pointer = 1;
    
    new_image = zeros(nrows, ncols)+1;
    new_image = uint8(new_image);
    
    for i=1:256
        locs = bins{i};
        sz = size(locs);
        len = sz(2);
        for j=1:len
            r_c = bins{i}(:,j);
            % current bin is filled, move to next bin
            if bin_counter >= npx_per_bin
                bin_pointer = bin_pointer + 1;
                new_image(r_c(1),r_c(2)) = bin_pointer;
                bin_counter = 0;
            % current bin has space, fill with bin value
            else
                new_image(r_c(1),r_c(2)) = bin_pointer;
                bin_counter = bin_counter + 1;
            end 
        end
    end
end

function new_image = bucket_fill_equalization(old_img)
    intensity_bins = pixels_by_intensities(old_img);
    new_image = bin_assignment(old_img, intensity_bins);
end

%% 1(c) helper
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

function new_image = rgb_yuv(old_img)
    nrows = size(old_img, 1);
    ncols = size(old_img, 2);
    new_image = zeros(nrows, ncols, 3);
    % method A and B use histogram bins so must be discrete
    new_image = uint8(new_image);
    
    for i=1:nrows
        for j=1:ncols
            % Y
            new_image(i,j,1) = 0.257*double(old_img(i,j,1)) ...
                               + 0.504*double(old_img(i,j,2)) ...
                               + 0.098*double(old_img(i,j,3)) + 16;
            % U
            new_image(i,j,2) = -0.148*double(old_img(i,j,1)) ...
                               - 0.291*double(old_img(i,j,2)) ...
                               + 0.439*double(old_img(i,j,3)) + 128;
            % V
            new_image(i,j,3) = 0.439*double(old_img(i,j,1)) ...
                               - 0.368*double(old_img(i,j,2)) ...
                               - 0.071*double(old_img(i,j,3)) + 128;
        end
    end
end

function new_image = yuv_rgb(old_img)
    nrows = size(old_img, 1);
    ncols = size(old_img, 2);
    new_image = zeros(nrows, ncols, 3);
    new_image = uint8(new_image);
    
    for i=1:nrows
        for j=1:ncols
            new_image(i,j,1) = uint8(1.164*(old_img(i,j,1) - 16) ...
                               + 1.596*(old_img(i,j,3) - 128));
            new_image(i,j,2) = uint8(1.164*(old_img(i,j,1) - 16) ...
                               - 0.813*(old_img(i,j,3) - 128) ...
                               - 0.391*(old_img(i,j,2) - 128));
            new_image(i,j,3) = uint8(1.164*(old_img(i,j,1) - 16) ...
                               + 2.018*(old_img(i,j,2) - 128));
        end
    end
end

function new_yuv_tensor = transform_lum_merge(func, yuv_tensor)
    nrows = size(yuv_tensor, 1);
    ncols = size(yuv_tensor, 2);
    lum_matrix = yuv_tensor(:,:,1);
    
    new_lum = func(lum_matrix);
    
    new_yuv_tensor = zeros(nrows, ncols, 3);
    new_yuv_tensor(:,:,1) = new_lum;
    new_yuv_tensor(:,:,2) = yuv_tensor(:,:,2);
    new_yuv_tensor(:,:,3) = yuv_tensor(:,:,3);
end

% scale luminance from [16 235] to [0 1]
function scaled = scale_lum(matrix)
    [nrows, ncols] = size(matrix);
    scaled = zeros(nrows, ncols);
    range = 235 - 16;
    for i=1:nrows
        for j=1:ncols
            scaled(i,j) = (double(matrix(i,j))-16)/range;
        end
    end
end

% scale luminance from [0 1] to [16 235]
function rescaled = rescale_lum(matrix)
    [nrows, ncols] = size(matrix);
    rescaled = zeros(nrows, ncols);
    range = 235 - 16;
    for i=1:nrows
        for j=1:ncols
            rescaled(i,j) = double(matrix(i,j)) * range + 16;
        end
    end
end

function new_yuv_tensor = transform_adapthisteq_merge(yuv_tensor, num_tiles,...
                            clip_lim, nbins)
    nrows = size(yuv_tensor, 1);
    ncols = size(yuv_tensor, 2);
    lum_matrix = yuv_tensor(:,:,1);
    scaled_lum = scale_lum(lum_matrix);
    
    if nargin == 1
        new_lum = adapthisteq(scaled_lum);
    elseif nargin == 2
        new_lum = adapthisteq(scaled_lum, 'NumTiles', num_tiles);
    elseif nargin == 3
        new_lum = adapthisteq(scaled_lum, 'NumTiles', num_tiles, ...
                                'ClipLimit', clip_lim);
    elseif nargin == 4
        new_lum = adapthisteq(scaled_lum, 'NumTiles', num_tiles, ...
                                'ClipLimit', clip_lim, ...
                                'NBins', nbins);
    end
    rescaled_lum = rescale_lum(new_lum);
    
    new_yuv_tensor = zeros(nrows, ncols, 3);
    new_yuv_tensor(:,:,1) = rescaled_lum;
    new_yuv_tensor(:,:,2) = yuv_tensor(:,:,2);
    new_yuv_tensor(:,:,3) = yuv_tensor(:,:,3);
end