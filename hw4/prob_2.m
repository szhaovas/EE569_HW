% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Mar 27

%% 2(a)
L5 = [1,4,6,4,1];
E5 = [-1,-2,0,2,1];
S5 = [-1,0,2,0,-1];
W5 = [-1,2,0,-2,1];
R5 = [1,-4,6,-4,1];
bases = [L5;E5;S5;W5;R5];
global filters;
filters = laws_filters(bases);

img = read_img_gray('Mosaic.raw', 512, 512);
img = adapthisteq(img);
img = double(img);
img = (img - mean(mean(img))) ./ std(std(img));
emat = get_energy_matrix(img, 61, 20);
emat(:,:,2:25) = emat(:,:,2:25) ./ emat(:,:,1);
emat = emat(:,:,2:25);

%%
global colors;
colors = [[107,143,159];[114,99,107];[175,128,74];[167,57,32];[144,147,104];[157,189,204]];
[img_nrows, img_ncols] = size(img);
long_emat = reshape(emat, img_nrows*img_ncols, [], 1);
centered_long_emat = long_emat - mean(long_emat);
long_emat_pca = pca(long_emat);
long_emat = centered_long_emat * long_emat_pca(:,1:3);
labels = kmeans(long_emat, 6); labels = reshape(labels, [img_nrows, img_ncols]);
labels = mode_filter(labels, 101);
figure;
imshow(color_textures(labels));

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

function padded_img = odd_mirror_pad(img_mat, width)
    [nrows, ncols] = size(img_mat);
    padded_img = zeros(nrows+width*2, ncols+width*2);
    % top and bottom rows
    padded_img(1:width, 1+width:ncols+width) = flip(img_mat(2:width+1,:), 1);
    padded_img(end-width+1:end, 1+width:ncols+width) = flip(img_mat(end-width:end-1,:), 1);
    % leftmost and rightmost cols
    padded_img(1+width:nrows+width, 1:width) = flip(img_mat(:,2:width+1), 2);
    padded_img(1+width:nrows+width, end-width+1:end) = flip(img_mat(:,end-width:end-1), 2);
    % top left corner
    padded_img(1:width,1:width) = rot90(flip(img_mat(2:width+1,2:width+1), 2), -1)';
    % top right corner
    padded_img(1:width,end-width+1:end) = rot90(flip(img_mat(2:width+1,end-width:end-1), 2), -1)';
    % bottom left corner
    padded_img(end-width+1:end,1:width) = rot90(flip(img_mat(end-width:end-1,2:width+1), 2), -1)';
    % bottom right corner
    padded_img(end-width+1:end,end-width+1:end) = rot90(flip(img_mat(end-width:end-1,end-width:end-1), 2), -1)';
    % main body
    padded_img(1+width:end-width,1+width:end-width) = img_mat(:,:);
end

function filters = laws_filters(bases)
    nbases = size(bases, 1);
    dim = size(bases, 2);
    filters = zeros(dim, dim, nbases);
    
    for base1=1:nbases
        for base2=1:nbases
            filters(:,:,(base1-1)*nbases+base2) = ...
                bases(base1,:)' * bases(base2,:);
        end
    end
end

function sum = convolve(ori_mat, filter)
    assert(all(size(filter) == size(ori_mat)));
    [nrows, ncols] = size(ori_mat);
    sum = 0;
    for i=1:nrows
        for j=1:ncols
            sum = sum + ori_mat(i,j) * filter(i,j);
        end
    end
end

function energy_matrix = get_energy_matrix(img, window_size, gs_std)
    global filters;
    padded_img = odd_mirror_pad(img, 2);
    [nrows, ncols] = size(img);
    
    temp_energy_matrix = zeros(nrows, ncols, 25);
    for v=1:25
        for r=1:nrows
            for c=1:ncols
                orig_matrix = padded_img(r:r+4,c:c+4);
                temp_energy_matrix(r,c,v) = convolve(orig_matrix, filters(:,:,v));
            end
        end
    end
    
    window_width = (window_size-1) / 2;
    padded_energy_matrix = zeros(nrows+2*window_width, ncols+2*window_width, 25);
    for v=1:25
        padded_energy_matrix(:,:,v) = odd_mirror_pad(temp_energy_matrix(:,:,v), window_width);
    end
    
%     temp_first_matrix = zeros(nrows+2*window_width, ncols+2*window_width, 1);
%     temp_first_matrix(:,:) = padded_energy_matrix(:,:,1);
%     for r=1:nrows
%         for c=1:ncols
%             padded_energy_matrix(r+window_width,c+window_width,1) = ...
%                 temp_first_matrix(r+window_width,c+window_width) - mean(mean(...
%                 temp_first_matrix(r:r+window_width*2,c:c+window_width*2)));
%         end
%     end
    if nargin == 3
        gs_filter = gaussian_filter_of_size(window_width, gs_std);
    end

    energy_matrix = zeros(nrows, ncols, 25);
    for r=1:nrows
        for c=1:ncols
            if nargin == 2
                energy_matrix(r,c,:) = mean(mean(...
                    padded_energy_matrix(r:r+window_width*2,c:c+window_width*2,:) .^ 2));
            else
                for i=1:25
                    energy_matrix(r,c,i) = convolve(...
                        padded_energy_matrix(r:r+window_width*2,c:c+window_width*2,i) .^ 2, ...
                        gs_filter);
                end
            end
        end
    end
end

function img = color_textures(labels)
    global colors;
    [nrows, ncols] = size(labels);
    img = zeros(nrows, ncols, 3, 'uint8');
    
    for r=1:nrows
        for c=1:ncols
            img(r,c,:) = colors(labels(r,c),:);
        end
    end
end

%%
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

function new_img = mode_filter(img, window_size, num_iters)
    if nargin == 2
        num_iters = 1;
    end
    [nrows, ncols] = size(img);
    window_width = (window_size-1) / 2;
    padded_img = odd_mirror_pad(img, window_width);
    new_img = zeros(nrows, ncols);
    
    for i=1:num_iters
        for r=1:nrows
            for c=1:ncols
                orig_matrix = padded_img(r:r+2*window_width,c:c+2*window_width);
                new_img(r,c) = mode(mode(orig_matrix));
            end
        end
    end
end