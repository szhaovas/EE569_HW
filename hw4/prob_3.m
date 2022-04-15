% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Mar 27

%%
cat1 = read_img_rgb('Cat_1.raw', 400, 600);
cat_dog = read_img_rgb('Cat_Dog.raw', 400, 600);
[cat1_f,cat1_d,~,cat_dog1_f,cat_dog1_d,~] = closest_to_largest(cat1, cat_dog, 5, 2);

figure;
subplot(1,2,1);
imshow(cat1);
cat1_kpts = vl_plotframe(cat1_f);
set(cat1_kpts,'color','y','linewidth',2);

subplot(1,2,2);
imshow(cat_dog);
cat_dog1_kpts = vl_plotframe(cat_dog1_f);
set(cat_dog1_kpts,'color','y','linewidth',2);

%%
dog1 = read_img_rgb('Dog_1.raw', 400, 600);
[dog1_f,dog1_d,~,cat_dog1_f,~,~] = closest_to_largest(dog1, cat_dog, 5, 2);

figure;
subplot(1,2,1);
imshow(dog1);
dog1_kpts = vl_plotframe(dog1_f);
set(dog1_kpts,'color','y','linewidth',2);

subplot(1,2,2);
imshow(cat_dog);
cat_dog1_kpts = vl_plotframe(cat_dog1_f);
set(cat_dog1_kpts,'color','y','linewidth',2);

%%
cat2 = read_img_rgb('Cat_2.raw', 400, 600);
[cat1_f,~,~,cat2_f,cat2_d,~] = closest_to_largest(cat1, cat2, 5, 2);

figure;
subplot(1,2,1);
imshow(cat1);
cat1_kpts = vl_plotframe(cat1_f);
set(cat1_kpts,'color','y','linewidth',2);

subplot(1,2,2);
imshow(cat2);
cat2_kpts = vl_plotframe(cat2_f);
set(cat2_kpts,'color','y','linewidth',2);

%%
[cat1_f,~,~,dog1_f,~,~] = closest_to_largest(cat1, dog1, 5, 2);

figure;
subplot(1,2,1);
imshow(cat1);
cat1_kpts = vl_plotframe(cat1_f);
set(cat1_kpts,'color','y','linewidth',2);

subplot(1,2,2);
imshow(dog1);
dog1_kpts = vl_plotframe(dog1_f);
set(dog1_kpts,'color','y','linewidth',2);

%%
dog2 = read_img_rgb('Dog_2.raw', 400, 600);
dog2_y = rgb_yuv(dog2); dog2_y = single(dog2_y(:,:,1));
[dog2_f,dog2_d] = vl_sift(dog2_y, 'PeakThresh', 5, 'edgethresh', 2);

% [labels,centroids] = kmeans(double([cat1_d,cat2_d,dog1_d,cat_dog1_d,dog2_d])', 8);
[labels,centroids] = kmeans(double([cat1_d,cat2_d,dog1_d,cat_dog1_d])', 8);
cat1_labels = labels(1:size(cat1_d,2));
cat2_labels = labels(1+size(cat1_d,2):size(cat1_d,2)+size(cat2_d,2));
dog1_labels = labels(1+size(cat1_d,2)+size(cat2_d,2):...
    size(cat1_d,2)+size(cat2_d,2)+size(dog1_d,2));
cat_dog1_labels = labels(1+size(cat1_d,2)+size(cat2_d,2)+size(dog1_d,2):...
    size(cat1_d,2)+size(cat2_d,2)+size(dog1_d,2)+size(cat_dog1_d,2));
% dog2_labels = labels(1+size(cat1_d,2)+size(cat2_d,2)+size(dog1_d,2)+size(cat_dog1_d,2):...
%     size(cat1_d,2)+size(cat2_d,2)+size(dog1_d,2)+size(cat_dog1_d,2)+size(dog2_d,2));
dog2_distances = pdist2(centroids, dog2_d', 'seuclidean');
[~,dog2_labels] = min(dog2_distances);

cat1_hist = histogram_bins(cat1_labels);
figure;
subplot(2,2,1)
bar(1:8,cat1_hist);
title('cat1');

cat2_hist = histogram_bins(cat2_labels);
subplot(2,2,2)
bar(1:8,cat2_hist);
title('cat2');

dog1_hist = histogram_bins(dog1_labels);
subplot(2,2,3)
bar(1:8,dog1_hist);
title('dog1');

dog2_hist = histogram_bins(dog2_labels);
subplot(2,2,4)
bar(1:8,dog2_hist);
title('dog2');

dog2_cat1_score = compare_histograms(dog2_hist, cat1_hist)
dog2_dog1_score = compare_histograms(dog2_hist, dog1_hist)

%%
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

%%
function [img1_f, img1_d, max_idx, img2_f, img2_d, closest_idx] = ...
    closest_to_largest(img1, img2, peak_thresh, edge_thresh)
    if nargin < 4
        edge_thresh = 10;
        if nargin < 3
            peak_thresh = 0;
        end
    end
    img1_y = rgb_yuv(img1); img1_y = single(img1_y(:,:,1));
%     img1_y = single(rgb2gray(img1));
    [img1_f,img1_d] = vl_sift(img1_y, 'PeakThresh', peak_thresh, 'edgethresh', edge_thresh);
    [~,max_idx] = max(img1_f(3,:));
    figure;
    subplot(1,2,1);
    imshow(img1);
    img1_kpts = vl_plotframe(img1_f(:,max_idx));
    set(img1_kpts,'color','y','linewidth',2);
%     hold on;
%     plot(img1_f(1,max_idx),img1_f(2,max_idx),'r+', 'MarkerSize', 50);
%     hold off;

    img2_y = rgb_yuv(img2); img2_y = single(img2_y(:,:,1));
%     img2_y = single(rgb2gray(img2));
    [img2_f,img2_d] = vl_sift(img2_y, 'PeakThresh', peak_thresh, 'edgethresh', edge_thresh);

    img1_features = double(img1_d(:,max_idx));
    closest_dist = Inf;
    closest_idx = 0;
    for i=1:size(img2_d,2)
        dist = sum((double(img2_d(:,i)) - img1_features) .^ 2);
        if dist < closest_dist
            closest_dist = dist;
            closest_idx = i;
        end
    end

%     [~,closest_idx] = min(sum((double(img2_d) - double(img1_features)) .^ 2));
%     [closest_idx, scores] = vl_ubcmatch(img1_d(:,max_idx), img2_d, 0);
%     closest_idx = closest_idx(2);

    pause(0.5);
    subplot(1,2,2);
    imshow(img2);
    img2_kpts = vl_plotframe(img2_f(:,closest_idx));
    set(img2_kpts,'color','y','linewidth',2);
%     hold on;
%     plot(img2_f(1,closest_idx),img2_f(2,closest_idx),'r+', 'MarkerSize', 50);
%     hold off;
end

%%
function bins = histogram_bins(labels)
    nlbs = length(labels);
    bins = zeros(1,8);
    
    for i=1:nlbs
        bins(labels(i)) = bins(labels(i)) + 1;
    end
    
    bins = bins ./ sum(bins);
end

function score = compare_histograms(hist1, hist2)
    assert(length(hist1) == length(hist2));
    k = length(hist1);
    
    sum1 = 0;
    sum2 = 0;
    for i=1:k
        if hist1(k) < hist2(k)
            sum1 = sum1 + hist1(k);
            sum2 = sum2 + hist2(k);
        else
            sum1 = sum1 + hist2(k);
            sum2 = sum2 + hist1(k);
        end
    end
%     for i=1:k
%         sum1 = sum1 + min(hist1(k),hist2(k));
%         sum2 = sum2 + max(hist1(k),hist2(k));
%     end
    
    score = double(sum1) / double(sum2);
end