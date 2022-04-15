% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Mar 27

%% 1(a)
L5 = [1,4,6,4,1];
E5 = [-1,-2,0,2,1];
S5 = [-1,0,2,0,-1];
W5 = [-1,2,0,-2,1];
R5 = [1,-4,6,-4,1];
bases = [L5;E5;S5;W5;R5];
global filters;
filters = laws_filters(bases);

train_files = dir(fullfile('train','*.raw'));
ntrain = length(train_files);
feature_set = zeros(25, ntrain);
for i=1:ntrain
    img = read_img_gray(['train/', train_files(i).name], 128, 128);
    img = adapthisteq(img);
    img = double(img);
    img = (img - mean(mean(img))) ./ std(std(img));
%     img = double(img) ./ 255;
%     img = (img - min(min(img))) ./ (max(max(img)) - min(min(img)));
    feature_set(:,i) = get_feature_vector(img);
end

%% discriminant power
blanket_features = feature_set(:,1:9);
brick_features = feature_set(:,10:18);
grass_features = feature_set(:,19:27);
stones_features = feature_set(:,28:36);

blanket_feature_means = mean(blanket_features, 2);
brick_feature_means = mean(brick_features, 2);
grass_feature_means = mean(grass_features, 2);
stones_feature_means = mean(stones_features, 2);

global_feature_means = mean([blanket_feature_means, brick_feature_means, ...
    grass_feature_means, stones_feature_means], 2);

intra_class_variance = sum((blanket_features-blanket_feature_means) .^ 2, 2) + ...
    sum((brick_features-brick_feature_means) .^ 2, 2) + ...
    sum((grass_features-grass_feature_means) .^ 2, 2) + ...
    sum((stones_features-stones_feature_means) .^ 2, 2);
inter_class_variance = 9*((blanket_feature_means-global_feature_means) .^ 2) + ...
    9*((brick_feature_means-global_feature_means) .^ 2) + ...
    9*((grass_feature_means-global_feature_means) .^ 2) + ...
    9*((stones_feature_means-global_feature_means) .^ 2);

discriminant_power = intra_class_variance ./ inter_class_variance;

%% PCA & plot
feature_set_tp = feature_set';
centered_feature_set_tp = feature_set_tp - mean(feature_set_tp);
feature_pca = pca(feature_set_tp);
reduced_feature_set = centered_feature_set_tp * feature_pca(:,1:3);
figure;
scatter3(reduced_feature_set(1:9,1),reduced_feature_set(1:9,2), ...
    reduced_feature_set(1:9,3),36,"red");
hold on;
scatter3(reduced_feature_set(10:18,1),reduced_feature_set(10:18,2), ...
    reduced_feature_set(10:18,3),36,"green");
scatter3(reduced_feature_set(19:27,1),reduced_feature_set(19:27,2), ...
    reduced_feature_set(19:27,3),36,"blue");
scatter3(reduced_feature_set(28:36,1),reduced_feature_set(28:36,2), ...
    reduced_feature_set(28:36,3),36,"yellow");
legend([{'blanket'},{'brick'},{'grass'},{'stones'}])
hold off;

%% test accuracy
% test_files = dir(fullfile('test','*.raw'));
% ntest = length(test_files);
test_feature_set = zeros(25, 12);
for i=1:12
    img = read_img_gray(['test/', int2str(i), '.raw'], 128, 128);
    img = adapthisteq(img);
    img = double(img);
    img = (img - mean(mean(img))) ./ std(std(img));
%     img = double(img) ./ 255;
%     img = (img - min(min(img))) ./ (max(max(img)) - min(min(img)));
    test_feature_set(:,i) = get_feature_vector(img);
end

test_set_tp = test_feature_set';
centered_test_set_tp = test_set_tp - mean(test_set_tp);
reduced_test_set = centered_test_set_tp * feature_pca(:,1:3);

distances = pdist2(reduced_feature_set, reduced_test_set, 'mahalanobis');

test_labels = [3;1;1;4;4;3;2;4;2;2;1;3];
[~, nn_idx] = min(distances);
labels = ceil(nn_idx ./ 9)';
correct = labels == test_labels;
accuracy = sum(correct) / 12

%% 1(b)
train_labels = [ones(9,1);ones(9,1)*2;ones(9,1)*3;ones(9,1)*4];

[~, full_km_centroids, full_km_purities] = km_label(feature_set_tp, 4, train_labels);
[~, full_km_labels] = min(pdist2(full_km_centroids, test_set_tp, 'seuclidean'));
full_km_accuracy = sum(full_km_labels' == test_labels) / 12
[~, reduced_km_centroids, reduced_km_purities] = km_label(reduced_feature_set, 4, train_labels);
[~, reduced_km_labels] = min(pdist2(reduced_km_centroids, reduced_test_set, 'seuclidean'));
reduced_km_accuracy = sum(reduced_km_labels' == test_labels) / 12

rf_model = TreeBagger(500, reduced_feature_set, train_labels, 'Method', 'classification');
rf_labels = str2double(rf_model.predict(reduced_test_set));
rf_accuracy = sum(rf_labels == test_labels) / 12

t = templateSVM('Standardize',true,'KernelFunction','polynomial');
svm_model = fitcecoc(reduced_feature_set, train_labels,'Learners',t);
svm_labels = predict(svm_model, reduced_test_set);
svm_accuracy = sum(svm_labels == test_labels) / 12

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

function padded_img = odd_mirror_pad2(img_mat)
    [nrows, ncols] = size(img_mat);
    padded_img = zeros(nrows+4, ncols+4);
    % top and bottom rows
    padded_img(1,3:ncols+2) = img_mat(3,:);
    padded_img(2,3:ncols+2) = img_mat(2,:);
    padded_img(end-1,3:ncols+2) = img_mat(end-1,:);
    padded_img(end,3:ncols+2) = img_mat(end-2,:);
    % leftmost and rightmost cols
    padded_img(3:ncols+2,1) = img_mat(:,3);
    padded_img(3:ncols+2,2) = img_mat(:,2);
    padded_img(3:ncols+2,end-1) = img_mat(:,end-1);
    padded_img(3:ncols+2,end) = img_mat(:,end-2);
    % top left corner
    padded_img(1,1) = img_mat(3,3);
    padded_img(1,2) = img_mat(3,2);
    padded_img(2,1) = img_mat(2,3);
    padded_img(2,2) = img_mat(2,2);
    % top right corner
    padded_img(1,end) = img_mat(3,end-2);
    padded_img(1,end-1) = img_mat(3,end-1);
    padded_img(2,end) = img_mat(2,end-2);
    padded_img(2,end-1) = img_mat(2,end-1);
    % bottom left corner
    padded_img(end,1) = img_mat(end-2,3);
    padded_img(end,2) = img_mat(end-2,2);
    padded_img(end-1,1) = img_mat(end-1,3);
    padded_img(end-1,2) = img_mat(end-1,2);
    % bottom right corner
    padded_img(end,end) = img_mat(end-2,end-2);
    padded_img(end,end-1) = img_mat(end-2,end-1);
    padded_img(end-1,end) = img_mat(end-1,end-2);
    padded_img(end-1,end-1) = img_mat(end-1,end-1);
    % main body
    padded_img(3:end-2,3:end-2) = img_mat(:,:);
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

function vec = get_feature_vector(img)
    global filters;
    padded_img = odd_mirror_pad2(img);
    [nrows, ncols] = size(img);
    
    energy_matrix = zeros(nrows, ncols, 25);
    for v=1:25
        for r=1:nrows
            for c=1:ncols
                orig_matrix = padded_img(r:r+4,c:c+4);
                energy_matrix(r,c,v) = convolve(orig_matrix, filters(:,:,v));
            end
        end
    end
    energy_matrix(:,:,1) = energy_matrix(:,:,1) - mean(mean(energy_matrix(:,:,1)));
    vec = mean(mean(energy_matrix .^ 2));
end

%% 1(b) helper
function [labels, centroids, purities] = km_label(X, k, true_label, centroids)
    if nargin == 3
        centroids = 'plus';
    end
    nlbs = size(X, 1);
    labels = zeros(nlbs, 1);
    km_labels = kmeans(X, k, 'Distance', 'correlation', 'Start', centroids);
    
    centroids = zeros(k, size(X, 2));
    purities = zeros(k, 1);
    for i=1:k
        this_label = km_labels == i;
        new_label = mode(true_label(this_label));
        labels(this_label) = new_label;
        centroids(i,:) = mean(X(this_label,:));
        purities(i) = sum(true_label(this_label) == new_label) ...
            / sum(this_label);
    end
end