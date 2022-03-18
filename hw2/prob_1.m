% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Feb 20

%% 1(a)
% tiger
tiger = read_img_rgb('HW2_images/Tiger.raw', 321, 481);
figure
subplot(2,3,1);
imshow(tiger)
title('original')
tiger_lum = rgb_lum(tiger);
[tiger_x, tiger_y] = get_gradients(tiger_lum);
scl_tiger_x = scale0255(tiger_x);
subplot(2,3,2);
imshow(scl_tiger_x)
title('gradient x')
scl_tiger_y = scale0255(tiger_y);
subplot(2,3,3);
imshow(scl_tiger_y)
title('gradient y')
tiger_scl_mag = magnitude0255(tiger_x, tiger_y);
subplot(2,3,4);
imshow(tiger_scl_mag)
title('gradient magnitude')
tiger_threshold = get_cutoff_val(tiger_scl_mag, 0.9);
tiger_sobel = get_edge(tiger_scl_mag, tiger_threshold);
subplot(2,3,5);
imshow(tiger_sobel)
title('thresholded')

% pig
pig = read_img_rgb('HW2_images/Pig.raw', 321, 481);
figure
subplot(2,3,1);
imshow(pig)
title('original')
pig_lum = rgb_lum(pig);
[pig_x, pig_y] = get_gradients(pig_lum);
scl_pig_x = scale0255(pig_x);
subplot(2,3,2);
imshow(scl_pig_x)
title('gradient x')
scl_pig_y = scale0255(pig_y);
subplot(2,3,3);
imshow(scl_pig_y)
title('gradient y')
pig_scl_mag = magnitude0255(pig_x, pig_y);
subplot(2,3,4);
imshow(pig_scl_mag)
title('gradient magnitude')
pig_threshold = get_cutoff_val(pig_scl_mag, 0.9);
pig_sobel = get_edge(pig_scl_mag, pig_threshold);
subplot(2,3,5);
imshow(pig_sobel)
title('thresholded')

%% 1(b)
tiger_canny = edge(tiger_lum,'canny', [0.1 0.4]);
figure
subplot(2,3,1);
imshow(1-tiger_canny)
title('[0.1 0.4]')
subplot(2,3,2);
imshow(1-edge(tiger_lum,'canny', [0.1 0.5]))
title('[0.1 0.5]')
subplot(2,3,3);
imshow(1-edge(tiger_lum,'canny', [0.2 0.5]))
title('[0.2 0.5]')
subplot(2,3,4);
imshow(1-edge(tiger_lum,'canny', [0.2 0.6]))
title('[0.2 0.6]')
subplot(2,3,5);
imshow(1-edge(tiger_lum,'canny', [0.3 0.6]))
title('[0.3 0.6]')
subplot(2,3,6);
imshow(1-edge(tiger_lum,'canny', [0.3 0.7]))
title('[0.3 0.7]')
pig_canny = edge(pig_lum,'canny', [0.1 0.4]);
figure
subplot(2,3,1);
imshow(1-pig_canny)
title('[0.1 0.4]')
subplot(2,3,2);
imshow(1-edge(pig_lum,'canny', [0.1 0.5]))
title('[0.1 0.5]')
subplot(2,3,3);
imshow(1-edge(pig_lum,'canny', [0.2 0.5]))
title('[0.2 0.5]')
subplot(2,3,4);
imshow(1-edge(pig_lum,'canny', [0.2 0.6]))
title('[0.2 0.6]')
subplot(2,3,5);
imshow(1-edge(pig_lum,'canny', [0.3 0.6]))
title('[0.3 0.6]')
subplot(2,3,6);
imshow(1-edge(pig_lum,'canny', [0.3 0.7]))
title('[0.3 0.7]')

%% 1(c)
cd edges-master
% NOTE: the following codes assume edges-master is one level below the
% directory containing HW2_images
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory
global model
tic, model=edgesTrain(opts); toc; % will load model if already trained
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=1;                 % set to true to enable nms
tiger_E = edgesDetect(tiger,model);
tiger_SE = (tiger_E > 0.1);
figure
imshow(1-(tiger_SE))
pig_E = edgesDetect(pig,model);
pig_SE = (pig_E > 0.1);
figure
imshow(1-(pig_SE))

%% 1(d)(1)
F = @(P,R) 2 * P .* R ./ (P + R);
% tiger
[tiger_sobel_P, tiger_sobel_R] = PR_each_gt((tiger_sobel==0), '../HW2_images/Tiger_GT.mat');
tiger_sobel_table = PRF_table(tiger_sobel_P, tiger_sobel_R);
tiger_sobel_overall_F = F(tiger_sobel_table{'avg',1},tiger_sobel_table{'avg',2});
[tiger_canny_P, tiger_canny_R] = PR_each_gt(tiger_canny, '../HW2_images/Tiger_GT.mat');
tiger_canny_table = PRF_table(tiger_canny_P, tiger_canny_R);
tiger_canny_overall_F = F(tiger_canny_table{'avg',1},tiger_canny_table{'avg',2});
[tiger_SE_P, tiger_SE_R] = PR_each_gt(tiger_SE, '../HW2_images/Tiger_GT.mat');
tiger_SE_table = PRF_table(tiger_SE_P, tiger_SE_R);
tiger_SE_overall_F = F(tiger_SE_table{'avg',1},tiger_SE_table{'avg',2});
% pig
[pig_sobel_P, pig_sobel_R] = PR_each_gt((pig_sobel==0), '../HW2_images/Pig_GT.mat');
pig_sobel_table = PRF_table(pig_sobel_P, pig_sobel_R);
pig_sobel_overall_F = F(pig_sobel_table{'avg',1},pig_sobel_table{'avg',2});
[pig_canny_P, pig_canny_R] = PR_each_gt(pig_canny, '../HW2_images/Pig_GT.mat');
pig_canny_table = PRF_table(pig_canny_P, pig_canny_R);
pig_canny_overall_F = F(pig_canny_table{'avg',1},pig_canny_table{'avg',2});
[pig_SE_P, pig_SE_R] = PR_each_gt(pig_SE, '../HW2_images/Pig_GT.mat');
pig_SE_table = PRF_table(pig_SE_P, pig_SE_R);
pig_SE_overall_F = F(pig_SE_table{'avg',1},pig_SE_table{'avg',2});

%% 1(d)(2)
sobel_thrs = num2cell((80:99)/100);
canny_thrs = {[0.1,0.2],[0.1,0.4],[0.2,0.4],[0.2,0.8],[0.4,0.8]};
SE_thrs = num2cell((5:15)/100);
% tiger
[tiger_sobel_P_mat, tiger_sobel_R_mat] = PR_each_gt_thrs(@sobel_edge, tiger_lum, '../HW2_images/Tiger_GT.mat', sobel_thrs);
tiger_sobel_P_meanByGt = mean(tiger_sobel_P_mat, 1);
tiger_sobel_R_meanByGt = mean(tiger_sobel_R_mat, 1);
tiger_sobel_F = F(tiger_sobel_P_meanByGt, tiger_sobel_R_meanByGt);
[tiger_canny_P_mat, tiger_canny_R_mat] = PR_each_gt_thrs(@canny_edge, tiger_lum, '../HW2_images/Tiger_GT.mat', canny_thrs);
tiger_canny_P_meanByGt = mean(tiger_canny_P_mat, 1);
tiger_canny_R_meanByGt = mean(tiger_canny_R_mat, 1);
tiger_canny_F = F(tiger_canny_P_meanByGt, tiger_canny_R_meanByGt);
[tiger_SE_P_mat, tiger_SE_R_mat] = PR_each_gt_thrs(@SE_edge, tiger, '../HW2_images/Tiger_GT.mat', SE_thrs);
tiger_SE_P_meanByGt = mean(tiger_SE_P_mat, 1);
tiger_SE_R_meanByGt = mean(tiger_SE_R_mat, 1);
tiger_SE_F = F(tiger_SE_P_meanByGt, tiger_SE_R_meanByGt);
% pig
[pig_sobel_P_mat, pig_sobel_R_mat] = PR_each_gt_thrs(@sobel_edge, pig_lum, '../HW2_images/Pig_GT.mat', sobel_thrs);
pig_sobel_P_meanByGt = mean(pig_sobel_P_mat, 1);
pig_sobel_R_meanByGt = mean(pig_sobel_R_mat, 1);
pig_sobel_F = F(pig_sobel_P_meanByGt, pig_sobel_R_meanByGt);
[pig_canny_P_mat, pig_canny_R_mat] = PR_each_gt_thrs(@canny_edge, pig_lum, '../HW2_images/Pig_GT.mat', canny_thrs);
pig_canny_P_meanByGt = mean(pig_canny_P_mat, 1);
pig_canny_R_meanByGt = mean(pig_canny_R_mat, 1);
pig_canny_F = F(pig_canny_P_meanByGt, pig_canny_R_meanByGt);
[pig_SE_P_mat, pig_SE_R_mat] = PR_each_gt_thrs(@SE_edge, pig, '../HW2_images/Pig_GT.mat', SE_thrs);
pig_SE_P_meanByGt = mean(pig_SE_P_mat, 1);
pig_SE_R_meanByGt = mean(pig_SE_R_mat, 1);
pig_SE_F = F(pig_SE_P_meanByGt, pig_SE_R_meanByGt);

%% 
figure
subplot(3,2,1);
plot((80:99)/100, tiger_sobel_F)
title('tiger sobel F-scores')
subplot(3,2,2);
plot((80:99)/100, pig_sobel_F)
title('pig sobel F-scores')
subplot(3,2,3);
plot3([0.1,0.1,0.2,0.2,0.4],[0.2,0.4,0.4,0.8,0.8],tiger_canny_F)
title('tiger canny F-scores')
subplot(3,2,4);
plot3([0.1,0.1,0.2,0.2,0.4],[0.2,0.4,0.4,0.8,0.8],pig_canny_F)
title('pig canny F-scores')
subplot(3,2,5);
plot((5:15)/100, tiger_SE_F)
title('tiger SE F-scores')
subplot(3,2,6);
plot((5:15)/100, pig_SE_F)
title('pig SE F-scores')

%% 1(a) helper
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

function lum_mat = rgb_lum(rgb_mat)
    nrows = size(rgb_mat,1);
    ncols = size(rgb_mat,2);
    lum_mat = zeros(nrows, ncols);
    
    for i=1:nrows
        for j=1:ncols
            lum_mat(i,j) = 0.2989*rgb_mat(i,j,1) + ...
                0.5870*rgb_mat(i,j,2) + ...
                0.1140*rgb_mat(i,j,3);
        end
    end
end

function padded_img = odd_mirror_pad(img_mat)
    [nrows, ncols] = size(img_mat);
    padded_img = zeros(nrows+2, ncols+2);
    % top and bottom rows
    padded_img(1,2:ncols+1) = img_mat(2,:);
    padded_img(end,2:ncols+1) = img_mat(end-1,:);
    % leftmost and rightmost cols
    padded_img(2:nrows+1,1) = img_mat(:,2);
    padded_img(2:nrows+1,end) = img_mat(:,end-1);
    % four corners
    padded_img(1,1) = img_mat(2,2);
    padded_img(1,end) = img_mat(2,end-1);
    padded_img(end,1) = img_mat(end-1,2);
    padded_img(end,end) = img_mat(end-1,end-1);
    
    padded_img(2:end-1,2:end-1) = img_mat(:,:);
end

function sum = convolve(ori_mat, filter)
    [nrows, ncols] = size(ori_mat);
    sum = 0;
    for i=1:nrows
        for j=1:ncols
            sum = sum + ori_mat(i,j) * filter(i,j);
        end
    end
end

function [grad_x, grad_y] = get_gradients(img)
    sobel_x = [[0.25, 0, -0.25];...
                [0.5, 0, -0.5];...
                [0.25, 0, -0.25]];
    sobel_y = [[-0.25, -0.5, -0.25];...
                [0, 0, 0];...
                [0.25, 0.5, 0.25]];
    padded_img = odd_mirror_pad(img);
    [nrows, ncols] = size(img);
    grad_x = zeros(nrows, ncols);
    grad_y = zeros(nrows, ncols);
    
    for i=1:nrows
        for j=1:ncols
            orig_mat = padded_img(i:(i+2),j:(j+2));
            grad_x(i,j) = convolve(orig_mat, sobel_x);
            grad_y(i,j) = convolve(orig_mat, sobel_y);
        end
    end
end

function scl_mat = scale0255(orig_mat)
    [nrows, ncols] = size(orig_mat);
    scl_mat = zeros(nrows, ncols);
    scl_mat = uint8(scl_mat);
    
    max = -Inf;
    min = Inf;
    for i=1:nrows
        for j=1:ncols
            curr = orig_mat(i,j);
            if curr > max
                max = curr;
            end
            if curr < min
                min = curr;
            end
        end
    end
    
    range = max-min;
    for i=1:nrows
        for j=1:ncols
            scl_mat(i,j) = round(255*(orig_mat(i,j)-min)/double(range));
        end
    end
end

function scl_mag_mat = magnitude0255(grad_x, grad_y)
    [nrows, ncols] = size(grad_x);
    mag_mat = zeros(nrows, ncols);
    
    for i=1:nrows
        for j=1:ncols
            mag_mat(i,j) = sqrt(grad_x(i,j)^2+grad_y(i,j)^2);
        end
    end
    
    scl_mag_mat = scale0255(mag_mat);
end

function bins = histogram_bins(mat)
    [nrows, ncols] = size(mat);
    bins = zeros(256,1);
    
    for i=1:nrows
        for j=1:ncols
%             bin_index = max(ceil(double(mat(i,j))/interval), 1);
            bins(mat(i,j)+1) = bins(mat(i,j)+1) + 1;
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

function sum_bins = cumulative_hist_bins(mat)
    bins = histogram_bins(mat);
    sum_bins = sum_histogram_bins(bins);
end

function [threshold] = get_cutoff_val(mat, percent)
    bins = cumulative_hist_bins(mat);
    num_pixels = bins(end);
    for i=1:256
        if bins(i)/num_pixels >= percent
            threshold = i;
            return
        end
    end
end

function edge_mat = get_edge(scl_mag_mat, threshold)
    [nrows, ncols] = size(scl_mag_mat);
    edge_mat = ones(nrows, ncols)*255;
    edge_mat = uint8(edge_mat);
    
    for i=1:nrows
        for j=1:ncols
            if scl_mag_mat(i,j) > threshold
                edge_mat(i,j) = 0;
            end
        end
    end
end
%% 1(d) helper
function edge_mat = sobel_edge(img, threshold)
    [grad_x, grad_y] = get_gradients(img);
    scl_mag = magnitude0255(grad_x, grad_y);
    threshold = get_cutoff_val(scl_mag, threshold);
    edge_mat = (get_edge(scl_mag, threshold)==0);
end

function edge_mat = canny_edge(img, threshold)
    edge_mat = edge(img,'canny', threshold);
end

function edge_mat = SE_edge(img, threshold)
    global model
    E = edgesDetect(img,model);
    edge_mat = (E > threshold);
end

function [P, R] = PR_each_gt(E, G)
    G=load(G);
    G=G.groundTruth;
    r=length(G);
    P = zeros(r,1);
    R = zeros(r,1);
    for i=1:r
        groundTruth{1} = G{i};
        save('gt_temp.mat', 'groundTruth');
        [~, cntR, sumR, cntP, sumP, ~] = edgesEvalImg(E, 'gt_temp.mat', 'thrs', 1);
        delete('gt_temp.mat');
        P(i) = cntP/sumP;
        R(i) = cntR/sumR;
    end
end

function table = PRF_table(P, R)
    F = 2 * P .* R ./ (P + R);
    table_mat = [[P, R, F];...
        [mean(P),mean(R),mean(F)]];
    table = array2table(table_mat, 'VariableNames', {'P','R','F'}, ...
        'RowNames', {'gt1','gt2','gt3','gt4','gt5','avg'});
end

function [P, R] = PR_each_gt_thrs(alg, img, G, T)
    G=load(G);
    G=G.groundTruth;
    r=length(G);
    c=length(T);
    P = zeros(r,c);
    R = zeros(r,c);
    for i=1:r
        groundTruth{1} = G{i};
        save('gt_temp.mat', 'groundTruth');
        for j=1:c
            E = alg(img, T{j});
            [~, cntR, sumR, cntP, sumP, ~] = edgesEvalImg(E, 'gt_temp.mat', 'thrs', 1);
            P(i,j) = cntP/sumP;
            R(i,j) = cntR/sumR;
        end
        delete('gt_temp.mat');
    end
end