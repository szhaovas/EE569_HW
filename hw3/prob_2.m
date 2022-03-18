%%
% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Mar 10

%%
% read images
left = read_img_rgb('raw_images/left.raw', 432, 576);
middle = read_img_rgb('raw_images/middle.raw', 432, 576);
right = read_img_rgb('raw_images/right.raw', 432, 576);
% extract luminance
left_y = rgb_yuv(left); left_y = left_y(:,:,1);
middle_y = rgb_yuv(middle); middle_y = middle_y(:,:,1);
right_y = rgb_yuv(right); right_y = right_y(:,:,1);
% SURF
left_points = detectSURFFeatures(left_y);
middle_points = detectSURFFeatures(middle_y);
right_points = detectSURFFeatures(right_y);
[left_features, left_points] = extractFeatures(left_y, left_points);
[middle_features, middle_points] = extractFeatures(middle_y, middle_points);
[right_features, right_points] = extractFeatures(right_y, right_points);
% match
lm_pairs = matchFeatures(left_features, middle_features, 'Unique', true);
mr_pairs = matchFeatures(middle_features, right_features, 'Unique', true);
left_match_points = left_points(lm_pairs(:,1),:);
lm_match_points = middle_points(lm_pairs(:,2),:);
mr_match_points = middle_points(mr_pairs(:,1),:);
right_match_points = right_points(mr_pairs(:,2),:);
figure;
showMatchedFeatures(left_y, middle_y, left_match_points, ...
    lm_match_points, 'montage');
figure;
showMatchedFeatures(middle_y, right_y, mr_match_points, ...
    right_match_points, 'montage');
% canvas
canvas = build_canvas(left, middle, right, 200);
canvas_y = rgb_yuv(canvas); canvas_y = canvas_y(:,:,1);
% homography
[best_lm_homography, ~] = RANSAC(canvas_y, ...
    [200,200], [200,776], left_match_points.Location([53, 7, 92, 30],:), ...
    lm_match_points.Location([53, 7, 92, 30],:), 1);
figure;
showMatchedFeatures(left_y, middle_y, left_match_points([53, 7, 92, 30]), ...
    lm_match_points([53, 7, 92, 30]), 'montage');
[best_mr_homography, ~] = RANSAC(canvas_y, ...
    [200,776], [200,1352], mr_match_points.Location([59, 10, 66, 1],:), ...
    right_match_points.Location([59, 10, 66, 1],:), 1);
figure;
showMatchedFeatures(middle_y, right_y, mr_match_points([59, 10, 66, 1]), ...
    right_match_points([59, 10, 66, 1]), 'montage');
% stitch
stitched_canvas = stitch_canvas(canvas, best_lm_homography, best_mr_homography, ...
    964, 1164);
figure;
imshow(stitched_canvas);

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

function homography = solve_homography(source, destination)
    assert(all(size(source)==[4,2]) && all(size(destination)==[4,2]));
    syms h11 h12 h13 h21 h22 h23 h31 h32
    equations = [];
    for i = 1:4
        xs = source(i,1);
        ys = source(i,2);
        xd = destination(i,1);
        yd = destination(i,2);
        eqn1 = h11*xs + h12*ys + h13 == (h31*xs + h32*ys + 1)*xd;
        eqn2 = h21*xs + h22*ys + h23 == (h31*xs + h32*ys + 1)*yd;
        equations = [equations, eqn1, eqn2];
    end
    [A,B] = equationsToMatrix(equations, ...
        [h11 h12 h13 h21 h22 h23 h31 h32]);
    solution = linsolve(A,B);
    homography = reshape([solution;1],[3,3])';
end

function new_coord = homogeneous_transform(homography, source)
    assert(all(size(homography)==[3,3]) && size(source,2)==2);
    npoints = size(source,1);
    padded_source = [source, ones(npoints, 1)];
    padded_source = padded_source';
    num_homography = double(homography);
    destination = num_homography * padded_source;
    new_coord = zeros(npoints,2);
    for i=1:npoints
        w = destination(3,i);
        new_coord(i,1) = destination(1,i) / w;
        new_coord(i,2) = destination(2,i) / w;
    end
end

function err = homography_error(homography, img, source_match_points)
    rounded_source_points = round(source_match_points);
    destination_points = homogeneous_transform(homography, source_match_points);
    rounded_destination_points = round(destination_points);
    source_pixel_values = img(rounded_source_points(:,1),...
        rounded_source_points(:,2));
    try
        destination_pixel_values = img(rounded_destination_points(:,1),...
            rounded_destination_points(:,2));
    catch
        err = Inf;
        return
    end
    err = mse(source_pixel_values, destination_pixel_values);
end

function canvas = build_canvas(left, middle, right, padding)
    left_nrows = size(left,1); left_ncols = size(left,2);
    middle_nrows = size(middle,1); middle_ncols = size(middle,2);
    right_nrows = size(right,1); right_ncols = size(right,2);
    width = left_ncols+middle_ncols+right_ncols;
    height = max([left_nrows,middle_nrows,right_nrows]);
    canvas = zeros(height+2*padding,width+2*padding,3, 'uint8');
    canvas((padding+1):(left_nrows+padding),...
        (padding+1):(left_ncols+padding),...
        :) = left(:,:,:);
    canvas((padding+1):(middle_nrows+padding),...
        (left_ncols+padding+1):(left_ncols+middle_ncols+padding),...
        :) = middle(:,:,:);
    canvas((padding+1):(right_nrows+padding),...
        (left_ncols+middle_ncols+padding+1):(left_ncols+middle_ncols+right_ncols+padding),...
        :) = right(:,:,:);
end

function [best_homography, best_ctrl_pts] = RANSAC(canvas, ...
    img1_pads, img2_pads, match_points1, match_points2, num_iter)
    % x => col; y => row
    padded_match_points1 = [match_points1(:,2)+img1_pads(1), ...
        match_points1(:,1)+img1_pads(2)];
    padded_match_points2 = [match_points2(:,2)+img2_pads(1), ...
        match_points2(:,1)+img2_pads(2)];
    num_pts = size(match_points1, 1);

    lowest_err = Inf;
    for i=1:num_iter
        % non-repeating control points
        ctrl_pt_idx = randperm(num_pts, 4);
        source = padded_match_points1(ctrl_pt_idx,:);
        destination = padded_match_points2(ctrl_pt_idx,:);
        
        homography = solve_homography(source, destination);
        err = homography_error(homography, canvas, padded_match_points1);
        if err < lowest_err
            best_homography = homography;
            best_ctrl_pts = ctrl_pt_idx;
            lowest_err = err;
        end
    end
end

function stitched_img = stitch_canvas(canvas, lm_homography, mr_homography, ...
    cutoff1, cutoff2)
    nrows = size(canvas,1); ncols = size(canvas,2);
    stitched_img = zeros(nrows, ncols, 3, 'uint8');

    lm_homography_inv = inv(lm_homography);
%     mr_homography_inv = inv(mr_homography);

    left_destination_coords = zeros(nrows*cutoff1,2);
    for r=1:nrows
        for c=1:cutoff1
            left_destination_coords((r-1)*cutoff1+c,:) = [r,c];
        end
    end
    left_source_coords = homogeneous_transform(lm_homography_inv, left_destination_coords);
    left_source_coords = round(left_source_coords);

    right_destination_coords = zeros(nrows*(ncols-cutoff2),2);
    for r=1:nrows
        for c=(cutoff2+1):ncols
            right_destination_coords((r-1)*(ncols-cutoff2)+c-cutoff2,:) = [r,c];
        end
    end
    right_source_coords = homogeneous_transform(mr_homography, right_destination_coords);
    right_source_coords = round(right_source_coords);

    for r=1:nrows
        for c=1:ncols
            if c <= cutoff1
                source_r = left_source_coords((r-1)*cutoff1+c, 1);
                source_c = left_source_coords((r-1)*cutoff1+c, 2);
                if source_c <= 0 || source_c > ncols || source_r <= 0 || source_r > nrows
                    stitched_img(r,c,1) = 0;
                    stitched_img(r,c,2) = 0;
                    stitched_img(r,c,3) = 0;
                else
                    stitched_img(r,c,1) = canvas(source_r,source_c,1);
                    stitched_img(r,c,2) = canvas(source_r,source_c,2);
                    stitched_img(r,c,3) = canvas(source_r,source_c,3);
                end
            elseif cutoff1 < c && c <= cutoff2
                stitched_img(r,c,1) = canvas(r,c,1);
                stitched_img(r,c,2) = canvas(r,c,2);
                stitched_img(r,c,3) = canvas(r,c,3);
            else
                source_r = right_source_coords((r-1)*(ncols-cutoff2)+c-cutoff2, 1);
                source_c = right_source_coords((r-1)*(ncols-cutoff2)+c-cutoff2, 2);
                if source_c <= 0 || source_c > ncols || source_r <= 0 || source_r > nrows
                    stitched_img(r,c,1) = 0;
                    stitched_img(r,c,2) = 0;
                    stitched_img(r,c,3) = 0;
                else
                    stitched_img(r,c,1) = canvas(source_r,source_c,1);
                    stitched_img(r,c,2) = canvas(source_r,source_c,2);
                    stitched_img(r,c,3) = canvas(source_r,source_c,3);
                end
            end
        end
    end
end

function idx = find_float_idx(nby2, x, y, accuracy)
    if nargin == 3
        accuracy = 1e-3;
    end
    
    idx = find(abs(nby2(:,1)-x) < accuracy & abs(nby2(:,2)-y) < accuracy);
end