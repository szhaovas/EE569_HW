%%
% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Mar 10

%% 3(a)
global thinning_conditional ST_unconditional;
thinning_conditional = {
    % bond 4
    {'010011000','010110000','000110010','000011010','001011001','111010000','100110100','000010111'};
    % bond 5
    {'110011000','010011001','011110000','001011010','011011000','110100000','000110110','000011011'};
    % bond 6
    {'110011001','011110100','111011000','011011001','111110000','110110100','100110110','000110111','000011111','001011011'};
    % bond 7
    {'111011001','111110100','100110111','001011111'};
    % bond 8
    {'011011011','111111000','110110110','000111111'};
    % bond 9
    {'111011011','011011111','111111100','111111001','111110110','110110111','100111111','001111111'};
    % bond 10
    {'111011111','111111101','111110111','101111111'}
};

ST_unconditional = {
    '00M0M0000','M000M0000',...
    '0000M00M0','0000MM000',...
    '00M0MM000','0MM0M0000','MM00M0000','M00MM0000','000MM0M00','0000M0MM0','0000M00MM','0000MM00M',...
    '0MMMM0000','MM00MM000','0M00MM00M','00M0MM0M0',...
    '0AM0MBM00','MB0AM000M','00MAM0MB0','M000MB0AM',...
    'MMDMMDDDD',...
    'DM0MMMD00','0MDMMM00D','00DMMM0MD','D00MMMDM0','DMDMM00M0','0M0MM0DMD','0M00MMDMD','DMD0MM0M0',...
    'MDMDMDABC','MDCDMBMDA','CBADMDMDM','ADMBMDCDM',...
    'DM00MMM0D','0MDMM0D0M','D0MMM00MD','M0D0MMDM0'
};

flower = read_img_gray('raw_images/flower.raw', 247, 247);
flower = (flower ~= 0);
figure; subplot(1,3,1);
imshow(flower);
title('original');
num_change = 1;
iter_counter = 0;
while num_change ~= 0
    if iter_counter == 20
        subplot(1,3,2);
        imshow(flower);
        title('20th thinning');
    end
    [flower, num_change] = thinning(flower);
    iter_counter = iter_counter + 1;
end
subplot(1,3,3);
imshow(flower);
title('complete thinning');

%%
jar = read_img_gray('raw_images/jar.raw', 252, 252);
jar = (jar ~= 0);
figure; subplot(1,3,1);
imshow(jar);
title('original');
num_change = 1;
iter_counter = 0;
while num_change ~= 0
    if iter_counter == 20
        subplot(1,3,2);
        imshow(jar);
        title('20th thinning');
    end
    [jar, num_change] = thinning(jar);
    iter_counter = iter_counter + 1;
end
subplot(1,3,3);
imshow(jar);
title('complete thinning');

%%
spring = read_img_gray('raw_images/spring.raw', 252, 252);
spring = (spring ~= 0);
figure; subplot(1,3,1);
imshow(spring);
title('original');
num_change = 1;
iter_counter = 0;
while num_change ~= 0
    if iter_counter == 20
        subplot(1,3,2);
        imshow(spring);
        title('20th thinning');
    end
    [spring, num_change] = thinning(spring);
    iter_counter = iter_counter + 1;
end
subplot(1,3,3);
imshow(spring);
title('complete thinning');

%% 3(b)
deer = read_img_gray('raw_images/deer.raw', 691, 550);
deer = (deer ~= 0);
figure;
imshow(deer);
hold on;
deer_shrink = bwmorph((deer == 0), 'shrink', Inf);
[defect_r, defect_c] = find(deer_shrink);
for i=1:length(defect_r)
    plot(defect_c(i),defect_r(i),'r+', 'MarkerSize', 50);
end
hold off;
% defect_coords = [[281,94];[339,191];[285,276];[353,332];[336,335];[208,499]];
% px_counts = zeros(size(defect_coords,1),1);
% for i=1:size(defect_coords,1)
%    [deer, count] = fix_defect(deer, 0, defect_coords(i,:));
%    px_counts(i) = count;
% end
global counter;
defect_coords = [defect_r, defect_c];
px_counts = zeros(size(defect_coords,1),1);
for i=1:size(defect_coords,1)
    counter = 0;
   deer = fix_defect(deer, defect_coords(i,:), 50);
   px_counts(i) = counter;
end
figure;
imshow(deer);

%% 3(c)
% (1)
beans = read_img_rgb('raw_images/beans.raw', 82, 494);
% beans = black_background(beans);
beans = binarize_rgb(beans, 230);
figure;
imshow(beans);
se = strel('cube',3);
beans = imerode(beans, se);
figure;
imshow(beans);
beans_shrink = bwmorph(beans, 'shrink', Inf);
figure;
imshow(beans_shrink);
num_beans = sum(sum(beans_shrink))
% (2)
[bean_r, bean_c] = find(beans_shrink);
bean_coords = [bean_r, bean_c];
size_counts = zeros(size(bean_coords,1),1);
for i=1:size(bean_coords,1)
    counter = 0;
   beans_copy = fix_defect((beans == 0), bean_coords(i,:), Inf);
   size_counts(i) = counter;
end
size_counts

%% 3(a) helper
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

function padded_img = zero_pad(img_matrix, pad_width)
    [nrows, ncols] = size(img_matrix);
    padded_img = zeros(nrows+pad_width*2, ncols+pad_width*2);
    
    padded_img((pad_width+1):(pad_width+nrows),(pad_width+1):(pad_width+ncols))...
        = img_matrix(:,:);
end

function hit = match_pattern(patch, unconditional)
    assert((size(patch,1)==3) && (size(patch,2)==3))
    global thinning_conditional ST_unconditional;
    if nargin == 1
        unconditional = false;
    end
    
    code_str = arrayfun(@(i) int2str(i), reshape(patch.',1,[]));
    hit = false;
    if unconditional
        for p = ST_unconditional
            pat = p{1};
            match = true;
            any_one_check = false;
            any_one_pass = false;
            for i=1:9
                switch pat(i)
                    case 'D'
                        continue
                    case {'A','B','C'}
                        any_one_check = true;
                        if code_str(i) == '1'
                            any_one_pass = true;
                        end
                    otherwise
                        if pat(i) ~= code_str(i)
                            match = false;
                            break
                        end
                end
            end
            if match && (~any_one_check || any_one_pass)
                hit = true;
                return
            end
        end
    else
        bond = check_bond(patch);
        try
            patterns = thinning_conditional{bond-3};
        catch
            hit = false;
            return
        end
        for p = patterns
            pat = p{1};
            if strcmp(code_str, pat)
                hit = true;
                return
            end
        end
    end
end

function bond = check_bond(patch)
    assert((size(patch,1)==3) && (size(patch,2)==3))
    bond = 0;
    for r=1:3
        for c=1:3
            if (patch(r,c)==0) || ((r==2)&&(c==2))
                continue
            else
                if mod(r,2) == 0
                    bond = bond + 2;
                else
                    if mod(c,2) == 0
                        bond = bond + 2;
                    else
                        bond = bond + 1;
                    end
                end
            end
        end
    end
end

function [new_img, num_change] = thinning(img)
    [nrows, ncols] = size(img);
    padded_img = zero_pad(img, 1);
    new_img = zeros(nrows, ncols, 'logical');
    num_change = 0;
    
    padded_mask = zeros(nrows+2, ncols+2, 'uint8');
    for r=1:nrows
        for c=1:ncols
            if padded_img(r+1,c+1) == 1
                patch = padded_img(r:r+2,c:c+2);
                hit = match_pattern(patch);
                if hit
                    padded_mask(r+1,c+1) = 1;
                end
            end
        end
    end
    
    for r=1:nrows
        for c=1:ncols
            if padded_mask(r+1,c+1) == 1
                patch = padded_img(r:r+2,c:c+2);
                hit = match_pattern(patch, true);
                if hit
                    new_img(r,c) = img(r,c);
                else
                    new_img(r,c) = 0;
                    num_change = num_change + 1;
                end
            else
                new_img(r,c) = img(r,c);
            end
        end
    end
end

% thinning_conditional = {
%     % TK 4
%     ['010011000','010110000','000110010','000011010'];
%     % STK 4
%     ['001011001','111010000','100110100','000010111'];
%     % ST 5-1
%     ['110011000','010011001','011110000','001011010'];
%     % ST 5-2
%     ['011011000','110100000','000110110','000011011'];
%     % ST 6
%     ['110011001','011110100'];
%     % STK 6
%     ['111011000','011011001','111110000','110110100','100110110','000110111','000011111','001011011'];
%     % STK 7
%     ['111011001','111110100','100110111','001011111'];
%     % STK 8
%     ['011011011','111111000','110110110','000111111'];
%     % STK 9
%     ['111011011','011011111','111111100','111111001','111110110','110110111','100111111','001111111'];
%     % STK 10
%     ['111011111','111111101','111110111','101111111']
% };

% ST_unconditional = {
%     % Spur
%     ['001010000','100010000'];
%     % Single 4-connection
%     ['000010010','000011000'];
%     % L Cluster
%     ['001011000','011010000','110010000','100110000','000110100','000010110','000010011','000011001'];
%     % 4-connected Offset
%     ['011110000','110011000','010011001','001011010'];
%     % Spur corner Cluster
%     ['0A101B100','1B0A10001','001A101B0','10001B0A1'];
%     % Corner Cluster
%     ['11D11DDDD'];
%     % Tee Branch
%     ['D10111D00','01D11100D','00D11101D','D00111D10','D1D110010','010110D1D','010011D1D','D1D011010'];
%     % Vee Branch
%     ['1D1D1DABC','1DCD1B1DA','CBAD1D1D1','AD1B1DCD1'];
%     % Diagonal Branch
%     ['D1001110D','01D110D01','D0111001D','10D011D10'];
% };

%% 3(b) helper
function img = fix_defect(img, coords, threshold)
    global counter;
    [nrows, ncols] = size(img);
    
    if img(coords(1), coords(2)) == 0
        img(coords(1), coords(2)) = 1;
        counter = counter + 1;
    end
    
    if counter <= threshold && 0 < coords(1)-1 && img(coords(1)-1, coords(2)) == 0
        img = fix_defect(img, [coords(1)-1, coords(2)], threshold);
    end
    
    if counter <= threshold && 0 < coords(2)-1 && img(coords(1), coords(2)-1) == 0
        img = fix_defect(img, [coords(1), coords(2)-1], threshold);
    end
    
    if counter <= threshold && coords(1)+1 <= nrows && img(coords(1)+1, coords(2)) == 0
        img = fix_defect(img, [coords(1)+1, coords(2)], threshold);
    end
    
    if counter <= threshold && coords(2)+1 <= ncols && img(coords(1), coords(2)+1) == 0
        img = fix_defect(img, [coords(1), coords(2)+1], threshold);
    end
    
    if counter > threshold
        img(coords(1), coords(2)) = 0;
    end
end

%% 3(c) helper
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

function new_img = black_background(img)
    nrows = size(img,1); ncols = size(img,2);
    new_img = zeros(nrows, ncols, 3, 'uint8');
    
    for r=1:nrows
        for c=1:ncols
            if all(img(r,c,:) > 240)
                new_img(r,c,1) = 0;
                new_img(r,c,2) = 0;
                new_img(r,c,3) = 0;
            else
                new_img(r,c,1) = img(r,c,1);
                new_img(r,c,2) = img(r,c,2);
                new_img(r,c,3) = img(r,c,3);
            end
        end
    end
end

function bin_img = binarize_rgb(img, y_threshold)
    img_y = rgb_yuv(img); img_y = img_y(:,:,1);
    
    nrows = size(img,1); ncols = size(img,2);
    bin_img = zeros(nrows, ncols, 'logical');
    for r=1:nrows
        for c=1:ncols
            if img_y(r,c) > y_threshold
                bin_img(r,c) = 0;
            else
                bin_img(r,c) = 1;
            end
        end
    end
end