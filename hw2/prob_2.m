% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Feb 20

%% 2(a)
bridge = read_img_gray('HW2_images/bridge.raw', 400, 600);
fixed_thrs_bridge = threshold(bridge, 128);
figure
% Note: subplot scales the image down. I'm using subplot here because I
% thought the quality of these two images to be not as important. But if
% they don't look alright, please try zooming in or displaying each in its
% own figure.
% figure
subplot(1,2,1)
imshow(fixed_thrs_bridge)
title('fixed threshold 128')
rand_thrs_bridge = threshold(bridge);
% figure
subplot(1,2,2)
imshow(rand_thrs_bridge)
title('random threshold')
[dt2_thrs_bridge, dt2_thrs] = dithering_threshold(bridge, 2);
figure
imshow(dt2_thrs_bridge)
[dt8_thrs_bridge, dt8_thrs] = dithering_threshold(bridge, 8);
figure
imshow(dt8_thrs_bridge)
[dt32_thrs_bridge, dt32_thrs] = dithering_threshold(bridge, 32);
figure
imshow(dt32_thrs_bridge)

%% 2(b)
FS = [[0,0,0];[0,0,7];[3,5,1]]/16;
FS_dif_bridge = diffuse(bridge, FS);
figure
imshow(FS_dif_bridge)
JJN = [[0,0,0,0,0];
       [0,0,0,0,0];
       [0,0,0,7,5];
       [3,5,7,5,3];
       [1,3,5,3,1]]/48;
JJN_dif_bridge = diffuse(bridge, JJN);
figure
imshow(JJN_dif_bridge)
Stucki = [[0,0,0,0,0];
          [0,0,0,0,0];
          [0,0,0,8,4];
          [2,4,8,4,2];
          [1,2,4,2,1]]/42;
Stucki_dif_bridge = diffuse(bridge, Stucki);
figure
imshow(Stucki_dif_bridge)

%% 2(2) helper
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

function new_img = threshold(img, thrs)
    [nrows, ncols] = size(img);
    new_img = zeros(nrows, ncols);
    new_img = uint8(new_img);
    
    for i=1:nrows
        for j=1:ncols
            if nargin == 1
                thrs = randi([0 255]);
            end
            if img(i,j) >= thrs
                new_img(i,j) = 255;
            end
        end
    end
end

% since the instruction doesn't ask for writing a generalized ditering
% matrix generator, and simply asks for dithering matrices of constant
% sizes, which in principle we should be allowed to even calculate by hand.
% I though it would be fine to use the matlab builtin matrix manipulators here
function dt_mat = dithering_matrix(sz)
    if sz == 1
        dt_mat = 0;
    else
        dt_mat_rec = dithering_matrix(sz/2);
        dt_mat = [[4*dt_mat_rec+1, 4*dt_mat_rec+2];...
                  [4*dt_mat_rec+3, 4*dt_mat_rec]];
    end
end

function thrs_mat = dithering_thrs_matrix(dt_mat)
    [nrows, ncols] = size(dt_mat);
    thrs_mat = zeros(nrows, ncols);
    
    for i=1:nrows
        for j=1:ncols
            thrs_mat(i,j) = 255*(dt_mat(i,j)+0.5)/(nrows*ncols);
        end
    end
end

function [new_img, thrs_mat] = dithering_threshold(img, sz)
    [nrows, ncols] = size(img);
    dt_mat = dithering_matrix(sz);
    thrs_mat = dithering_thrs_matrix(dt_mat);
    new_img = zeros(nrows, ncols);
    new_img = uint8(new_img);
    
    for i=1:nrows
        for j=1:ncols
            if img(i,j) > thrs_mat(mod(i-1,sz)+1,mod(j-1,sz)+1)
                new_img(i,j) = 255;
            end
        end
    end
end

%% 2(b) helper
function new_img = diffuse(old_img, dif_mat)
    img = double(old_img);
    [nrows, ncols] = size(img);
    sz = (length(dif_mat)-1)/2;
    new_img = zeros(nrows, ncols);
    new_img = uint8(new_img);
    
    for i=1:nrows
        if mod(i,2) == 1
            for j=1:ncols
                if img(i,j) > 128
                    new_img(i,j) = 255;
                    error = img(i,j) - 255;
                else
                    error = img(i,j);
                end
                
                for m=(i-sz):(i+sz)
                    for n=(j-sz):(j+sz)
                        if (m >= 1) && (m <= nrows) && (n >= 1) && (n <= ncols)
                            img(m,n) = img(m,n) + error*dif_mat(m-i+sz+1,n-j+sz+1);
                        end
                    end
                end
            end
        else
            for j=ncols:-1:1
                if img(i,j) > 128
                    new_img(i,j) = 255;
                    error = img(i,j) - 255;
                else
                    error = img(i,j);
                end
                
                for m=(i-sz):(i+sz)
                    for n=(j-sz):(j+sz)
                        if (m >= 1) && (m <= nrows) && (n >= 1) && (n <= ncols)
                            img(m,n) = img(m,n) + error*dif_mat(m-i+sz+1,end-(n-j+sz));
                        end
                    end
                end
            end
        end
    end
end