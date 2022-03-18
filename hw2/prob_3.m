% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Feb 20

%% 3(a)
bird = read_img_rgb('HW2_images/bird.raw', 375, 500);
FS = [[0,0,0];[0,0,7];[3,5,1]]/16;
bird_cmy = rgb_cmy(bird);
FS_bird_c = diffuse(bird_cmy(:,:,1), FS);
FS_bird_m = diffuse(bird_cmy(:,:,2), FS);
FS_bird_y = diffuse(bird_cmy(:,:,3), FS);
bird_rgb = rgb_cmy(cat(3, FS_bird_c, FS_bird_m, FS_bird_y));
figure
imshow(bird_rgb)

%% 3(b)
bird_mbvq = MBVQ_diffuse(bird, FS);
figure
imshow(bird_mbvq)

%% 3(a) helper
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

function new_img = rgb_cmy(img)
    nrows=size(img,1); ncols=size(img,2);
    new_img = zeros(nrows,ncols,3);
    for i=1:nrows
        for j=1:ncols
            new_img(i,j,1) = 255 - img(i,j,1);
            new_img(i,j,2) = 255 - img(i,j,2);
            new_img(i,j,3) = 255 - img(i,j,3);
        end
    end
end

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

%% 3(b) helper
function vertex = get_quad_nearest_vertex(R, G, B)
    if (R+G) > 255
        if (G+B) > 255
            if (R+G+B) > 510
                quad = 'CMYW';
            else
                quad = 'MYGC';
            end
        else
            quad = 'RGMY';
        end
    else
        if (G+B) <= 255
            if (R+G+B) <= 255
                quad = 'KRGB';
            else
                quad = 'RGBM';
            end
        else
            quad = 'CMGB';
        end
    end
    
    vertex = getNearestVertex(quad, R, G, B);
end

function rgb = v_rgb(vertex)
    switch vertex
        case 'red'
            rgb = [255,0,0];
        case 'green'
            rgb = [0,255,0];
        case 'blue'
            rgb = [0,0,255];
        case 'yellow'
            rgb = [255,255,0];
        case 'cyan'
            rgb = [0,255,255];
        case 'magenta'
            rgb = [255,0,255];
        case 'white'
            rgb = [255,255,255];
        case 'black'
            rgb = [0,0,0];
    end
end

function new_img = MBVQ_diffuse(old_img, dif_mat)
    img = double(old_img);
    nrows=size(img,1); ncols=size(img,2);
    sz = (length(dif_mat)-1)/2;
    new_img = zeros(nrows, ncols, 3);
    new_img = uint8(new_img);
    
    for i=1:nrows
        if mod(i,2) == 1
            for j=1:ncols
                v = get_quad_nearest_vertex(img(i,j,1),img(i,j,2),img(i,j,3));
                rgb = v_rgb(v);
                new_img(i,j,1) = rgb(1);
                r_error = img(i,j,1) - rgb(1);
                new_img(i,j,2) = rgb(2);
                g_error = img(i,j,2) - rgb(2);
                new_img(i,j,3) = rgb(3);
                b_error = img(i,j,3) - rgb(3);
                
                for m=(i-sz):(i+sz)
                    for n=(j-sz):(j+sz)
                        if (m >= 1) && (m <= nrows) && (n >= 1) && (n <= ncols)
                            img(m,n,1) = img(m,n,1) + r_error*dif_mat(m-i+sz+1,n-j+sz+1);
                            img(m,n,2) = img(m,n,2) + g_error*dif_mat(m-i+sz+1,n-j+sz+1);
                            img(m,n,3) = img(m,n,3) + b_error*dif_mat(m-i+sz+1,n-j+sz+1);
                        end
                    end
                end
            end
        else
            for j=ncols:-1:1
                v = get_quad_nearest_vertex(img(i,j,1),img(i,j,2),img(i,j,3));
                rgb = v_rgb(v);
                new_img(i,j,1) = rgb(1);
                r_error = img(i,j,1) - rgb(1);
                new_img(i,j,2) = rgb(2);
                g_error = img(i,j,2) - rgb(2);
                new_img(i,j,3) = rgb(3);
                b_error = img(i,j,3) - rgb(3);
                
                for m=(i-sz):(i+sz)
                    for n=(j-sz):(j+sz)
                        if (m >= 1) && (m <= nrows) && (n >= 1) && (n <= ncols)
                            img(m,n,1) = img(m,n,1) + r_error*dif_mat(m-i+sz+1,end-(n-j+sz));
                            img(m,n,2) = img(m,n,2) + g_error*dif_mat(m-i+sz+1,end-(n-j+sz));
                            img(m,n,3) = img(m,n,3) + b_error*dif_mat(m-i+sz+1,end-(n-j+sz));
                        end
                    end
                end
            end
        end
    end
end