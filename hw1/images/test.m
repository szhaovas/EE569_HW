function test()
    I = read_img_rgb('House_ori.raw', 512, 768)
    imshow(I);
end

function img_matrix = read_img_grey(file, row, col)
    f = fopen(file,'r');
    img_matrix = fread(f, [col row],'uint8=>uint8'); 
    fclose(f);
end

function img_matrix = read_img_rgb(file, row, col)
    f = fopen(file,'r');
    img_stream = fread(f, col*row*3,'uint8=>uint8');
    img_matrix = reshape(img_stream, [col row 3]);
    fclose(f);
end

% function bilinear_demosaic(img_matrix)
