% ZHAO SHIHAN
% 5927678670
% shihanzh@usc.edu
% Mar 10

%% 1(1)
forky = read_img_rgb('raw_images/Forky.raw', 328, 328);
forky_star = transform(forky);
figure
imshow(forky_star)

%%
twenty_two = read_img_rgb('raw_images/22.raw', 328, 328);
twenty_two_star = transform(twenty_two);
figure
imshow(twenty_two_star)

%% 1(2)
forky_reverse = transform(forky_star, true);
figure
imshow(forky_reverse)

%%
twenty_two_reverse = transform(twenty_two_star, true);
figure
imshow(twenty_two_reverse)

%% 1(1) helper
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

function solution = solve_transform(xy, uv)
    syms a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 b4 b5
    len = size(xy, 1);
    equations = [];
    for i = 1:len
        x = xy(i,1);
        y = xy(i,2);
        eqn_u = a0 + a1*x + a2*y + a3*(x^2) + a4*(x*y) + a5*(y^2) == uv(i,1);
        eqn_v = b0 + b1*x + b2*y + b3*(x^2) + b4*(x*y) + b5*(y^2) == uv(i,2);
        equations = [equations, eqn_u, eqn_v];
    end
    [A,B] = equationsToMatrix(equations, ...
        [a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 b4 b5]);
    solution = linsolve(A,B);
end

function new_img = transform(img, reverse)
    if nargin == 1
        reverse = false;
    end
    nrows = size(img,1); ncols = size(img,2);
    new_img = zeros(nrows, ncols, 3, 'uint8');
    
    % up
    xy_up = [[nrows/2,ncols/2];[nrows/4,ncols/4];[nrows/4,3*ncols/4];[1,1];[65,ncols/2];[1,ncols]];
    uv_up = [[nrows/2,ncols/2];[nrows/4,ncols/4];[nrows/4,3*ncols/4];[1,1];[1,ncols/2];[1,ncols]];
    if reverse
        coef_up = reshape(solve_transform(uv_up, xy_up),[6,2])';
    else
        coef_up = reshape(solve_transform(xy_up, uv_up),[6,2])';
    end
    num_coef_up = double(coef_up);
    for i=1:(nrows/2)
        for j=i:(ncols-i+1)
            uv = num_coef_up*[1,i,j,i^2,i*j,j^2]';
            round_u = round(uv(1));
            round_v = round(uv(2));
            if round_u > 0
                new_img(i,j,:) = img(round_u,round_v,:);
            end
        end
    end
    % right
    xy_right = [[nrows/2,ncols/2];[nrows/4,3*ncols/4];[3*nrows/4,3*ncols/4];[1,ncols];[nrows/2,ncols-64];[nrows,ncols]];
    uv_right = [[nrows/2,ncols/2];[nrows/4,3*ncols/4];[3*nrows/4,3*ncols/4];[1,ncols];[nrows/2,ncols];[nrows,ncols]];
    if reverse
        coef_right = reshape(solve_transform(uv_right, xy_right),[6,2])';
    else
        coef_right = reshape(solve_transform(xy_right, uv_right),[6,2])';
    end
    num_coef_right = double(coef_right);
    for j=(ncols/2):ncols
        for i=(nrows-j+1):j
            uv = num_coef_right*[1,i,j,i^2,i*j,j^2]';
            round_u = round(uv(1));
            round_v = round(uv(2));
            if round_v <= ncols 
                new_img(i,j,:) = img(round_u,round_v,:);
            end
        end
    end
    % down
     xy_down = [[nrows/2,ncols/2];[3*nrows/4,ncols/4];[3*nrows/4,3*ncols/4];[nrows,1];[nrows-64,ncols/2];[nrows,ncols]];
     uv_down = [[nrows/2,ncols/2];[3*nrows/4,ncols/4];[3*nrows/4,3*ncols/4];[nrows,1];[nrows,ncols/2];[nrows,ncols]];
     if reverse
        coef_down = reshape(solve_transform(uv_down, xy_down),[6,2])';
     else
        coef_down = reshape(solve_transform(xy_down, uv_down),[6,2])';
     end
     num_coef_down = double(coef_down);
     for i=(nrows/2):nrows
         for j=(ncols-i+1):i
             uv = num_coef_down*[1,i,j,i^2,i*j,j^2]';
             round_u = round(uv(1));
             round_v = round(uv(2));
             if round_u <= nrows
                 new_img(i,j,:) = img(round_u,round_v,:);
             end
         end
     end
    % left
    xy_left = [[nrows/2,ncols/2];[nrows/4,ncols/4];[3*nrows/4,ncols/4];[1,1];[nrows/2,65];[nrows,1]];
    uv_left = [[nrows/2,ncols/2];[nrows/4,ncols/4];[3*nrows/4,ncols/4];[1,1];[nrows/2,1];[nrows,1]];
    if reverse
        coef_left = reshape(solve_transform(uv_left, xy_left),[6,2])';
    else
        coef_left = reshape(solve_transform(xy_left, uv_left),[6,2])';
    end
    num_coef_left = double(coef_left);
    for j=1:(ncols/2)
        for i=j:(nrows-j+1)
            uv = num_coef_left*[1,i,j,i^2,i*j,j^2]';
            round_u = round(uv(1));
            round_v = round(uv(2));
            if round_v > 0
                new_img(i,j,:) = img(round_u,round_v,:);
            end
        end
    end
end