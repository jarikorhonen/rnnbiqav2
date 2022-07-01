% --------------------------------------------------------------
%   processLiveChallenge.m
%
%   This function processes LIVE Image Quality Challenge
%   database to produce the training images and obtain the
%   approximated probabilistic representations for the quality
%   scores.
%
%   Usage: processLiveChallenge(path)
%   Inputs: 
%       path: string with the path to LIVE Challenge database
%   Outuput: dummy
%
%   Note that you need to change the paths in this file to the 
%   actual folder where the database is located.
%   

function res = processLiveChallenge() %path, out_path)

out_path = 'j:\\train_images_bilat';
path = 'e:\\livechallenge';

    % First, load the MOS scores and stadard deviations
    % Make sure to change the paths to point to the database
    load([path '\\data\\allmos_release.mat']);
    load([path '\\data\\allstddev_release.mat']);

    % Use truncated Gaussian distribution to approximate 5-point
    % probabilistic representation of the quality scores
    bins = [];
    rng(666);
    for i=8:length(AllMOS_release)
        stdthis = AllStdDev_release(i); %+rand(1)*2-1;
        for j=1:52
            stdthis = AllStdDev_release(i); %+rand(1)*2-1;
            b1 = truncGaussianCDF(20,AllMOS_release(i), stdthis,0,100);
            b2 = truncGaussianCDF(40,AllMOS_release(i), stdthis,0,100)-b1;
            b3 = truncGaussianCDF(60,AllMOS_release(i), stdthis,0,100)-b2-b1;
            b4 = truncGaussianCDF(80,AllMOS_release(i), stdthis,0,100)-b3-b2-b1;
            b5 = 1-b4-b3-b2-b1;
            bins = [bins; b1 b2 b3 b4 b5];
        end
    end

    % Save the results
    LiveC_prob = bins;    
    LiveC_std = [];

    bins_none = [0.2 0.2 0.2 0.2 0.2];


    % Load image names in Live Challenge database
    load([path '\\data\\allImages_release.mat']);
    patch_size = [224 224];
    patch_idx = 1;

    % Loop through all the test images (skip the first seven for training
    for im_no=8:length(AllImages_release)

        imfile = sprintf('%s\\Images\\%s', path, AllImages_release{im_no});
        num_patch = 1;
        
        % Initialize variables
        im = imread(imfile);
        im = imbilatfilt(im,25,3);
        [height,width,~] = size(im);
        x_numb = ceil(width/patch_size(1));
        y_numb = ceil(height/patch_size(2));
        x_step = 1;
        y_step = 1;
        if x_numb>1 && y_numb>1
            x_step = floor((width-patch_size(1))/(x_numb-1));
            y_step = floor((height-patch_size(2))/(y_numb-1));
        end
        im_patches = [];
%         num_patch = 1;
%         x_step = floor(x_step*2/3);
%         y_step = floor(y_step*2/3);

        idx = 1;

        % Extract patches from the image
        for i=1:x_step:width-patch_size(1)+1
            for j=1:y_step:height-patch_size(2)+1
                if idx<10
                    y_range = j:j+patch_size(2)-1;
                    x_range = i:i+patch_size(1)-1;
                    im_patch = im(y_range, x_range,:);
                    % Make four rotated versions of each patch
                    for q=1:4
                        filename = sprintf('%s\\%04d_%02d.png', ...
                                           out_path, im_no-7, num_patch);               
                                       
                        imwrite(im_patch,filename);   
                        std_this = mean([std2(im_patch(:,:,1)) std2(im_patch(:,:,2)) std2(im_patch(:,:,3))]);
%                         LiveC_std = [LiveC_std; std_this];
                        if std_this < 15
                            alfa = ((20-std_this)/20).^2;
                            LiveC_prob(patch_idx,:) = LiveC_prob(patch_idx,:).*(1-alfa) + bins_none.*alfa;
                        end

                        im_patch = imrotate(im_patch,90);
                        num_patch = num_patch + 1;
                        patch_idx = patch_idx + 1;
                    end
                end
                idx = idx + 1;
            end
        end 
        
        rng_x1 = [181,181,303,303];
        rng_x2 = [404,404,526,526];
        rng_y1 = [181,303,181,303];
        rng_y2 = [404,526,404,526];
        for j=0:3
            im2 = imrotate(im,j*90+45,'bilinear');
            for i=1:4
                im_patch = im2(rng_x1(i):rng_x2(i),rng_y1(i):rng_y2(i),:);
                filename = sprintf('%s\\%04d_%02d.png', ...
                                   out_path, im_no-7, num_patch);  
                imwrite(im_patch,filename); 
                std_this = mean([std2(im_patch(:,:,1)) std2(im_patch(:,:,2)) std2(im_patch(:,:,3))]);
%                 LiveC_std = [LiveC_std; std_this];
                if std_this < 15
                     alfa = ((20-std_this)/20).^2;
                     LiveC_prob(patch_idx,:) = LiveC_prob(patch_idx,:).*(1-alfa) + bins_none.*alfa;
                end
                num_patch = num_patch + 1;
                patch_idx = patch_idx + 1;
            end
        end
    end
%     save([out_path '\\LiveC_std2.mat'],'LiveC_std');
    save([out_path '\\LiveC_prob2.mat'],'LiveC_prob');
    res = 0;   
end

% Truncated Gaussian cumulative distribution function
function X = truncGaussianCDF(x,my,sigma,a,b)

    if x<=a 
        X=0;
    elseif x>=b
        X=1;
    else
        X = (Th(my,sigma,x)-Th(my,sigma,a))/ ...
            (Th(my,sigma,b)-Th(my,sigma,a));
    end
end

% Theta function for computing truncated Gaussian cdf
function X = Th(my,sigma,x)
    X = (1+erf((x-my)/(sigma*sqrt(2))))/2;
end