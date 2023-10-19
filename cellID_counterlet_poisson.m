clear all;
close all;
clc

numframes=91;

frames = cell(1,numframes);
frames_pre_seg = cell(1,numframes);
frames_post_seg = cell(1,numframes);

for i = 1:numframes
    frames{i} = imread(sprintf('home/j/matlab_main/data_set/challenge/Fluo-N2DH-GOWT1/01/t%03d.tif',i-1));
end



%% Load the image
for i = 1:numframes
I = im2double(frames{i});
%increase gamma 
I = imadjust(I, [], [], 0.25);
%figure;imshow(I);
I = imnoise(I,'gaussian',0,0.03);
scale = 1e10;
I = scale*imnoise(I/scale,'poisson');
%figure;imshow(I);


%% Gaussian Denoise
Gaussian_Denoised_I = imgaussfilt(I,9);
%imshow(Gaussian_Denoised_I);

%% Poisson Denoise
%Copyright (c) 2011, SANDEEP PALAKKAL
%All rights reserved.

mx = 10; % Maximum intensity of the true image
mn = 0.9; % Minimum intensity of the true image

[z im] = poisson_count( Gaussian_Denoised_I, mn, mx );

J = 5; % No. of wavelet scales
let_id = 2; %PURE-LET 0, 1, or 2.
nSpin = 5; % No. of cycle spins.
y = cspin_purelet(z,let_id,J,nSpin);

%% reformat & brighten 
K = uint8(y);
%figure;
K = imadjust(K);
%imshow(K,[]);

%% Implement Contourlet Transform
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Yue M. Lu and Minh N. Do
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%	Demo_denoising.m
%	
%   First Created: 09-02-05
%	Last Revision: 07-13-09
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Denoising using the ContourletSD transform by simple hardthreholding

dfilt = 'pkva12';
nlev_SD = [2 2 3 4 5];
smooth_func = @rcos;

% Pyr_mode can take the value of 1, 1.5 or 2. It specifies which one of
% the three variants of ContourletSD to use. The main differences are the
% redundancies of the transforms, which are 2.33, 1.59, or 1.33,
% respectively.

Pyr_mode = 1; 
% redundancy = 2. Set Pyr_mode = 2 for a less redundant version of the
% transform.

X = K;
X = double(X);

sigma = 30; % noise intensity

% Get the noisy image
Xn = X;

% load pre-computed scaling factor for 1024 * 1024
eval(['load SDmm_' num2str(size(Xn,1)) '_' num2str(Pyr_mode)]);

% take counterlet transform
Y = ContourletSDDec(Xn, nlev_SD, Pyr_mode, smooth_func, dfilt);

% Apply hard thresholding on coefficients
Yth = Y;
for m = 2:length(Y)
  thresh = 3*sigma + sigma * (m == length(Y));
  for k = 1:length(Y{m})
    Yth{m}{k} = Y{m}{k}.* (abs(Y{m}{k}) > thresh*E{m}{k});
  end
end

% ContourletSD reconstruction
Xd = ContourletSDRec(Yth, Pyr_mode, smooth_func, dfilt);

L = imadjust(Xd/255);
%figure;imagesc(L);colormap(gray);
%imshow(L);
%title(['Denoising using the contourletSD transform. ']);


%% Smooth Image

% used irfranview to verify ~75pixel radius on biggest cells. Filtered down
% later using area
background = imopen(L,strel('disk',75));
% gaussian smoothing
smoothed_background = imgaussfilt(background,8);
% foreground
I2 = L - smoothed_background;
frames_pre_seg{i}=I2;
end

% Set segmentation parameters (adjust as needed)
minArea = 750;  % Minimum area to consider as a cell
maxArea = 11000;  % Maximum area to consider as a cell

% Preallocate a cell array to store cell information for each frame
cellData = cell(1, numframes);

for i = 1:numframes
    % Segment the cells using your preferred segmentation algorithm
    % Replace the following line with your own segmentation code
    binaryImage = imbinarize(frames_pre_seg{i});
    
    % Clean up the binary image (optional)
    bw = bwareaopen(binaryImage, minArea);
    
    bw = imfill(bw,'holes');


    C =~bw;
    D = -bwdist(C);
    D(C) = -Inf;
    L1 = watershed(D);
    Wi = label2rgb(L1);
%% Use this for Watershedding segmentation
    frames_post_seg{i} = Wi;
    %stats = regionprops(L1, 'Centroid', 'Area', 'BoundingBox');

%% Use this for default
    %frames_post_seg{i}=bw;
    stats = regionprops(L1, 'Centroid', 'Area', 'BoundingBox');

    % Measure properties of the connected components

    stats = regionprops(L1, 'Centroid', 'Area', 'BoundingBox');
    
    % Filter out cells based on area
    validCells = [stats.Area] >= minArea & [stats.Area] <= maxArea;
    stats = stats(validCells);
    
    % Store the centroid, size, and other relevant information
    cellData{i} = struct('Centroid', {stats.Centroid}, 'Size', {stats.Area}, 'BoundingBox', {stats.BoundingBox});
end

% Preallocate cell array to store linked indices
linkIdx = cell(1, numframes);

% Threshold for linking cells between frames (adjust as needed)
threshold = 100;

% Find pairs of cells to connect
for i = 1:numframes-1
    currentFrame = cellData{i};
    nextFrame = cellData{i + 1};
    
    % Calculate pairwise distances using centroids
    distances = zeros(numel(currentFrame), numel(nextFrame));
    
    for j = 1:numel(currentFrame)
        centroidCurrent = currentFrame(j).Centroid;
        
        for k = 1:numel(nextFrame)
            centroidNext = nextFrame(k).Centroid;
            distances(j, k) = norm(centroidCurrent - centroidNext);
        end
    end
    
    % Update distance matrix
    distanceMatrix(1:size(distances, 1), i) = min(distances, [], 2);
end

% Plot tracks
figure;

for i = 1:numframes-1
    % Plot frame
    imshow(frames_post_seg{i});
    hold on;

    % Plot cell centroids and tracks
    currentFrame = cellData{i};
    for j = 1:numel(currentFrame)
        centroid = currentFrame(j).Centroid;
        plot(centroid(1), centroid(2), 'r.', 'MarkerSize', 10);

        if i < 10
            nextFrame = cellData{i + 1};
            linkedIdx = linkIdx{i};

            if ~isempty(linkedIdx)
                nextCentroid = nextFrame(linkedIdx(j)).Centroid;
                line([centroid(1), nextCentroid(1)], [centroid(2), nextCentroid(2)], 'Color', 'g');
            end
        end
    end
    
    hold off;
    frame_with_centroids = getframe(gcf);
    file_name = sprintf('/home/j/matlab_main/outputs/test4/image_%s_%03d.png', "gotw", i);  % Adjust the file name format as desired
    
    % Save the image with the generated file name
    saveas(gcf, file_name);
end


