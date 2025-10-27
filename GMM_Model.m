% Script for Hand Mask Generation using Gaussian Mixture Models (GMM)
% Phase 1: Train GMM on Cb, Cr skin pixels from the training set.
% Phase 2: Apply GMM to new images for prediction.

clear; close all; clc;

% --- Configuration ---
trainImageFolder = 'Training-Dataset\Training-Dataset\Images';   % .jpg
trainMaskFolder  = 'Training-Dataset\Training-Dataset\Masks-Ideal';    % .bmp
validationImageFolder = 'Validation-Dataset\Validation-Dataset\Images'; % For testing prediction
validationOutputFolder = 'Validation-Dataset\Validation-Dataset\Masks'; % Updated Output folder name

numGaussians = 10; % Number of Gaussian components (can often use fewer for 2D)
gmmModelFile = 'skinGMM_CbCr_model.mat'; % Updated model filename

% --- Morphological Kernels (Tune size/shape based on your data!) ---
se_open = strel('disk', 3);
se_close = strel('disk', 5);

% --- Probability Threshold for Prediction ---
probThreshold = 0.5; % Adjust this based on results (often lower needed for CbCr only)

% =========================================================================
% Phase 1: GMM Training (Run Once)
% =========================================================================
fprintf('--- Phase 1: Training Skin GMM (CbCr) ---\n');

% --- Get Training Files ---
trainImageFiles = dir(fullfile(trainImageFolder, '*.jpg'));
trainMaskFiles = dir(fullfile(trainMaskFolder, '*.bmp'));
trainImageNames = {trainImageFiles.name};
trainMaskNames = {trainMaskFiles.name};

% --- Match Training Files ---
[~, imageBasenames, ~] = cellfun(@fileparts, trainImageNames, 'UniformOutput', false);
[~, maskBasenames, ~] = cellfun(@fileparts, trainMaskNames, 'UniformOutput', false);
[commonNames, ia, ib] = intersect(imageBasenames, maskBasenames);

if isempty(commonNames)
    error('No matching image-mask pairs found for training!');
end
trainImageNames = trainImageNames(ia);
trainMaskNames = trainMaskNames(ib);
numTrainPairs = length(trainImageNames);
fprintf('Found %d training pairs.\n', numTrainPairs);

% --- Collect Skin Pixel Data (Cb, Cr Only) ---
maxTotalSkinPixels = numTrainPairs * 500 * 500; % Generous estimate
skinPixelsCbCr = zeros(maxTotalSkinPixels, 2, 'double'); % Store Cb, Cr (2 Columns)
pixelCount = 0;

fprintf('Collecting CbCr skin pixel data from training images...\n');
for i = 1:numTrainPairs
    try
        % Load image and mask
        rgbImage = im2double(imread(fullfile(trainImageFolder, trainImageNames{i})));
        mask = im2double(imread(fullfile(trainMaskFolder, trainMaskNames{i})));
        if ndims(mask) == 3, mask = mask(:,:,1); end
        mask_logical = mask > 0.5; % Ensure logical mask (Hand=1)

        % Ensure 3-channel RGB
        if ndims(rgbImage) == 2, rgbImage = repmat(rgbImage, [1 1 3]);
        elseif size(rgbImage,3) == 4, rgbImage = rgbImage(:,:,1:3);
        elseif size(rgbImage,3) ~= 3, continue;
        end

        % Convert to YCbCr
        ycbcrImage = rgb2ycbcr(rgbImage);
        % Y  = ycbcrImage(:,:,1); % Y channel is NOT used
        Cb = ycbcrImage(:,:,2);
        Cr = ycbcrImage(:,:,3);

        % Extract skin pixels using the mask for Cb and Cr channels
        skinCb = Cb(mask_logical);
        skinCr = Cr(mask_logical);

        numSkinPixels = length(skinCb); % Use length of Cb (same as Cr)
        if numSkinPixels > 0
            if pixelCount + numSkinPixels > maxTotalSkinPixels
                warning('Exceeded preallocated pixel storage. Increase maxTotalSkinPixels.');
                numToAdd = maxTotalSkinPixels - pixelCount;
            else
                numToAdd = numSkinPixels;
            end

            startIdx = pixelCount + 1;
            endIdx = pixelCount + numToAdd;
            % Store only Cb and Cr
            skinPixelsCbCr(startIdx:endIdx, 1) = skinCb(1:numToAdd);
            skinPixelsCbCr(startIdx:endIdx, 2) = skinCr(1:numToAdd);
            pixelCount = endIdx;
        end

        if mod(i, 50) == 0 || i == numTrainPairs
            fprintf('Processed %d/%d training images for skin pixels...\n', i, numTrainPairs);
        end

    catch ME
        warning('Error processing training pair %d (%s): %s. Skipping.', i, trainImageNames{i}, ME.message);
    end
end

% Trim unused rows
skinPixelsCbCr = skinPixelsCbCr(1:pixelCount, :);

if isempty(skinPixelsCbCr)
    error('No skin pixels were collected. Check masks or image paths.');
end
fprintf('Collected %d CbCr skin pixels.\n', pixelCount);

% --- Train the GMM ---
fprintf('Training GMM with %d components on CbCr data...\n', numGaussians);
options = statset('MaxIter', 500, 'Display', 'iter');
try
    % Train on the 2D data
    skinGMM = fitgmdist(skinPixelsCbCr, numGaussians, 'Options', options, 'CovarianceType', 'full', 'SharedCovariance', false);
    fprintf('GMM training complete.\n');
catch ME
    error('GMM training failed: %s\nConsider reducing numGaussians or checking data.', ME.message);
end

% --- Save the GMM ---
fprintf('Saving GMM model to %s...\n', gmmModelFile);
save(gmmModelFile, 'skinGMM');

fprintf('--- GMM Training Phase Complete ---\n\n');

% =========================================================================
% Phase 2: Prediction using the CbCr GMM
% =========================================================================
fprintf('--- Phase 2: Predicting Masks using CbCr GMM ---\n');

% --- Load the Trained GMM ---
if ~exist(gmmModelFile, 'file')
    error('Trained GMM file "%s" not found. Run Phase 1 first.', gmmModelFile);
end
fprintf('Loading trained GMM from %s...\n', gmmModelFile);
load(gmmModelFile, 'skinGMM'); % Loads the 'skinGMM' variable

% --- Get Validation Files ---
validationImageFiles = dir(fullfile(validationImageFolder, '*.jpg'));
validationImageNames = {validationImageFiles.name};
numValidationImages = length(validationImageNames);

if numValidationImages == 0
    warning('No validation images found in %s.', validationImageFolder);
    return;
end

% --- Create Output Folder ---
if ~exist(validationOutputFolder, 'dir')
    mkdir(validationOutputFolder);
    fprintf('Created output folder: %s\n', validationOutputFolder);
end

fprintf('Processing %d validation images...\n', numValidationImages);
for i = 1:numValidationImages
    [~, baseName, ~] = fileparts(validationImageNames{i}); % Get base name for saving
    fprintf('Predicting mask for: %s\n', validationImageNames{i});
    try
        % Load image
        rgbImage = im2double(imread(fullfile(validationImageFolder, validationImageNames{i})));

        % --- Call Prediction Function (CbCr version) ---
        finalMask = predictHandMaskGMM_CbCr(rgbImage, skinGMM, probThreshold, se_open, se_close);

        % --- Save the final mask using the original base name ---
        outputFilename = fullfile(validationOutputFolder, [baseName '.png']); % Save as PNG with original name
        imwrite(uint8(finalMask * 255), outputFilename);

    catch ME
        warning('Error predicting mask for %s: %s. Skipping.', validationImageNames{i}, ME.message);
    end
end

fprintf('--- Prediction Phase Complete. Masks saved to %s ---\n', validationOutputFolder);

% =========================================================================
% Prediction Function Definition (CbCr version)
% =========================================================================
function finalMask = predictHandMaskGMM_CbCr(rgbImage, skinGMM, threshold, seOpen, seClose)
    % Predicts a hand mask using a pre-trained CbCr skin GMM.

    % Ensure 3-channel RGB for conversion
    if ndims(rgbImage) == 2, rgbImage = repmat(rgbImage, [1 1 3]);
    elseif size(rgbImage,3) == 4, rgbImage = rgbImage(:,:,1:3);
    elseif size(rgbImage,3) ~= 3
       error('Input image must be RGB or Grayscale.');
    end

    imgHeight = size(rgbImage, 1);
    imgWidth = size(rgbImage, 2);

    % 1. Convert to YCbCr
    ycbcrImage = rgb2ycbcr(rgbImage);
    % Y  = ycbcrImage(:,:,1); % Not used
    Cb = ycbcrImage(:,:,2);
    Cr = ycbcrImage(:,:,3);

    % 2. Get CbCr data for all pixels
    allPixelsCbCr = [Cb(:), Cr(:)]; % Now a N x 2 matrix

    % 3. Calculate PDF (Probability Density) using the skin GMM on CbCr data
    skinProbability = pdf(skinGMM, allPixelsCbCr);

    % Check for NaN/Inf probabilities
    if any(isnan(skinProbability)) || any(isinf(skinProbability))
       warning('NaN or Inf detected in GMM PDF output. Clamping problematic values to 0.');
       skinProbability(isnan(skinProbability) | isinf(skinProbability)) = 0;
    end

    % 4. Reshape probability map back to image dimensions
    probMap = reshape(skinProbability, imgHeight, imgWidth);

    % 5. Threshold the probability map
    binaryMask = probMap > threshold;

    % 6. Morphological Cleaning
    cleanedMask = imopen(binaryMask, seOpen);
    cleanedMask = imclose(cleanedMask, seClose);

    % 7. Contour Filtering - Keep largest connected component
    cc = bwconncomp(cleanedMask);
    if cc.NumObjects == 0
        finalMask = false(imgHeight, imgWidth);
        return;
    end
    stats = regionprops(cc, 'Area');
    [~, largestCompIdx] = max([stats.Area]);

    finalMask = false(imgHeight, imgWidth);
    finalMask(cc.PixelIdxList{largestCompIdx}) = true;

    % Convert logical mask to double [0,1]
    finalMask = im2double(finalMask);

end