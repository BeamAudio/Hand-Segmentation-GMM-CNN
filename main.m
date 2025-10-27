% =============================================================================
% HAND SEGMENTATION PIPELINE (GMM + PRIMARY CNN + POST-PROCESSING CNN)
% =============================================================================
%
% This script implements a three-stage hand segmentation pipeline:
%   1.  A Gaussian Mixture Model (GMM) creates a skin probability map
%       from the Cb/Cr channels of a YCbCr image.
%   2.  A Primary Convolutional Neural Network (CNN) refines the initial
%       segmentation using skin probability and gradient maps as 2-channel input.
%   3.  A Post-Processing CNN further refines the output using the primary
%       CNN's result and the gradient map to smooth edges and fill holes.
%
% REQUIREMENTS:
%   - MATLAB
%   - Image Processing Toolbox
%   - Statistics and Machine Learning Toolbox
%   - Deep Learning Toolbox
%
% PHASES:
%   1. Train/Load GMM for skin color modeling.
%   2. Generate feature maps (skin probability + gradient) as CNN input.
%   3. Train/Load Primary CNN.
%   4. Generate Primary CNN outputs for Post-Processing CNN training.
%   5. Train/Load Post-Processing CNN.
%   6. Process datasets using both CNNs, evaluate, save outputs, and visualize
%      results including intermediate feature maps for selected samples.
%
% =============================================================================

% Clear workspace and close figures
clearvars; clc; 

% Define global parameters
globalParams = struct();
globalParams.INPUT_SIZE = 480; % Size for all network inputs
globalParams.TRAINING_SAMPLES_PRIMARY = 80; % Max samples for primary CNN
globalParams.TRAINING_SAMPLES_POST = 60; % Max samples for post-processing CNN
globalParams.BATCH_SIZE = 4; % Mini-batch size for training
globalParams.MAX_EPOCHS_PRIMARY = 20; % Training epochs for primary CNN
globalParams.MAX_EPOCHS_POST = 20; % Training epochs for post-processing CNN
globalParams.INIT_LR_PRIMARY = 1e-3; % Initial learning rate for primary CNN
globalParams.INIT_LR_POST = 1e-3; % Initial learning rate for post-processing CNN
globalParams.DISPLAY_INDICES = [10, 15]; % Indices of samples to display results for
globalParams.HAND_VALUE = 0; % Pixel value for hand in final mask
globalParams.BACKGROUND_VALUE = 1; % Pixel value for background in final mask
globalParams.FORCE_RETRAIN_GMM = false; % Force retrain GMM if model exists
globalParams.FORCE_RETRAIN_PRIMARY = false; % Force retrain Primary CNN if model exists
globalParams.FORCE_RETRAIN_POST = false; % Force retrain Post-processing CNN if model exists



% Define file paths
paths = struct();
paths.trainImageFolder = fullfile('Training-Dataset', 'Training-Dataset', 'Images');
paths.trainMaskFolder  = fullfile('Training-Dataset', 'Training-Dataset', 'Masks-Ideal');
paths.valImageFolder   = fullfile('Validation-Dataset', 'Validation-Dataset', 'Images');
paths.valMaskFolder    = fullfile('Validation-Dataset', 'Validation-Dataset', 'Masks-Ideal');
paths.trainFeatureFolder = fullfile('Training-Dataset', 'Training-Dataset', 'Features-GMM-Grad');
paths.valFeatureFolder   = fullfile('Validation-Dataset', 'Validation-Dataset', 'Features-GMM-Grad');
paths.trainOutputFolder = fullfile('Training-Dataset', 'Training-Dataset', 'Masks');
paths.valOutputFolder   = fullfile('Validation-Dataset', 'Validation-Dataset', 'Masks');
paths.primaryCnnOutputsFolder = fullfile('Training-Dataset', 'Training-Dataset', 'PrimaryCNN_Outputs'); % New folder for primary outputs
paths.gmmModelFile = 'skinGMM_CbCr_model.mat';
paths.primaryCnnModelFile = sprintf('handRefineCNN_%d.mat', globalParams.INPUT_SIZE);
paths.postCnnModelFile = sprintf('postProcessCNN_%d.mat', globalParams.INPUT_SIZE);

% Create necessary directories
if ~exist(paths.trainFeatureFolder, 'dir'), mkdir(paths.trainFeatureFolder); end
if ~exist(paths.valFeatureFolder, 'dir'), mkdir(paths.valFeatureFolder); end
if ~exist(paths.trainOutputFolder, 'dir'), mkdir(paths.trainOutputFolder); end
if ~exist(paths.valOutputFolder, 'dir'), mkdir(paths.valOutputFolder); end
if ~exist(paths.primaryCnnOutputsFolder, 'dir'), mkdir(paths.primaryCnnOutputsFolder); end

fprintf('--- Hand Segmentation Pipeline Started (Input Size: %dx%d) ---\n', globalParams.INPUT_SIZE, globalParams.INPUT_SIZE);

% =============================================================================
% PHASE 1: TRAIN/LOAD GMM
% =============================================================================
fprintf('--- Phase 1: Training/Loading GMM ---\n');
gmmModel = []; % Initialize variable for compatibility
if exist(paths.gmmModelFile, 'file') && ~globalParams.FORCE_RETRAIN_GMM
    fprintf('Found existing GMM model file: %s\n', paths.gmmModelFile);
    try
        loadedData = load(paths.gmmModelFile);
        % Check for the new required variables
        if isfield(loadedData, 'skinGMM') && isfield(loadedData, 'nonSkinGMM') && ...
           isfield(loadedData, 'priorSkin') && isfield(loadedData, 'priorNonSkin')
            
            fprintf('GMM file contains skinGMM, nonSkinGMM, and priors. Ready.\n');
            gmmModel = loadedData.skinGMM; % Assign for compatibility, though generateFeatures won't use it
        else
            fprintf('Saved GMM file is missing required models/priors. Retraining...\n');
            gmmModel = trainGMM(paths.trainImageFolder, paths.trainMaskFolder, paths.gmmModelFile);
        end
    catch ME
        fprintf('Error loading GMM file: %s. Forcing retrain...\n', ME.message);
        gmmModel = trainGMM(paths.trainImageFolder, paths.trainMaskFolder, paths.gmmModelFile);
    end
else
    if globalParams.FORCE_RETRAIN_GMM
        fprintf('Forcing retrain of GMM (Skin and Non-Skin)...\n');
    else
        fprintf('No GMM file found. Training GMM (Skin and Non-Skin)...\n');
    end
    gmmModel = trainGMM(paths.trainImageFolder, paths.trainMaskFolder, paths.gmmModelFile);
    fprintf('GMMs trained and saved.\n');
end

% =============================================================================
% PHASE 2: GENERATE FEATURES
% =============================================================================
fprintf('--- Phase 2: Generating Feature Maps ---\n');
% The gmmModel variable is passed but will be ignored by the new function,
% which loads the model file directly.
generateFeatures(paths.trainImageFolder, paths.trainFeatureFolder, gmmModel);
generateFeatures(paths.valImageFolder, paths.valFeatureFolder, gmmModel);

% =============================================================================
% PHASE 3: TRAIN OR LOAD PRIMARY CNN
% =============================================================================
fprintf('--- Phase 3: Training/Loading Primary CNN ---\n');
if exist(paths.primaryCnnModelFile, 'file') && ~globalParams.FORCE_RETRAIN_PRIMARY
    fprintf('Loading saved Primary CNN model...\n');
    load(paths.primaryCnnModelFile, 'primaryCnnNet');
    fprintf('Primary CNN model loaded.\n');
else
    fprintf('Training Primary CNN...\n');
    primaryCnnNet = trainPrimaryCNN(paths.trainFeatureFolder, paths.trainMaskFolder, globalParams);
    save(paths.primaryCnnModelFile, 'primaryCnnNet');
    fprintf('Primary CNN trained and saved.\n');
end
if isempty(primaryCnnNet)
    error('Primary CNN model could not be loaded or trained.');
end

% =============================================================================
% PHASE 4: GENERATE PRIMARY CNN OUTPUTS FOR POST-PROCESSING TRAINING
% =============================================================================
fprintf('--- Phase 4: Generating Primary CNN Outputs for Post-Processing Training ---\n');
generatePrimaryOutputsForTraining(paths.trainFeatureFolder, paths.primaryCnnOutputsFolder, primaryCnnNet);

% =============================================================================
% PHASE 5: TRAIN OR LOAD POST-PROCESSING CNN
% =============================================================================
fprintf('--- Phase 5: Training/Loading Post-Processing CNN ---\n');
if exist(paths.postCnnModelFile, 'file') && ~globalParams.FORCE_RETRAIN_POST
    fprintf('Loading saved Post-Processing CNN model...\n');
    load(paths.postCnnModelFile, 'postCnnNet');
    fprintf('Post-Processing CNN model loaded.\n');
else
    fprintf('Training Post-Processing CNN...\n');
    postCnnNet = trainPostProcessingCNN(paths.trainFeatureFolder, paths.primaryCnnOutputsFolder, paths.trainMaskFolder, globalParams);
    save(paths.postCnnModelFile, 'postCnnNet');
    fprintf('Post-Processing CNN trained and saved.\n');
end
if isempty(postCnnNet)
    error('Post-Processing CNN model could not be loaded or trained.');
end

% =============================================================================
% PHASE 6: PROCESS DATASETS & VISUALIZE
% =============================================================================
fprintf('--- Phase 6: Processing Datasets ---\n');
processDataset(paths.trainFeatureFolder, paths.trainMaskFolder, paths.trainOutputFolder, ...
     globalParams, paths.trainImageFolder, primaryCnnNet, postCnnNet, []);
processDataset(paths.valFeatureFolder, paths.valMaskFolder, paths.valOutputFolder, ...
     globalParams, paths.valImageFolder, primaryCnnNet, postCnnNet, globalParams.DISPLAY_INDICES);

fprintf('--- Pipeline Finished ---\n');

% =============================================================================
% FUNCTION DEFINITIONS
% =============================================================================
function gmmModel = trainGMM(imgFolder, maskFolder, modelFile)
    imgFiles = dir(fullfile(imgFolder, '*.jpg'));
    mskFiles = dir(fullfile(maskFolder, '*.bmp'));
    imgNames = {imgFiles.name}; mskNames = {mskFiles.name};
    [~, iB, ~] = cellfun(@fileparts, imgNames, 'UniformOutput', false);
    [~, mB, ~] = cellfun(@fileparts, mskNames, 'UniformOutput', false);
    [common, ia, ib] = intersect(iB, mB);
    if isempty(common), error('No matching image/mask pairs found for GMM training!'); end
    imgNames = imgNames(ia); mskNames = mskNames(ib);
    N = length(imgNames);
    
    % Allocate space for both skin and non-skin pixels
    % Use a reasonable cap, e.g., 5 million pixels per class
    maxPixelsPerClass = 20000000; 
    skinPixels = zeros(maxPixelsPerClass, 4, 'double');
    nonSkinPixels = zeros(maxPixelsPerClass, 4, 'double');
    skinPixelCount = 0;
    nonSkinPixelCount = 0;

    fprintf('Collecting skin and non-skin pixels from %d training images...\n', N);
    for i = 1:N
        fprintf('  GMM Data: %d/%d\r', i, N);
        try
            rgb = im2double(imread(fullfile(imgFolder, imgNames{i})));
            mask = im2double(imread(fullfile(maskFolder, mskNames{i})));
            if ndims(mask) == 3, mask = mask(:,:,1); end
            
            % Get *both* masks
            handMask = mask < 0.5;
            nonHandMask = mask >= 0.5; % The non-skin pixels

            if ndims(rgb) == 2, rgb = repmat(rgb, [1 1 3]); end
            if size(rgb,3) == 4, rgb = rgb(:,:,1:3); end
            if size(rgb,3) ~= 3, continue; end
            
            ycbcrImg = rgb2ycbcr(rgb);
            Cb = ycbcrImg(:,:,2); Cr = ycbcrImg(:,:,3);
            hsvImg = rgb2hsv(rgb);
            H = hsvImg(:,:,1); S = hsvImg(:,:,2);
            
            % Get skin pixels
            skinCb = Cb(handMask); skinCr = Cr(handMask);
            skinH = H(handMask); skinS = S(handMask);
            nSkin = length(skinCb);
            
            % Get non-skin pixels
            nonSkinCb = Cb(nonHandMask); nonSkinCr = Cr(nonHandMask);
            nonSkinH = H(nonHandMask); nonSkinS = S(nonHandMask);
            nNonSkin = length(nonSkinCb);

            % Add skin pixels
            if nSkin > 0
                add = min(nSkin, maxPixelsPerClass - skinPixelCount);
                s = skinPixelCount + 1; e = skinPixelCount + add;
                skinPixels(s:e, 1) = skinCb(1:add);
                skinPixels(s:e, 2) = skinCr(1:add);
                skinPixels(s:e, 3) = skinH(1:add);
                skinPixels(s:e, 4) = skinS(1:add);
                skinPixelCount = e;
            end
            
            % Add non-skin pixels
            if nNonSkin > 0
                add = min(nNonSkin, maxPixelsPerClass - nonSkinPixelCount);
                s = nonSkinPixelCount + 1; e = nonSkinPixelCount + add;
                nonSkinPixels(s:e, 1) = nonSkinCb(1:add);
                nonSkinPixels(s:e, 2) = nonSkinCr(1:add);
                nonSkinPixels(s:e, 3) = nonSkinH(1:add);
                nonSkinPixels(s:e, 4) = nonSkinS(1:add);
                nonSkinPixelCount = e;
            end

        catch ME
             fprintf('\nWarning: Error processing image %s: %s\n', imgNames{i}, ME.message);
        end
    end
    
    fprintf('\nCollected %d skin pixels and %d non-skin pixels.\n', skinPixelCount, nonSkinPixelCount);
    
    % Trim arrays
    skinPixels = skinPixels(1:skinPixelCount, :);
    nonSkinPixels = nonSkinPixels(1:nonSkinPixelCount, :);
    
    if isempty(skinPixels) || isempty(nonSkinPixels)
        error('Not enough skin or non-skin pixels collected for GMM training!');
    end
    
    % Handle Data Imbalance: Subsample non-skin pixels
    % To prevent non-skin from dominating and for faster training
    % Let's use a 5:1 ratio of non-skin to skin
    maxNonSkin = skinPixelCount * 6;
    if nonSkinPixelCount > maxNonSkin
        fprintf('Subsampling non-skin pixels from %d to %d (5x skin pixels)\n', ...
                nonSkinPixelCount, maxNonSkin);
        nonSkinPixels = datasample(nonSkinPixels, maxNonSkin, 'Replace', false);
        nonSkinPixelCount = size(nonSkinPixels, 1);
    end

    % Calculate priors based on the (potentially subsampled) training data
    totalPixels = skinPixelCount + nonSkinPixelCount;
    priorSkin = skinPixelCount / totalPixels;
    priorNonSkin = nonSkinPixelCount / totalPixels;
    
    fprintf('Training Priors: Skin=%.4f, NonSkin=%.4f\n', priorSkin, priorNonSkin);
    

    
    opts = statset('MaxIter', 300, 'Display', 'iter');
    
    % Fit GMM for SKIN
    fprintf('Training Skin GMM (5 components)...\n');
    skinGMM = fitgmdist(skinPixels, 5, 'Options', opts, 'CovarianceType', 'diagonal');
    
    % Fit GMM for NON-SKIN (use more components as it's more varied)
    fprintf('Training Non-Skin GMM (10 components)...\n');
    nonSkinGMM = fitgmdist(nonSkinPixels, 10, 'Options', opts, 'CovarianceType', 'diagonal'); 
    
    % Save BOTH models and the priors
    fprintf('Saving GMM models and priors to %s...\n', modelFile);
    save(modelFile, 'skinGMM', 'nonSkinGMM', 'priorSkin', 'priorNonSkin');
    
    % Return the skin GMM just for compatibility with the main script's load check
    gmmModel = skinGMM;
end

function generateFeatures(imgFolder, featFolder, gmmModel) % gmmModel is no longer used
    probFolder = fullfile(featFolder, 'prob');
    gradFolder = fullfile(featFolder, 'grad');
    if ~exist(probFolder, 'dir'), mkdir(probFolder); end
    if ~exist(gradFolder, 'dir'), mkdir(gradFolder); end
    
    imgFiles = dir(fullfile(imgFolder, '*.jpg'));
    imgNames = {imgFiles.name};
    N = length(imgNames);
    
    % --- Load the GMMs and Priors ---
    % We load them once outside the loop.
    gmmModelFile = 'skinGMM_CbCr_model.mat'; % Assumes the name defined in paths
    fprintf('Loading GMMs and priors from %s...\n', gmmModelFile);
    try
        gmmData = load(gmmModelFile);
        skinGMM = gmmData.skinGMM;
        nonSkinGMM = gmmData.nonSkinGMM;
        priorSkin = gmmData.priorSkin;
        priorNonSkin = gmmData.priorNonSkin;
    catch ME
        error('Could not load the GMM models and priors. Retrain the GMMs. Error: %s', ME.message);
    end
    
    fprintf('Generating features for %s (%d images)...\n', featFolder, N);
    parfor i = 1:N
        [~, base, ~] = fileparts(imgNames{i});
        probPath = fullfile(probFolder, [base '.png']);
        gradPath = fullfile(gradFolder, [base '.png']);
        if exist(probPath, 'file') && exist(gradPath, 'file'), continue; end
        try
            rgb = im2double(imread(fullfile(imgFolder, imgNames{i})));
            if ndims(rgb) == 2, rgb = repmat(rgb, [1 1 3]); end
            if size(rgb,3) == 4, rgb = rgb(:,:,1:3); end
            if size(rgb,3) ~= 3, continue; end
            
            ycbcrImg = rgb2ycbcr(rgb);
            Cb = ycbcrImg(:,:,2); Cr = ycbcrImg(:,:,3);
            hsvImg = rgb2hsv(rgb);
            H = hsvImg(:,:,1); S = hsvImg(:,:,2);
            
            % Reshape channels into column vectors
            pixelsToEvaluate = [Cb(:), Cr(:), H(:), S(:)];
            
            % Calculate likelihoods for BOTH classes
            likelihoodSkin = pdf(skinGMM, pixelsToEvaluate);
            likelihoodNonSkin = pdf(nonSkinGMM, pixelsToEvaluate);
            
            % Calculate posterior probability using Bayes' rule
            % P(skin|x) = [P(x|skin) * P(skin)] / [P(x|skin) * P(skin) + P(x|non-skin) * P(non-skin)]
            numerator = likelihoodSkin .* priorSkin;
            denominator = numerator + (likelihoodNonSkin .* priorNonSkin);
            
            % Add epsilon to prevent division by zero (e.g., for pure black pixels)
            posteriorSkin = numerator ./ (denominator + eps);
            
            % Reshape back to an image
            probMap = reshape(posteriorSkin, size(Cb));
            
            % --- CRITICAL CHANGE: DO NOT NORMALIZE ---
            % The probMap is now a true probability [0, 1].
            % mat2gray(probMap) would destroy this information.
            
            % Calculate gradient
            Y = ycbcrImg(:,:,1); 
            Y_smooth = imgaussfilt(Y, 1.0);
            [gradMag, ~] = imgradient(Y_smooth, 'sobel');
            gradMap = mat2gray(gradMag);
            
            % Save the true probability map (imwrite handles [0,1] float data)
            imwrite(probMap, probPath);
            imwrite(uint8(gradMap * 255), gradPath);
        catch ME
             fprintf('\nWarning: Error generating features for %s: %s\n', imgNames{i}, ME.message);
        end
    end
    fprintf('Feature generation complete for %s.\n', featFolder);
end

function net = trainPrimaryCNN(featFolder, maskFolder, params)
    probDir = fullfile(featFolder, 'prob');
    gradDir = fullfile(featFolder, 'grad');
    probFiles = dir(fullfile(probDir, '*.png'));
    pNames = {probFiles.name};
    N = min(params.TRAINING_SAMPLES_PRIMARY, length(pNames));
    X = []; Y = [];
    fprintf('Loading data for Primary CNN training (up to %d samples)...\n', N);
    for i = 1:N
        try
            [~, base, ~] = fileparts(pNames{i});
            probMap = im2double(imread(fullfile(probDir, pNames{i})));
            gradMap = im2double(imread(fullfile(gradDir, [base '.png'])));
            if ndims(probMap) == 3, probMap = probMap(:,:,1); end % <-- ADD
            if ndims(gradMap) == 3, gradMap = gradMap(:,:,1); end % <-- ADD
            if isempty(probMap) || isempty(gradMap)
                fprintf('\nWarning: Empty feature map detected for training sample %s. Skipping.\n', pNames{i});
                continue;
            end
            maskPath = fullfile(maskFolder, [base '.bmp']);
            if ~exist(maskPath, 'file'), maskPath = fullfile(maskFolder, [base '.png']); end
            if ~exist(maskPath, 'file'), continue; end
            ideal = imread(maskPath);
            if ndims(ideal) == 3, ideal = ideal(:,:,1); end
            ideal = ideal < 0.5; % Logical mask

            p_resized = imresize(single(probMap), [params.INPUT_SIZE params.INPUT_SIZE], 'bilinear');
            g_resized = imresize(single(gradMap), [params.INPUT_SIZE params.INPUT_SIZE], 'bilinear');
            if ~ismatrix(p_resized) || ~ismatrix(g_resized) || isempty(p_resized) || isempty(g_resized)
                 fprintf('\nWarning: Invalid resized feature map for training sample %s. Skipping.\n', pNames{i});
                 continue;
            end
            i_resized = imresize(single(ideal), [params.INPUT_SIZE params.INPUT_SIZE], 'nearest');
            
            inputVol = cat(3, p_resized, g_resized);
             if ~isequal(size(inputVol), [params.INPUT_SIZE, params.INPUT_SIZE, 2])
                 fprintf('\nWarning: Incorrect input volume size for training sample %s. Skipping.\n', pNames{i});
                 continue;
             end
            X = cat(4, X, inputVol);
            Y = cat(4, Y, i_resized);
        catch ME
            fprintf('\nWarning: Error loading training sample %s: %s\n', pNames{i}, ME.message);
        end
    end
    if isempty(X), error('No valid data loaded for Primary CNN training!'); end
    fprintf('\nLoaded %d valid samples for training.\n', size(X, 4));
    fprintf('Defining Primary CNN architecture for %dx%d input...\n', params.INPUT_SIZE, params.INPUT_SIZE);
    layers = [
        imageInputLayer([params.INPUT_SIZE params.INPUT_SIZE 2], 'Normalization','none', 'Name','input')
        convolution2dLayer(5, 2, 'Padding','same', 'Name','conv1')
        batchNormalizationLayer('Name','bn1')
        reluLayer('Name','relu1')
        convolution2dLayer(5, 4, 'Padding','same', 'Name','conv2')
        batchNormalizationLayer('Name','bn2')
        reluLayer('Name','relu2')
        convolution2dLayer(3, 16, 'Padding','same', 'Name','conv3')
        batchNormalizationLayer('Name','bn3')
        reluLayer('Name','relu3')
        convolution2dLayer(3, 16, 'Padding','same', 'Name','conv4')
        batchNormalizationLayer('Name','bn4')
        reluLayer('Name','relu4')
        convolution2dLayer(3, 16, 'Padding','same', 'Name','conv5')
        batchNormalizationLayer('Name','bn5')
        reluLayer('Name','relu5')
        convolution2dLayer(3, 16, 'Padding','same', 'Name','conv6')
        batchNormalizationLayer('Name','bn6')
        reluLayer('Name','relu6')
        convolution2dLayer(1, 1, 'Padding','same', 'Name','conv_out')
        sigmoidLayer('Name','sigmoid')
        regressionLayer('Name','output')];
    opts = trainingOptions('adam', ...
        'MaxEpochs', params.MAX_EPOCHS_PRIMARY, 'MiniBatchSize',min(params.BATCH_SIZE,size(X,4)), ...
        'InitialLearnRate',params.INIT_LR_PRIMARY, 'Shuffle','every-epoch', ...
        'Plots','training-progress', 'Verbose',true, 'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.5, 'LearnRateDropPeriod', 5);
    fprintf('Starting Primary CNN training...\n');
    net = trainNetwork(X, Y, layers, opts);
    fprintf('Primary CNN training finished.\n');
end

function generatePrimaryOutputsForTraining(featFolder, outputFolder, primaryCnnNet)
    probDir = fullfile(featFolder, 'prob');
    gradDir = fullfile(featFolder, 'grad');
    probFiles = dir(fullfile(probDir, '*.png'));
    pNames = {probFiles.name};
    N = length(pNames);
    fprintf('Generating Primary CNN outputs for %d training samples...\n', N);
    for i = 1:N
        [~, base, ~] = fileparts(pNames{i});
        outputPath = fullfile(outputFolder, [base '_primary_output.png']);
        if exist(outputPath, 'file'), continue; end % Skip if already exists
        
        try
            probMap = im2double(imread(fullfile(probDir, pNames{i})));
            gradMap = im2double(imread(fullfile(gradDir, [base '.png'])));
            if ndims(probMap) == 3, probMap = probMap(:,:,1); end % <-- ADD
            if ndims(gradMap) == 3, gradMap = gradMap(:,:,1); end % <-- ADD
            
            % Prepare input for primary CNN
            p_resized = imresize(single(probMap), [480 480], 'bilinear');
            g_resized = imresize(single(gradMap), [480 480], 'bilinear');
            inputVol = cat(3, p_resized, g_resized);
            inputVol4D = inputVol(:,:,:,1);
            
            % Predict with primary CNN
            Y_primary = predict(primaryCnnNet, inputVol4D);
            
            % Save the output
            imwrite(Y_primary, outputPath);
        catch ME
            fprintf('\nWarning: Error generating Primary CNN output for %s: %s\n', pNames{i}, ME.message);
        end
    end
    fprintf('Primary CNN outputs generation complete for %s.\n', outputFolder);
end

function net = trainPostProcessingCNN(featFolder, primaryOutputFolder, maskFolder, params)
    probDir = fullfile(featFolder, 'prob');
    gradDir = fullfile(featFolder, 'grad');
    mskFiles = dir(fullfile(maskFolder, '*.bmp'));
    if isempty(mskFiles), mskFiles = dir(fullfile(maskFolder, '*.png')); end
    mNames = {mskFiles.name};
    N = min(params.TRAINING_SAMPLES_POST, length(mNames));
    X = []; Y = [];
    fprintf('Loading data for Post-Processing CNN training (up to %d samples)...\n', N);
    for i = 1:N
        try
            [~, base, ~] = fileparts(mNames{i});
            maskPath = fullfile(maskFolder, mNames{i});
            ideal = imread(maskPath);
            if ndims(ideal) == 3, ideal = ideal(:,:,1); end
            ideal = ideal < 0.5; % Logical mask
            
            % Load corresponding gradient map
            gradPath = fullfile(gradDir, [base '.png']);
            if ~exist(gradPath, 'file')
                fprintf('\nWarning: Gradient map not found for sample %s. Skipping.\n', base);
                continue;
            end
            gradMap = im2double(imread(gradPath));
            
            
            % Load corresponding primary CNN output
            primaryOutputPath = fullfile(primaryOutputFolder, [base '_primary_output.png']);
            if ~exist(primaryOutputPath, 'file')
                fprintf('\nWarning: Primary CNN output not found for sample %s. Skipping.\n', base);
                continue;
            end
            primaryOutputMap = im2double(imread(primaryOutputPath));
            if ndims(primaryOutputMap) == 3, primaryOutputMap = primaryOutputMap(:,:,1); end % <-- ADD
            
            target = imresize(single(ideal), [params.INPUT_SIZE params.INPUT_SIZE], 'nearest');
            grad_resized = imresize(single(gradMap), [params.INPUT_SIZE params.INPUT_SIZE], 'bilinear');
            primary_resized = imresize(single(primaryOutputMap), [params.INPUT_SIZE params.INPUT_SIZE], 'bilinear');
            
            % Input is a 2-channel volume (primary output + gradient)
            inputVol = cat(3, primary_resized, grad_resized);
            
            if ~isequal(size(inputVol), [params.INPUT_SIZE, params.INPUT_SIZE, 2])
                 fprintf('\nWarning: Incorrect input volume size for training sample %s. Skipping.\n', mNames{i});
                 continue;
            end
            X = cat(4, X, inputVol);
            Y = cat(4, Y, target);
        catch ME
            fprintf('\nWarning: Error loading training sample %s: %s\n', mNames{i}, ME.message);
        end
    end
    if isempty(X), error('No valid data loaded for Post-Processing CNN training!'); end
    fprintf('\nLoaded %d valid samples for Post-Processing CNN training.\n', size(X, 4));
    fprintf('Defining Post-Processing CNN architecture for %dx%d input...\n', params.INPUT_SIZE, params.INPUT_SIZE);
    layers = [
        % Encoder (Contracting Path)
        imageInputLayer([params.INPUT_SIZE params.INPUT_SIZE 2], 'Normalization','none', 'Name','input')
        
        % Block 1 (8 -> 4 filters)
        convolution2dLayer(3, 4, 'Padding','same', 'Name','enc_conv1')
        batchNormalizationLayer('Name','enc_bn1')
        reluLayer('Name','enc_relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name','enc_pool1') 
        
        % Block 2 (16 -> 8 filters)
        convolution2dLayer(3, 16, 'Padding','same', 'Name','enc_conv2')
        batchNormalizationLayer('Name','enc_bn2')
        reluLayer('Name','enc_relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name','enc_pool2') 
        
        % Block 3 (32 -> 16 filters)
        convolution2dLayer(3, 32, 'Padding','same', 'Name','enc_conv3')
        batchNormalizationLayer('Name','enc_bn3')
        reluLayer('Name','enc_relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name','enc_pool3') 
        
        % Bottleneck (64 -> 32 filters)
        convolution2dLayer(3, 64, 'Padding','same', 'Name','bottleneck_conv')
        batchNormalizationLayer('Name','bottleneck_bn')
        reluLayer('Name','bottleneck_relu')
        
        % Decoder (Expansive Path)
        % Block 4 (32 -> 16 filters)
        transposedConv2dLayer(2, 32, 'Stride', 2, 'Cropping', 'same', 'Name','dec_upconv3')
        batchNormalizationLayer('Name','dec_upbn3')
        reluLayer('Name','dec_uprelu3')
        convolution2dLayer(3, 32, 'Padding','same', 'Name','dec_conv4')
        batchNormalizationLayer('Name','dec_bn4')
        reluLayer('Name','dec_relu4')
        
        % Block 5 (16 -> 8 filters)
        transposedConv2dLayer(2,16, 'Stride', 2, 'Cropping', 'same', 'Name','dec_upconv2')
        batchNormalizationLayer('Name','dec_upbn2')
        reluLayer('Name','dec_uprelu2')
        convolution2dLayer(3, 16, 'Padding','same', 'Name','dec_conv5')
        batchNormalizationLayer('Name','dec_bn5')
        reluLayer('Name','dec_relu5')
        
        % Block 6 (8 -> 4 filters)
        transposedConv2dLayer(2, 4, 'Stride', 2, 'Cropping', 'same', 'Name','dec_upconv1')
        batchNormalizationLayer('Name','dec_upbn1')
        reluLayer('Name','dec_uprelu1')
        convolution2dLayer(3, 4, 'Padding','same', 'Name','dec_conv6')
        batchNormalizationLayer('Name','dec_bn6')
        reluLayer('Name','dec_relu6')
        
        % Output
        convolution2dLayer(1, 1, 'Padding','same', 'Name','conv_out')
        sigmoidLayer('Name','sigmoid')
        regressionLayer('Name','output')
    ];
    opts = trainingOptions('adam', ...
        'MaxEpochs', params.MAX_EPOCHS_POST, 'MiniBatchSize',min(params.BATCH_SIZE,size(X,4)), ...
        'InitialLearnRate',params.INIT_LR_POST, 'Shuffle','every-epoch', ...
        'Plots','training-progress', 'Verbose',true, 'LearnRateSchedule', 'piecewise', 'LearnRateDropFactor', 0.5, 'LearnRateDropPeriod', 5);
    fprintf('Starting Post-Processing CNN training...\n');
    net = trainNetwork(X, Y, layers, opts);
    fprintf('Post-Processing CNN training finished.\n');
end

function processDataset(featDir, maskDir, outDir, params, imgDir, primaryCnnNet, postCnnNet, displayIndices)
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    probDir = fullfile(featDir, 'prob');
    probFiles = dir(fullfile(probDir, '*.png'));
    pNames = {probFiles.name};
    N = length(pNames);
    if N == 0, fprintf('No feature files found in %s. Skipping processing.\n', featDir); return; end
    fprintf('Processing %s (%d images using both CNNs)...\n', outDir, N);
    f1Scores = nan(N, 1);
    if isempty(displayIndices), displayIndices = []; end
    displayIndices = displayIndices(displayIndices >= 1 & displayIndices <= N);
    if isempty(primaryCnnNet), error('Primary CNN network object is empty.'); end
    if isempty(postCnnNet), error('Post-Processing CNN network object is empty.'); end
    for k = 1:N
        fprintf('  Processing: %d/%d\r', k, N);
        base = '';
        try
            [~, base, ~] = fileparts(pNames{k});
            probMap = im2double(imread(fullfile(probDir, pNames{k})));
            gradMap = im2double(imread(fullfile(featDir, 'grad', [base '.png'])));

            if ndims(probMap) == 3, probMap = probMap(:,:,1); end
            if ndims(gradMap) == 3, gradMap = gradMap(:,:,1); end

            if isempty(probMap) || isempty(gradMap)
                fprintf('\nWarning: Could not load feature maps for sample %d (%s). Skipping.\n', k, pNames{k});
                continue;
            end
            [H, W] = size(probMap);
            maskPath = fullfile(maskDir, [base '.bmp']);
            if ~exist(maskPath, 'file'), maskPath = fullfile(maskDir, [base '.png']); end
            if ~exist(maskPath, 'file')
                 fprintf('\nWarning: Ground truth mask not found for %s. Cannot evaluate. Skipping.\n', base);
                 continue;
            end
            gtMask = imread(maskPath);
            if ndims(gtMask) == 3, gtMask = gtMask(:,:,1); end
            gtLogical = gtMask < 0.5;
            softMap = [];
            binaryMask = [];
            
            try
                % Resize inputs to network size
                p_resized = imresize(single(probMap), [params.INPUT_SIZE params.INPUT_SIZE], 'bilinear');
                g_resized = imresize(single(gradMap), [params.INPUT_SIZE params.INPUT_SIZE], 'bilinear');
                if ~ismatrix(p_resized) || ~ismatrix(g_resized) || isempty(p_resized) || isempty(g_resized)
                    error('Invalid dimensions after resizing feature maps.');
                end
                inputVol = cat(3, p_resized, g_resized);
                if ~isequal(size(inputVol), [params.INPUT_SIZE, params.INPUT_SIZE, 2])
                     error('Input volume dimensions are incorrect (%s).', mat2str(size(inputVol)));
                end
                inputVol4D = cat(4, inputVol);

                

                % Primary CNN prediction
                Y_primary = predict(primaryCnnNet, inputVol4D);
                Y_primary = squeeze(Y_primary);

                % --- Check Primary CNN output dimensions ---
                if ~isequal(size(Y_primary), [params.INPUT_SIZE, params.INPUT_SIZE])
                     error('Primary CNN output dimensions are incorrect (%s). Expected [%d, %d].', mat2str(size(Y_primary)), params.INPUT_SIZE, params.INPUT_SIZE);
                end
                % --- End of check ---
                
                % Post-processing CNN
                grad_resized_pp = imresize(single(gradMap), [params.INPUT_SIZE params.INPUT_SIZE], 'bilinear');
                postInput = cat(3, Y_primary, grad_resized_pp);
                postInput4D = cat(4, postInput);

                if ~isequal(size(grad_resized_pp), [params.INPUT_SIZE, params.INPUT_SIZE])
                     error('Post-processing gradient map dimensions are incorrect (%s). Expected [%d, %d].', mat2str(size(grad_resized_pp)), params.INPUT_SIZE, params.INPUT_SIZE);
                end

                Y_post = predict(postCnnNet, postInput4D);
                Y_post = squeeze(Y_post);
                
                % Resize back to original size
                softMap = imresize(Y_post, [H W], 'bilinear');
                if isempty(softMap) || ~ismatrix(softMap)
                     error('Post-processed prediction (softMap) is invalid.');
                end
                
                % Save feature maps if selected for display
                if ismember(k, displayIndices)
                    fprintf('\nSaving feature maps for sample %d: %s\n', k, base);
                    saveFeatureMaps(primaryCnnNet, postCnnNet, inputVol4D, postInput4D, outDir, base);
                end
                
                binaryMask = imbinarize(softMap, graythresh(softMap));
            catch ME_cnn
                 fprintf('\nError during CNN prediction/feature extraction for sample %d (%s): %s. Skipping CNN part.\n', k, base, ME_cnn.message);
                 continue;
            end
            if ~isempty(binaryMask)
                finalMask = ones(H, W) * params.BACKGROUND_VALUE;
                finalMask(binaryMask) = params.HAND_VALUE;
                imwrite(uint8(finalMask * 255), fullfile(outDir, [base '.png']));
                f1Scores(k) = calculateF1(binaryMask, gtLogical);
            else
                f1Scores(k) = NaN;
            end
            
            % Display results if selected
            if ismember(k, displayIndices)
                fprintf('Displaying results for sample %d: %s\n', k, base);
                imgPath = fullfile(imgDir, [base '.jpg']);
                if exist(imgPath, 'file')
                    try
                        origImg = im2double(imread(imgPath));
                        if ndims(origImg) == 2, origImg = repmat(origImg, [1 1 3]); end
                        figure('Name', sprintf('Detailed Results: %s (Sample %d)', base, k), 'NumberTitle', 'off', 'WindowState', 'maximized');
                        subplot(2,3,1); imshow(origImg); title('Original Image');
                        subplot(2,3,2); imshow(probMap, []); title('Input: GMM Prob');
                        subplot(2,3,3); imshow(gradMap, []); title('Input: Grad Mag');
                        subplot(2,3,4); imshow(Y_primary, []); title('Primary CNN Output');
                        subplot(2,3,5); imshow(softMap, []); title('Post-Proc Output');
                        subplot(2,3,6); imshow(gtLogical); title('Ground Truth');
                        sgtitle(sprintf('Segmentation Pipeline - %s', base), 'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold');
                        drawnow;
                    catch ME_plot
                        fprintf('\nWarning: Could not display results for %s: %s\n', base, ME_plot.message);
                    end
                else
                     fprintf('\nWarning: Original image %s not found for display.\n', imgPath);
                end
                if ~isempty(softMap)
                    displayFeatureMaps(outDir, base);
                end
            end
        catch ME_outer
            fprintf('\nError processing sample %d (%s): %s\n', k, pNames{k}, ME_outer.message);
            continue;
        end
    end
    fprintf('\nProcessing complete for %s.\n', outDir);
    validF1 = f1Scores(~isnan(f1Scores));
    if ~isempty(validF1)
        meanF1 = mean(validF1);
        datasetName = extractAfter(featDir, filesep);
        fprintf('Mean F1 Score for %s: %.4f\n', datasetName, meanF1);
    else
         fprintf('No valid F1 scores calculated for %s.\n', outDir);
    end
end

function f1 = calculateF1(pred, gt)
    pred = pred(:); gt = gt(:);
    tp = sum(pred & gt); fp = sum(pred & ~gt); fn = sum(~pred & gt);
    prec = tp / (tp + fp + eps); rec = tp / (tp + fn + eps);
    f1 = 2 * prec * rec / (prec + rec + eps);
end


function saveFeatureMaps(primaryNet, postNet, primaryInput4D, postInput4D, outDir, baseName)
    % Primary CNN layers
    primaryLayerNames = {'relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'sigmoid'};
    
    baseFeatureDir = fullfile(outDir, [baseName '_CNN_FeatureMaps']);
    if ~exist(baseFeatureDir, 'dir'), mkdir(baseFeatureDir); end
    
    % Primary CNN Feature Maps
    primaryBaseDir = fullfile(baseFeatureDir, 'Primary_CNN');
    if ~exist(primaryBaseDir, 'dir'), mkdir(primaryBaseDir); end
    fprintf('  Extracting Primary CNN activations for %s... \n', baseName);
    for i = 1:length(primaryLayerNames)
        layerName = primaryLayerNames{i};
        layerDir = fullfile(primaryBaseDir, sprintf('%02d_%s', i, layerName));
        if ~exist(layerDir, 'dir'), mkdir(layerDir); end
        try
            acts = activations(primaryNet, primaryInput4D, layerName, 'OutputAs', 'channels');
            acts_sq = squeeze(acts);
            if ndims(acts_sq) == 2, numChannels = 1; else, numChannels = size(acts_sq, 3); end
            fprintf('    - Layer ''%s'' (%d channels)... \n', layerName, numChannels);
            parfor c = 1:numChannels
                if numChannels > 1, featureMap = acts_sq(:, :, c); else, featureMap = acts_sq; end
                featureMapNorm = mat2gray(featureMap);
                fileName = fullfile(layerDir, sprintf('channel_%03d.png', c));
                imwrite(uint8(featureMapNorm * 255), fileName);
            end
        catch ME
            fprintf('    - FAILED to extract/save from layer ''%s'': %s\n', layerName, ME.message);
        end
    end
    
    % --- WORKAROUND ---
    % The 'activations' call for postNet is bugged. We will skip it.
    fprintf('  Skipping Post-Processing CNN feature extraction due to internal MATLAB bug.\n');
    
    % % Post-Processing CNN layers (THIS BLOCK IS BUGGED)
    % postLayerNames = {
    %     'enc_relu1', 'enc_relu2', 'enc_relu3', 
    %     'bottleneck_relu', 
    %     'dec_relu4', 'dec_relu5', 'dec_relu6', 
    %     'sigmoid'
    % };
    % 
    % postBaseDir = fullfile(baseFeatureDir, 'Post_Processing_CNN');
    % if ~exist(postBaseDir, 'dir'), mkdir(postBaseDir); end
    % fprintf('  Extracting Post-Processing CNN activations for %s... \n', baseName);
    % for i = 1:length(postLayerNames)
    %     layerName = postLayerNames{i};
    %     layerDir = fullfile(postBaseDir, sprintf('%02d_%s', i, layerName));
    %     if ~exist(layerDir, 'dir'), mkdir(layerDir); end
    %     try
    %         acts = activations(postNet, postInput4D, layerName, 'OutputAs', 'channels');
    %         acts_sq = squeeze(acts);
    %         if ndims(acts_sq) == 2, numChannels = 1; else, numChannels = size(acts_sq, 3); end
    %         fprintf('    - Layer ''%s'' (%d channels)... \n', layerName, numChannels);
    %         parfor c = 1:numChannels
    %             if numChannels > 1, featureMap = acts_sq(:, :, c); else, featureMap = acts_sq; end
    %             featureMapNorm = mat2gray(featureMap);
    %             fileName = fullfile(layerDir, sprintf('channel_%03d.png', c));
    %             imwrite(uint8(featureMapNorm * 255), fileName);
    %         end
    %     catch ME
    %         fprintf('    - FAILED to extract/save from layer ''%s'': %s\n', layerName, ME.message);
    %     end
    % end
    % --- END OF WORKAROUND ---
   
    fprintf('  Feature maps saved to %s\n', baseFeatureDir);
end

function displayFeatureMaps(outDir, baseName)
    baseFeatureDir = fullfile(outDir, [baseName '_CNN_FeatureMaps']);
    if ~exist(baseFeatureDir, 'dir'), return; end
    netFolders = dir(baseFeatureDir);
    netFolders = netFolders([netFolders.isdir] & ~startsWith({netFolders.name}, '.'));
    if isempty(netFolders), return; end
    fprintf('Displaying saved feature maps for %s...\n', baseName);
    
    for net_idx = 1:length(netFolders)
        netName = netFolders(net_idx).name;
        netPath = fullfile(baseFeatureDir, netName);
        layerFolders = dir(netPath);
        layerFolders = layerFolders([layerFolders.isdir] & ~startsWith({layerFolders.name}, '.'));
        if isempty(layerFolders), continue; end
        
        % Create a separate figure for each network type
        figure('Name', sprintf('Feature Maps: %s - %s', baseName, netName), 'NumberTitle', 'off', 'WindowState', 'maximized');
        
        numLayers = length(layerFolders);
        plotRows = ceil(numLayers / 2);
        plotCols = 2;
        
        for i = 1:numLayers
            layerName = layerFolders(i).name;
            layerPath = fullfile(netPath, layerName);
            imgFiles = dir(fullfile(layerPath, 'channel_*.png'));
            if isempty(imgFiles), continue; end
            imgPaths = fullfile({imgFiles.folder}, {imgFiles.name});
            
            ax = subplot(plotRows, plotCols, i);
            try
                montage(imgPaths, 'Parent', ax, 'ThumbnailSize', [128 128]);
                title(ax, layerName, 'Interpreter', 'none');
            catch ME_montage
                 title(ax, sprintf('%s - Error', layerName), 'Interpreter', 'none');
                 fprintf('Warning: Could not display montage for %s: %s\n', layerName, ME_montage.message);
            end
        end
        sgtitle(sprintf('Intermediate CNN Activations - %s - %s', baseName, netName), 'Interpreter', 'none', 'FontSize', 14, 'FontWeight', 'bold');
    end
    drawnow; % Ensure all figures are rendered
end