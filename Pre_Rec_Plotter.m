% --- Add F1 score calculation and storage ---



% --- Configuration ---
predMaskFolder = 'Validation-Dataset\Validation-Dataset\Masks'; % ADJUST FOLDER NAME
predMaskExtension = '*.png'; % ADJUST EXTENSION
gtMaskFolder = 'Validation-Dataset\Validation-Dataset\Masks-Ideal';   % .bmp or .png
gtMaskExtension = '*.bmp'; % ADJUST EXTENSION
outputFolder = 'Validation-Dataset\Validation-Dataset\PR_F1_Plot'; % Updated folder name

% =========================================================================
% --- Setup ---
% =========================================================================
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
    fprintf('Created output folder: %s\n', outputFolder);
end

% --- Get File Lists ---
predMaskFiles = dir(fullfile(predMaskFolder, predMaskExtension));
gtMaskFiles = dir(fullfile(gtMaskFolder, gtMaskExtension));
if isempty(predMaskFiles), error('No prediction mask files found in "%s".', predMaskFolder); end
if isempty(gtMaskFiles), error('No ground truth mask files found in "%s".', gtMaskFolder); end
predMaskNames = {predMaskFiles.name};
gtMaskNames = {gtMaskFiles.name};

% --- Match Files ---
[~, predBasenames, ~] = cellfun(@fileparts, predMaskNames, 'UniformOutput', false);
[~, gtBasenames, ~] = cellfun(@fileparts, gtMaskNames, 'UniformOutput', false);
[commonNames, ia, ib] = intersect(predBasenames, gtBasenames);
if isempty(commonNames), error('No matching prediction-ground truth pairs found!'); end
predMaskNames = predMaskNames(ia);
gtMaskNames = gtMaskNames(ib);
numPairs = length(predMaskNames);
fprintf('Found %d matching prediction-ground truth pairs.\n', numPairs);

% --- Store Results ---
precisionPoints = zeros(numPairs, 1);
recallPoints = zeros(numPairs, 1);
f1Scores = zeros(numPairs, 1); % ADDED: Array for F1 scores
filenames = cell(numPairs, 1);
validResult = true(numPairs, 1);

% =========================================================================
% --- Process Each Pair ---
% =========================================================================
fprintf('Calculating Precision, Recall, and F1-Score for each pair...\n');

for i = 1:numPairs
    filenames{i} = commonNames{i};

    try
        % --- Load Predicted Binary Mask ---
        predMaskPath = fullfile(predMaskFolder, predMaskNames{i});
        predMask = imread(predMaskPath);
        predMask = im2double(predMask);
        if ndims(predMask) == 3, predMask = predMask(:,:,1); end
        predMaskLogical = predMask > 0.5; % Hand=1

        % --- Load Ground Truth Mask ---
        gtMaskPath = fullfile(gtMaskFolder, gtMaskNames{i});
        gtMask = imread(gtMaskPath);
        gtMask = im2double(gtMask);
        if ndims(gtMask) == 3, gtMask = gtMask(:,:,1); end
        gtMaskLogical = gtMask > 0.5; % Hand=1

        % --- Check Size Compatibility ---
        if ~isequal(size(predMaskLogical), size(gtMaskLogical))
            warning('Size mismatch for %s. Skipping.', commonNames{i});
            validResult(i) = false;
            continue;
        end

        % --- Calculate Metrics for this pair ---
        predFlat = predMaskLogical(:);
        gtFlat = gtMaskLogical(:);

        tp = sum(predFlat & gtFlat);
        fp = sum(predFlat & ~gtFlat);
        fn = sum(~predFlat & gtFlat);

        % Calculate Precision (P)
        if (tp + fp) == 0
            P = 0;
        else
            P = tp / (tp + fp);
        end
        precisionPoints(i) = P;

        % Calculate Recall (R)
        if (tp + fn) == 0
            R = 0;
        else
            R = tp / (tp + fn);
        end
        recallPoints(i) = R;

        % --- Calculate F1 Score --- % ADDED SECTION
        if (P + R) == 0
            F1 = 0; % Avoid division by zero if P and R are both 0
        else
            F1 = 2 * (P * R) / (P + R);
        end
        f1Scores(i) = F1; % Store F1 score
        % --- END ADDED SECTION ---

    catch ME
        warning('Error processing pair %d (%s): %s. Skipping.', i, commonNames{i}, ME.message);
        validResult(i) = false;
    end
end % --- End Loop ---

% --- Filter out invalid results ---
precisionPoints = precisionPoints(validResult);
recallPoints = recallPoints(validResult);
f1Scores = f1Scores(validResult); % ADDED: Filter F1 scores
filenames = filenames(validResult);
numValid = length(precisionPoints);

if numValid == 0
    error('No valid Precision/Recall/F1 points were calculated.');
end
fprintf('Calculated P/R/F1 points for %d pairs.\n', numValid);

% =========================================================================
% --- Plot Precision-Recall Scatter Plot ---
% =========================================================================
fprintf('Generating PR scatter plot...\n');
hFig = figure;
scatter(recallPoints, precisionPoints, 36, f1Scores, 'filled'); % Use F1 score for color
colormap('jet'); % Or 'jet', 'parula', etc.
cb = colorbar; % Add colorbar
ylabel(cb, 'F1 Score'); % Label colorbar
xlabel('Recall (Sensitivity)');
ylabel('Precision');
title(sprintf('Precision-Recall Points for %d Samples (Color = F1 Score)', numValid));
grid on;
axis([0 1 0 1.05]);

% --- Optional: Add Average Point ---
avgRecall = mean(recallPoints);
avgPrecision = mean(precisionPoints);
avgF1 = mean(f1Scores); % ADDED: Calculate average F1
hold on;
scatter(avgRecall, avgPrecision, 100, 'r', 'p', 'filled');
hold off;
legend('Individual Masks', sprintf('Average (R=%.3f, P=%.3f, F1=%.3f)', avgRecall, avgPrecision, avgF1), 'Location', 'best');
fprintf('Average Recall: %.4f\n', avgRecall);
fprintf('Average Precision: %.4f\n', avgPrecision);
fprintf('Average F1 Score: %.4f\n', avgF1); % ADDED: Display average F1

% --- Save the plot ---
plotFilename = fullfile(outputFolder, 'Precision_Recall_F1_Scatter_Plot.png');
try
    saveas(hFig, plotFilename);
    fprintf('PR-F1 scatter plot saved to: %s\n', plotFilename);
catch ME_save
    warning('Could not save plot %s: %s', plotFilename, ME_save.message);
end
% close(hFig); % Optional

% --- Save Raw Data ---
% ADDED: f1Scores to table
resultsTable = table(filenames, recallPoints, precisionPoints, f1Scores, 'VariableNames', {'Filename', 'Recall', 'Precision', 'F1_Score'});
resultsFilename = fullfile(outputFolder, 'precision_recall_f1_points.csv');
try
    writetable(resultsTable, resultsFilename);
    fprintf('Individual P/R/F1 points saved to: %s\n', resultsFilename);
catch ME_csv
    warning('Could not save P/R/F1 points CSV: %s', ME_csv.message);
end

fprintf('--- PR/F1 Plot Generation Complete ---\n');