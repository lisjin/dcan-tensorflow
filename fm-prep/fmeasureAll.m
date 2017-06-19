function [fmVals, imgNamesAll] = fmeasureAll()
% Use fmeasure function to detect focus measures of all 13x768 Hoescht
% images. Save these in fmVals, and all image names in imgNamesAll.

NUM_IMAGES = 1536;
dirs = getDirNames('BBBC*images*');
fmVals = ones(numel(dirs), NUM_IMAGES);
imgNamesAll = cell(1, numel(dirs));

addpath('../../Documents/MATLAB/fmeasure/');
disp('Processing directory...');
for i = 1:numel(dirs)
    disp(i);

    % Collect all image names (e.g., those prefixed with 'mcf')
    wildcard = char([dirs{i}, '/mcf*']);
    imgNames = getDirNames(wildcard);

    % Only process image names containing 'w1' (indicates Hoechst images)
    tmp = cellfun(@(x) strfind(x, 'w1'), imgNames, 'UniformOutput', false);
    imgNames = imgNames(~cellfun('isempty', tmp));
    imgNamesAll{i} = imgNames;
    for j = 1:numel(imgNames)
        img = imread(char([dirs{i} '/' imgNames{j}]));
        fmVals(i, j) = fmeasure(img, 'WAVS');
    end
end
end
