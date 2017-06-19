function topIndices = fmeasureAll()
NUM_IMAGES = 1536;
dirs = getDirNames('BBBC*images*');
fmVals = ones(numel(dirs), NUM_IMAGES);

addpath('../../Documents/MATLAB/fmeasure/');
disp(['Processing directory...']);
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

topIndices = fmArgmax(fmVals, imgNamesAll);
save('topIndices', 'topIndices');
end
