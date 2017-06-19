function topIndices = fmArgmax(fmVals, imgNamesAll)
% Use argmax of each column of fmVals (13x768) to get z-index with best
% focus measure. Save this index along with image name in topIndices.

IDX_OFFSET = 10;
numImgs = size(imgNamesAll{1}, 2);

topIndices(numImgs).ind = 0;
topIndices(numImgs).imgName = '';
for k = 1:numImgs
    [~, argmax] = max(fmVals(:, k));
    imgInd = argmax + IDX_OFFSET;

    topIndices(1, k).ind = imgInd;
    topIndices(1, k).imgName = imgNamesAll{argmax}{k};
end
end
