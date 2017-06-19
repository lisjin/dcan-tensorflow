function topIndices = fmArgmax(fmVals, imgNamesAll)
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
