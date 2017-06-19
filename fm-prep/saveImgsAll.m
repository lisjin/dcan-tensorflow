% Script to find and save the optimal focal plane out of z-stack of 13
% image stacks (z = 11-23).
% Output directory: 'BBBC006_v1_focused/'

[fmVals, imgNamesAll] = fmeasureAll();
topIndices = fmArgmax(fmVals, imgNamesAll);
saveImgs(topIndices);
