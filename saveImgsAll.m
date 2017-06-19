[fmVals, imgNamesAll] = fmeasureAll();
topIndices = fmArgmax(fmVals, imgNamesAll);
saveImgs(topIndices);
