function saveImgs(topIndices)
% Using topIndices (first column is z-index, second is image file name),
% save corresponding images in output directory.

NUM_IMGS = 768;
OUT_DIR = 'BBBC006_v1_focused';
mkdir(OUT_DIR);

for i = 1:NUM_IMGS
    wildcard = char(['BBBC006*' num2str(topIndices(1, i).ind)]);
    dirName = getDirNames(wildcard);

    source = char([dirName{1} '/' topIndices(1, i).imgName]);
    copyfile(source, OUT_DIR);
end
end
