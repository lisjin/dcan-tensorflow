addpath('../fm-prep');
IN_DIR = 'BBBC006_v1_labels/';
OUT_DIR_C = 'BBBC006_v1_contours';
OUT_DIR_S = 'BBBC006_v1_segments';
mkdir(OUT_DIR_C);
mkdir(OUT_DIR_S);

imgNames = getDirNames([IN_DIR 'mcf*.png']);
for i = 1:numel(imgNames)
    img = imread(char([IN_DIR imgNames{i}]));
    binImg = img > 0;
    imwrite(binImg, [OUT_DIR_S '/' imgNames{i}]);

    vals = unique(nonzeros(img));
    bwImg = zeros(size(img));
    for j = 1:length(vals)
        cur = (img == vals(j, 1));
        [bounds, ~] = bwboundaries(cur, 'noholes');
        indices = bounds{1};
        for k = 1:size(indices, 1)
            bwImg(indices(k, 1), indices(k, 2)) = 1;
        end
    end
    imwrite(bwImg, [OUT_DIR_C '/' imgNames{i}]);
end
