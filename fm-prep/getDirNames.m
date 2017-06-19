function [dirNames] = getDirNames(wildcard)
% Get all directory names within directory matching wildcard.

dirs = dir(wildcard);
dirNames = {dirs.name};
end
