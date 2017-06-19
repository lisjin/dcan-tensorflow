function [dirNames] = getDirNames(wildcard)
dirs = dir(wildcard);
dirNames = {dirs.name};
end
