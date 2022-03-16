clear all; clc; close all


%% 

%get folder where the images are"
imageFolder = uigetdir

filePattern = fullfile(imageFolder, '*.png');
pngFiles = dir(filePattern);
for k = 1:length(pngFiles)
  baseFileName = pngFiles(k).name;
  fullFileName = fullfile(imageFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  
  imageArray = imread(fullFileName);
  imageArrayGS = rgb2gray(imageArray)
  edge = edge(imageArrayGS);
  
  [centers,radii] = imfindcircles(imageArray, [10 2000], 'ObjectPolarity','dark', 'Sensitivity',0.9)
  %imshow(imageArray);  % Display image.
  %imshow(edge);
  %h = viscircles(centers,radii);
  %drawnow; % Force display to update immediately.
  break
end
