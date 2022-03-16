%evalin('base','clear variables')

imageFolder = uigetdir

I = imread('C:\Users\JD\OneDrive\OneDrive - Texas A&M University\URS\Sabot Stripper Pictures\Black and White\0059.png')
I = rgb2gray(I);
edge1 = edge(I);
figure;
imshow(edge1)