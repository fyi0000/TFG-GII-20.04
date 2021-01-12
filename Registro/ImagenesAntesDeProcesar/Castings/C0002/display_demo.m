% This example display the defects of image 16
GT = load('ground_truth.txt');
I = imread('C0002_0016.png');
imshow(I,[])
hold on
ii = find(GT(:,1) == 16);
n = length(ii);
for i = 1:n
    t = GT(ii(i),2:end);
    plot(t([1 2 2 1 1]),t([3 3 4 4  3]),'r');
end
