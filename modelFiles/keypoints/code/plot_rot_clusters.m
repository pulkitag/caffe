close all;
isCheck = false;

if isCheck
	dat = load('data/test_clustering.mat');
	points  = dat.points;
	centers = dat.centers;
	assgn   = dat.assgn;

	N   = size(points,1);
	nCl = size(centers,1);

	%Plot the centers
	x = zeros(nCl,1);
	y = x;
	z = x;
	%quiver3(0,0,0,centers(1,1),centers(1,2),centers(1,3)); 

	colors = {'r','g','b'};
	for i=1:1:nCl
		idx = find(assgn==i-1);
		x = i*ones(length(idx),1);
		y = x;
		z = x;
		quiver3(x,y,z,points(idx,1),points(idx,2),points(idx,3),'color',colors{i});
		hold on;
		quiver3(0,0,0,centers(i,1),centers(i,2),centers(i,3),'color',colors{i},'LineWidth',3); 
	end
else
	dat = load('data/exprigid_lblkmedoids30_20_clusters.mat');
	centers = dat.centers;
	N       = size(centers,1);
	x       = zeros(N,1);
	y       = x;
	z       = x;
	quiver3(x,y,z,centers(:,1),centers(:,2),centers(:,3));
end

