%import cyclic color-scheme%
phasemap();
phasebar('rad');

r = 0.5; %r
n = 10;
ct = r/10;
num = 21; %num of vertices
vor = 1.0; %voronooi length

%specifications below: x_pos for postition, S for color%
T = readtable('out.csv');
pos_slice = T(:,4:((num*3) + 3));
thet_slice = T(:,(num*3 + 4):end);

pos_Arr = table2array(pos_slice);
thet_Arr = table2array(thet_slice);

h = figure;
phasemap();
phasebar('rad');
filename = 'testAnimated_thin.gif';
si = size(pos_Arr);
for i = 1:1:si(1)
    if mod(i,1000) == 1 
        x_pos = reshape(pos_Arr(i,:),[3,num]);
        S = phasewrap(thet_Arr(i,:));
        colorplot(x_pos,r,n,ct,S);
        xlim([-1,num * vor]);
        ylim([-10,10]);
        zlim([-10,10]);
        drawnow;
        
        frame = getframe(h);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);

        if i == 1
            imwrite(imind,cm,filename,'gif','Loopcount',inf);
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append');
        end %cond
    end %cond
end 

function [x,y,z,C]=colorplot(curve,r,n,ct,S)
% Usage: same as above but this gives colours the rod according to a 
% scalar field S along the curve.

% Arguments:
% curve: [3,N] vector of curve data
% r      the radius of the tube
% n      number of points to use on circumference. Defaults to 8
% ct     threshold for collapsing points. Defaults to r/2 

  if nargin<3 || isempty(n), n=8;
     if nargin<2, error('Give at least curve and radius');
     end
  end
  if size(curve,1)~=3
    error('Malformed curve: should be [3,N]');
  end
  if nargin<4 || isempty(ct)
    ct=0.5*r;
  end

  
  %Collapse points within 0.5 r of each other
  npoints=1;
  for k=2:(size(curve,2)-1)
    if norm(curve(:,k)-curve(:,npoints))>ct
      npoints=npoints+1;
      curve(:,npoints)=curve(:,k);
    end
  end
  %Always include endpoint
  if norm(curve(:,end)-curve(:,npoints))>0
    npoints=npoints+1;
    curve(:,npoints)=curve(:,end);
  end

  %deltavecs: average for internal points.
  %           first strecth for endpoitns.
  dv=curve(:,[2:end,end])-curve(:,[1,1:end-1]);

  %make nvec not parallel to dv(:,1)
  nvec=zeros(3,1);
  [~,idx]=min(abs(dv(:,1))); nvec(idx)=1;

  xyz=zeros(3,n+1,npoints+2);
  Col=zeros(3,n+1,npoints+2); 

  %precalculate cos and sing factors:
  cfact=repmat(cos(linspace(0,2*pi,n+1)),[3,1]);
  sfact=repmat(sin(linspace(0,2*pi,n+1)),[3,1]);
  
  %Main loop: propagate the normal (nvec) along the tube
  for k=1:npoints
    convec=cross(nvec,dv(:,k));
    convec=convec./norm(convec);
    nvec=cross(dv(:,k),convec);
    nvec=nvec./norm(nvec);
    %update xyz:
    xyz(:,:,k+1)=repmat(curve(:,k),[1,n+1])+...
        cfact.*repmat(r*nvec,[1,n+1])...
        +sfact.*repmat(r*convec,[1,n+1]);
    Col(:,:,k)=S(k);
  end
  %finally, cap the ends:
  Col(:,:,end)=S(end);
  xyz(:,:,1)=repmat(curve(:,1),[1,n+1]);
  xyz(:,:,end)=repmat(curve(:,end),[1,n+1]);
  %,extract results:
  x=squeeze(xyz(1,:,:));
  y=squeeze(xyz(2,:,:));
  z=squeeze(xyz(3,:,:));
  Ct=squeeze(Col(1,:,:));
  C=Ct;
  %... and plot:
  if nargout<3, surf(x,y,z,C); end
end