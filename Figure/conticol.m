function [colmat] = conticol(color, n, varargin)
%%
%  Usage: Suppose for red color conticol([1, 0, 0], 10, 'col2dark') or 
%   conticol([1, 0, 0], 10, 'col2fade')
%  colmat: [n x 3] matrix containing the continous range of color
%  degradation from white to red

colmat = color;
if nargin == 3
    gradupto = 0.8;
else
    gradupto = varargin{2};
end
%disp([fromgrad]);
if strcmp(varargin{1}, 'col2dark')
    colmat = zeros(n, 3);
    
    idx = find(color == 1); decr = gradupto * ones(size(idx));
    if length(idx) == 0
        error('Cannot convert for opt blacktocol for black');
    end
    for i = [1 : n]
        colmat(i, idx) = color(idx) - decr * (i - 1) / n;
    end
    
elseif strcmp(varargin{1}, 'col2fade')
    colmat = ones(n, 3);
    
    idx = find(color == 0); incr = gradupto * ones(size(idx));
    if length(idx) == 0
        error('Cannot convert for opt whitetocol for white');
    end
    for i = [1 : n]
        colmat(i, idx) = color(idx) + incr * (i - 1) / n;
    end
end