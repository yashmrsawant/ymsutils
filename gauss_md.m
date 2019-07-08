
function [f] = gauss_md(X, Mu, Sigma)
%GAUSS_MD Summary of this function goes here
%   X : n x 1, Mu ; n x 1, Sigma : n x n
%   
    n = size(X, 1);
    f = 1 / (2 * pi) ^ (n / 2) * abs(1 / det(Sigma)) ^ 0.5 * ...
        exp(-1 * (X - Mu)' * inv(Sigma) * (X - Mu));
    
end

