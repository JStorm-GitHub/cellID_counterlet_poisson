% ------------------------------------------------------------------------
% Function: POISSON_COUNT
% ------------------------------------------------------------------------
% Usage 1: [Y IMG] = POISSON_COUNT( X, MN, MX )
% 
% Input parameters:
% X = Input image
% MN = Minimum Poisson parameter in the output; default = minimum pixel
%      value of X (See Description)
% MX = Maximum Poisson parameter in the output; default = minimum pixel
%      value of X (See Description)
%  
% Output parameters:
% Y = A Poisson noisy version of X
% IMG = Input image X scaled to [MN MX]
% 
% Description: Returns a Poisson noisy realization of X, by first scaling
% X to [MN MX]; i.e., the minimum value pixel is mapped to MN and maximum
% value is set to MX, before computing the Poisson count image.
% ------------------------------------------------------------------------
% Author: Sandeep Palakkal (sandeep.dion@gmail.com)
% Affiliation: Indian Institute of Technology Madras
% Created on: Feb 22, 2011 
% Modified on: Feb 22, 2011
% ------------------------------------------------------------------------

function [y img]= poisson_count( x, mn, mx )

if nargin == 3
  x = x - min( x(:) );
  x = x ./ max(x(:));
  img = mn + x * (mx-mn);
else
  img = x;
end
y = poissrnd( img );