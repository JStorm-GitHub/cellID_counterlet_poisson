% ------------------------------------------------------------------------
% Function: PURELET for Poisson image denoising with cycle spinning
% ------------------------------------------------------------------------
% Usage 1: XHAT = CSPIN_PURELET( X, LET_ID, J, nSpin )
%
% Input parameters:
% X = Poisson noisy input image 
% LET_ID = LET ID; should be 0, 1, or 2. See [1].
% J = No. of Haar wavelet scales.
% nSpin = No. of shifts. The first one is [0 0] by default. Remaining
%         nSpin-1 shifts are randomly computed using rand function.
%  
% Output parameters:
% XHAT = Estimated image
% 
% Description: Denoises a Poisson-count noisy image using PURELET with
% cycle spinning
% ------------------------------------------------------------------------
% References:
% [1] F. Luisier, C. Vonesch, T. Blu, M. Unser, "Fast Interscale Wavelet
%     Denoising of Poisson-corrupted Images", Signal Processing, vol. 90,
%     no. 2, pp. 415-427, February 2010.
% ------------------------------------------------------------------------
% Author: Sandeep Palakkal (sandeep.dion@gmail.com)
% Affiliation: Indian Institute of Technology Madras
% Created on: Feb 11, 2011
% Modified on: Mar 19, 2011
% ------------------------------------------------------------------------

function y = cspin_purelet(  x, let_id, J, nSpin )

[m n] = size( x );
shifts = round( rand(nSpin-1,2).*repmat( [m n], nSpin-1, 1 ) );
shifts = [0 0; shifts];
y = zeros( m, n );
for ii = 1:nSpin
  y1 = purelet( circshift( x, shifts(ii,:) ), let_id, J );
  y = y + circshift( y1, -shifts(ii,:) );
end
y = y/nSpin;