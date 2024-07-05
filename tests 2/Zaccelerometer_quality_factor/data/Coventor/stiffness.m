function [kTotal] = stiffness(w, h, E, l1, l2, o)
% STIFFNESS Calculates the total stiffness of folded beams in a MEMS accelerometer.
%
% Calculation related to GAXL98 structure
% 
% Inputs:
%   w - Beam width
%   h - Beam thickness (unused in this version)
%   E - Young's modulus of the beam material
%   l1 - Length of the longer beams
%   l2 - Length of the shorter beams
%   o - Overetch affecting the effective width of the beam
%   t - Thickness of the beam
%
% Output:
%   kTotal - Total stiffness of the folded beam structure

    effectiveWidth = w - 2 * o; 
    J = (1/12) * h * effectiveWidth^3; 

    % Stiffness of individual beams based on their length
    k1 = 12 * E * J / ((l1)^3) / 4; 
    k2 = 12 * E * J / ((l2)^3) / 2; 

    kTotal = 2 / (1/k1 + 1/k2);
end
