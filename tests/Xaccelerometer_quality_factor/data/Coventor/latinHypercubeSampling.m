% Latin Hypercube Sampling
function samples = latinHypercubeSampling(dimensions, intervals, numSamples)
    % dimensions: Number of dimensions or parameters
    % intervals: Continuous intervals for each parameter [min1, max1; min2, max2; ...]
    % numSamples: Number of samples to generate
    
    % Generate random samples in the unit hypercube
    unitSamples = lhsdesign(numSamples, dimensions);
    
    % Map the unit samples to the specified intervals
    samples = zeros(numSamples, dimensions);
    for i = 1:dimensions
        samples(:, i) = intervals(i, 1) + unitSamples(:, i) * (intervals(i, 2) - intervals(i, 1));
    end
end