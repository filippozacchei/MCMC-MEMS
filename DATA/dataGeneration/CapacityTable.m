classdef CapacityTable
    
    properties
        Time        % Time steps
        AmplitudeX  % Amplitude for x-axis
        AmplitudeY  % Amplitude for y-axis
        AmplitudeZ  % Amplitude for z-axis
        Overetch    % Overetch value
        Capacity    % Capacity value
    end
    
    methods
        function obj = CapacityTable(timeSteps, amplitudeX, amplitudeY, amplitudeZ, overetch, capacity)
            % Constructor
            obj.Time = timeSteps;
            obj.AmplitudeX = amplitudeX;
            obj.AmplitudeY = amplitudeY;
            obj.AmplitudeZ = amplitudeZ;
            obj.Overetch = overetch;
            obj.Capacity = capacity;
        end

    end
end
