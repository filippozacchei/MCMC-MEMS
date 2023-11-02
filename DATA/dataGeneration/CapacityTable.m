classdef CapacityTable
    
    properties
        Time        % Time steps
        AmplitudeX  % Amplitude for x-axis
        AmplitudeY  % Amplitude for y-axis
        AmplitudeZ  % Amplitude for z-axis
        Overetch    % Overetch value
        Capacity    % Capacity value
        T_X         % Amplitude for x-axis
        T_Y         % Amplitude for x-axis
        T_Z         % Amplitude for x-axis
    end
    
    methods
        function obj = CapacityTable(timeSteps, amplitudeX, amplitudeY, amplitudeZ, T_X, T_Y, T_Z, overetch, capacity)
            % Constructor
            obj.Time = timeSteps;
            obj.AmplitudeX = amplitudeX;
            obj.AmplitudeY = amplitudeY;
            obj.AmplitudeZ = amplitudeZ;
            obj.T_X = T_X;
            obj.T_Y = T_Y;
            obj.T_Z = T_Z;
            obj.Overetch = overetch;
            obj.Capacity = capacity;
        end

    end
end
