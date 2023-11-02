classdef MEMS_Table
    
    properties
        Overetch % Overetch value [Scalar]
        Offset % Mechanical Offset Value [Scalar]
        Thickness % Epipoly Thickness Value [Scalar]
        Time % Time steps [Vector]
        Displacement % X Displacement Value [Vector]
    end
    
    methods
        function obj = MEMS_Table(Overetch_, Offset_, Thickness_, Time_, Displacement_)
            % Constructor
            obj.Overetch = Overetch_;
            obj.Offset = Offset_;
            obj.Thickness = Thickness_;
            obj.Time = Time_;
            obj.Displacement = Displacement_;
        end

    end
end
