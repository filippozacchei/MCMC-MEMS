clc
clear all
close all

U_input_filename = 'U_testing_halfSine.mat';
C_output_filename = 'C_testing_HalfSine.csv';
U_output_filename = 'U_testing_HalfSine.csv';

%% Files

% Load input
load(U_input_filename)

% Open the CSV files in append modes
fileID_C = fopen(C_output_filename, 'a');
fileID_U = fopen(U_output_filename, 'a');

fprintf(fileID_C, '# %s,%s,%s,%s,%s,%s, \n', "overetch", "offset", "thickness", "ts", "tf", "Values");
fprintf(fileID_U, '# %s,%s,%s,%s,%s,%s, \n', "overetch", "offset", "thickness", "ts", "tf", "Values");
      
% Define the capacity function
capacity = @(x, offset, thickness, overetch) 10 * 8.854e-12 * thickness * 1e-6 * (101 - 2 * overetch) * (1 ./ (1.2 + 2 * overetch - offset - x) ...
    - 1./ (1.2 + 2 * overetch + offset + x));

%% Generate Files

currentSimulation=1;
totalSimulations=size(X_dataset,1);

% Generate the capacity values for each parameter combination
for i = currentSimulation:totalSimulations
    i
    cap = X_dataset{i};
    overetch = cap.Overetch;
    offset = cap.Offset;
    thickness = cap.Thickness;
    ts = cap.Time(2)-cap.Time(1);
    tf = cap.Time(end);
    capacity_values = capacity(1e6*cap.Displacement, offset, thickness, overetch);
    plot(0:ts:tf,capacity_values)
    hold on
    
    % Write the line to the CapacityDataset.csv file
    fprintf(fileID_U, '%f,%f,%f,%f,%f', overetch, offset, thickness, ts, tf);
    fprintf(fileID_C, '%f,%f,%f,%f,%f', overetch, offset, thickness, ts, tf);

    % Add the vector values to the line
    for m = 1:numel(cap.Displacement)
        fprintf(fileID_U, ',%.21f', cap.Displacement(m));
        fprintf(fileID_C, ',%.27f', capacity_values(m));
        pause(0.0001)
    end

    % Add a newline character at the end of the line
    fprintf(fileID_U, '\n');
    fprintf(fileID_C, '\n');
end

% Close the Capacity CSV file
fclose(fileID_U);