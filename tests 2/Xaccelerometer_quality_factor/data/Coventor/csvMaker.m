clc
clear all
close all

U_input_filename = 'training_qFactor_all.mat';
C_output_filename = 'C_training_qFactor.csv';
load(U_input_filename)

%% Configuration I

% Load input

% Open the CSV files in append modes
fileID_C = fopen(C_output_filename, 'a');

fprintf(fileID_C, '# %s,%s,%s,%s,%s,%s,%s, \n', "overetch", "offset", "thickness", "qFactor", "ts", "tf", "Values");

currentSimulation=1;
totalSimulations=800;

% Generate the capacity values for each parameter combination
for i = currentSimulation:totalSimulations
    i
    cap = C_dataset{i};
    overetch = cap.Overetch;
    offset = cap.Offset;
    thickness = cap.Thickness;
    qFactor = S_dataset.qFactor(i);
    ts = cap.Time(2)-cap.Time(1);
    tf = cap.Time(end);
    capacity_values = cap.Displacement;

    plot(0:ts:tf,1e15*capacity_values,'.-')
    hold on
    
    % Write the line to the CapacityDataset.csv file
    fprintf(fileID_C, '%f,%f,%f,%f,%f,%f', overetch, offset, thickness, qFactor, ts, tf);

    % Add the vector values to the line
    for m = 1:numel(cap.Displacement)
        fprintf(fileID_C, ',%.27f', capacity_values(m));
    end

    % Add a newline character at the end of the line
    fprintf(fileID_C, '\n');
end

% Close the Capacity CSV file
fclose(fileID_C);

xlabel("t [s]", "Fontsize", 14)
ylabel("{\Delta}C(t) [fF]", "Fontsize", 14)
grid on 

%% Configuration II

S_output_filename = 'S_training_tris.csv';

currentSimulation=1;
totalSimulations=900;

% Open the CSV files in append modes
fileID_S = fopen(S_output_filename, 'a');

fprintf(fileID_S, '# %s,%s,%s,%s \n', "overetch", "offset", "thickness", "sensitivity");

% Generate the capacity values for each parameter combination
for i = currentSimulation:totalSimulations
    disp(i)
    S = S_dataset.dC_1g(i);
    overetch = S_dataset.overetch(i);
    offset = S_dataset.offset(i);
    thickness = S_dataset.thickness(i);
    
    % Write the line to the CapacityDataset.csv file
    fprintf(fileID_S, '%f,%f,%f,%f', overetch, offset, thickness, S);

    % Add a newline character at the end of the line
    fprintf(fileID_S, '\n');

    pause(0.0001)
end

% Close the Capacity CSV file
fclose(fileID_S);
