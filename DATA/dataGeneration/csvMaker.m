clc
clear all
close all

U_input_filename = 'training_dataset.mat';
C_output_filename = 'C_training.csv';

%% Configuration I

% Load input
load(U_input_filename)

% Open the CSV files in append modes
fileID_C = fopen(C_output_filename, 'a');

fprintf(fileID_C, '# %s,%s,%s,%s,%s,%s, \n', "overetch", "offset", "thickness", "ts", "tf", "Values");

currentSimulation=1;
totalSimulations=length(C1_dataset_training);

% Generate the capacity values for each parameter combination
for i = currentSimulation:totalSimulations
    i
    cap1 = C1_dataset_training{i};
    cap2 = C2_dataset_training{i};
    overetch = cap1.Overetch;
    offset = cap1.Offset;
    thickness = cap1.Thickness;
    ts = cap1.Time(2)-cap1.Time(1);
    tf = cap1.Time(end);

    capacity_values = cap1.Displacement - cap2.Displacement;
    plot(0:ts:tf,1e15*capacity_values,'.-')
    hold on
    
    % Write the line to the CapacityDataset.csv file
    fprintf(fileID_C, '%f,%f,%f,%f,%f', overetch, offset, thickness, ts, tf);

    % Add the vector values to the line
    for m = 1:numel(cap1.Displacement)
        fprintf(fileID_C, ',%.27f', capacity_values(m));
        pause(0.0001)
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

S_input_filename = 'S_training_dataset.mat';
S_output_filename = 'S_training.csv';

% Load input
load(S_input_filename)

currentSimulation=1;
totalSimulations=size(S_training_dataset,1);

% Open the CSV files in append modes
fileID_S = fopen(S_output_filename, 'a');

fprintf(fileID_S, '# %s,%s,%s,%s \n', "overetch", "offset", "thickness", "sensitivity");

% Generate the capacity values for each parameter combination
for i = currentSimulation:totalSimulations
    disp(i)
    S = S_training_dataset.dC_1g(i);
    overetch = S_training_dataset.overetch(i);
    offset = S_training_dataset.offset(i);
    thickness = S_training_dataset.thickness(i);
    
    % Write the line to the CapacityDataset.csv file
    fprintf(fileID_S, '%f,%f,%f,%f', overetch, offset, thickness, S);

    % Add a newline character at the end of the line
    fprintf(fileID_S, '\n');

    pause(0.0001)
end

% Close the Capacity CSV file
fclose(fileID_S);
