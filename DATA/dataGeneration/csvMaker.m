clc
clear all
close all

load Capacity.mat
load U.mat
load V.mat

% Open the Capacity CSV file in append mode
fileID_C = fopen('CapacityDataset.csv', 'a');
% Open the U CSV file in append mode
fileID_U = fopen('UDataset.csv', 'a');
% Open the V CSV file in append mode
fileID_V = fopen('VDataset.csv', 'a');

fprintf(fileID_C, '%s,%s,%s,%s,%s \n', "amplitudeX", "amplitudeY", "amplitudeZ", "overetch", "Values");
fprintf(fileID_U, '%s,%s,%s,%s,%s \n', "amplitudeX", "amplitudeY", "amplitudeZ", "overetch", "Values");
fprintf(fileID_V, '%s,%s,%s,%s,%s \n', "amplitudeX", "amplitudeY", "amplitudeZ", "overetch", "Values");
                

% Generate the capacity values for each parameter combination
for i = 1:size(Capacity_dataset,1)
    i
    cap = Capacity_dataset{i};
    amplitudeX_val = cap.AmplitudeX;
    amplitudeY_val = cap.AmplitudeY;
    amplitudeZ_val = cap.AmplitudeZ;
    TX_val = cap.T_X(i);
    TY_val = cap.T_Y(i);
    TZ_val = cap.T_Z(i);
    overetch = cap.Overetch;
    
    % Write the line to the CapacityDataset.csv file
    fprintf(fileID_C, '%f,%f,%f,%f,%f,%f,%f', amplitudeX_val, amplitudeY_val, amplitudeZ_val, TX_val, TY_val, TZ_val, overetch);
    fprintf(fileID_U, '%f,%f,%f,%f,%f,%f,%f', amplitudeX_val, amplitudeY_val, amplitudeZ_val, TX_val, TY_val, TZ_val, overetch);
    fprintf(fileID_V, '%f,%f,%f,%f,%f,%f,%f', amplitudeX_val, amplitudeY_val, amplitudeZ_val, TX_val, TY_val, TZ_val, overetch);
    

    % Get the capacity vector values for the current combination
    U_values = U_dataset{i};
    V_values = V_dataset{i};
    
    % Add the vector values to the line
    for m = 1:numel(cap.Capacity)
        fprintf(fileID_C, ',%.27f', cap.Capacity(m));
        fprintf(fileID_U, ',%.21f', U_values.Capacity(m));
        fprintf(fileID_V, ',%.21f', V_values.Capacity(m));
    end

    % Add a newline character at the end of the line
    fprintf(fileID_C, '\n');
    fprintf(fileID_U, '\n');
    fprintf(fileID_V, '\n');
end

% Close the Capacity CSV file
fclose(fileID_C);
fclose(fileID_U);