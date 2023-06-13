%In order to avoid possible errors due to changes in the state of the MATLAB 
%workspace, it is recommended to clean the workspace in the beginning of each
%script using clearvars command.
%clearvars
clear all
close all
clc %clears the MATLAB command window and closes all open figures 

h = cov.memsplus.Simulation('Accelerometer.3dsch');
% What are the inputs, their units, and type choices
h.ExposedConnectors.printWithProperties()

%%
g = 9.81;
ts = 1e-5;                             % Acquisition Time Step
tf = 2e-3;                              % Final Time
T = 1e-3;

% Function to generate sin impulse with varying amplitude
generateSinImpulse = @(t, A) A * sin(t/T*pi) .* (t < T);

% Parameters for varying amplitudes (0 to 10g)
amplitudeX = linspace(0, 10 * g, 11);
amplitudeY = linspace(0, 10 * g, 3);
amplitudeZ = linspace(0, 0 * g, 3);

% Set Overetch
overetch_values = linspace(0.3, 0.5, 5 );      % Modify overetch value as desired

% Cell array to store CapacityTable objects for each simulation
Capacity_dataset = cell(numel(amplitudeX), numel(amplitudeY), numel(amplitudeZ), numel(overetch_values));
U_dataset = cell(numel(amplitudeX), numel(amplitudeY), numel(amplitudeZ), numel(overetch_values));
V_dataset = cell(numel(amplitudeX), numel(amplitudeY), numel(amplitudeZ), numel(overetch_values));

% Generate the progress bar
totalSimulations = numel(amplitudeX) * numel(amplitudeY) * numel(amplitudeZ) * numel(overetch_values);
currentSimulation = 0;

% Generate the capacity values for each parameter combination
for i = 1:numel(amplitudeX)
    for j = 1:numel(amplitudeY)
        for k = 1:numel(amplitudeZ)
            for l = 1:numel(overetch_values)
                amplitudeX_val = amplitudeX(i);
                amplitudeY_val = amplitudeY(j);
                amplitudeZ_val = amplitudeZ(k);
                overetch = overetch_values(l);
                
                % Set Overetch
                h.Variables.Overetch = overetch;

                % Create a CapacityTable object

                
                % Set damping
                h.Variables.alpha = 31400;
                
                % Configure the alternative inputs based on the current case
                tax = @(t) generateSinImpulse(t, amplitudeX_val);
                tay = @(t) generateSinImpulse(t, amplitudeY_val);
                taz = @(t) generateSinImpulse(t, amplitudeZ_val);
                
                % Create and compute a DC solution
                dc = h.Analyses.add('DC');
                dc.Verbose = 0;
                dc.Properties.ExposedConnectorsValues.ProofMass = 1.0;
                dc.Properties.ExposedConnectorsValues.tax = 0*g;
                dc.Properties.ExposedConnectorsValues.tay = 0*g;
                dc.Properties.ExposedConnectorsValues.taz = 0*g;
                dc.run;
                
                % Create and set up a transient analysis
                tran = dc.add('Transient');
                tran.Properties.TimeSpan.Values = 0:ts:tf;
                tran.Properties.Tolerances.RelativeTolerance = 1e-6;
                tran.Properties.ExposedConnectorsValues.tax = tax;
                tran.Properties.ExposedConnectorsValues.tay = tay;
                tran.Properties.ExposedConnectorsValues.taz = taz;
                tran.runtimeSettings.Plotter.doPlotting = false;
                tran.run();
                
                % Retrieve the capacity values
                capacity = tran.Result.Outputs.Cap7.Values + ...
                           tran.Result.Outputs.Cap8.Values + ...
                           tran.Result.Outputs.Cap9.Values + ...
                           tran.Result.Outputs.Cap10.Values + ...
                           tran.Result.Outputs.Cap11.Values + ...
                           tran.Result.Outputs.Cap12.Values + ...
                           tran.Result.Outputs.Cap13.Values + ...
                           tran.Result.Outputs.Cap14.Values + ...
                           tran.Result.Outputs.Cap15.Values + ...
                           tran.Result.Outputs.Cap16.Values + ...
                           tran.Result.Outputs.Cap17.Values + ...
                           tran.Result.Outputs.Cap18.Values + ...
                           tran.Result.Outputs.Cap19.Values + ...
                           tran.Result.Outputs.Cap20.Values + ...
                           tran.Result.Outputs.Cap21.Values + ...
                           tran.Result.Outputs.Cap22.Values + ...
                           tran.Result.Outputs.Cap23.Values + ...
                           tran.Result.Outputs.Cap24.Values + ...
                           tran.Result.Outputs.Cap25.Values + ...
                           tran.Result.Outputs.Cap26.Values ;
                
                % Add the capacity values to the CapacityTable
                timeSteps = 0:ts:tf;
                Capacity_table = CapacityTable(timeSteps, amplitudeX_val, amplitudeY_val, amplitudeZ_val, overetch,capacity);
                U_table = CapacityTable(timeSteps, amplitudeX_val, amplitudeY_val, amplitudeZ_val, overetch, tran.Result.States.ProofMass_x.Values);
                V_table = CapacityTable(timeSteps, amplitudeX_val, amplitudeY_val, amplitudeZ_val, overetch, tran.Result.States.ProofMass_y.Values);
                Capacity_dataset{i, j, k, l} = Capacity_table;
                U_dataset{i, j, k, l} = U_table;
                V_dataset{i, j, k, l} = V_table;

                
                % Display progress update
                currentSimulation = currentSimulation + 1;
                fprintf('Simulation %d of %d\n', currentSimulation, totalSimulations);
                % Update loading bar
                progress = currentSimulation / totalSimulations;
                loadingBarWidth = 30;
                numCompleted = floor(loadingBarWidth * progress);
                numRemaining = loadingBarWidth - numCompleted;
                loadingBar = [repmat('#', 1, numCompleted), repmat('-', 1, numRemaining)];
                fprintf('[%s] %.1f%%\n', loadingBar, progress * 100);
            end
        end
    end
end


%%
g = 9.81;
ts = 1e-5;                             % Acquisition Time Step
tf = 2e-3;                              % Final Time
T = 1e-3;

% Function to generate sin impulse with varying amplitude
generateSinImpulse = @(t, A) A * sin(t/T*pi) .* (t < T);

% Parameters for varying amplitudes (0 to 10g)
amplitudeX = linspace(0, 10 * g, 11);
amplitudeY = linspace(0, 10 * g, 3);
amplitudeZ = linspace(0, 0 * g, 3);

% Set Overetch
overetch_values = linspace(0.3, 0.5, 5 );      % Modify overetch value as desired

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
for i = 1:numel(amplitudeX)
    for j = 1:numel(amplitudeY)
        for k = 1:numel(amplitudeZ)
            for l = 1:numel(overetch_values)
                amplitudeX_val = amplitudeX(i);
                amplitudeY_val = amplitudeY(j);
                amplitudeZ_val = amplitudeZ(k);
                overetch = overetch_values(l);
                
                % Write the line to the CapacityDataset.csv file
                fprintf(fileID_C, '%f,%f,%f,%f', amplitudeX_val, amplitudeY_val, amplitudeZ_val, overetch);
                fprintf(fileID_U, '%f,%f,%f,%f', amplitudeX_val, amplitudeY_val, amplitudeZ_val, overetch);
                fprintf(fileID_V, '%f,%f,%f,%f', amplitudeX_val, amplitudeY_val, amplitudeZ_val, overetch);
                

                % Get the capacity vector values for the current combination
                C_values = Capacity_dataset{i, j, k, l};
                U_values = U_dataset{i, j, k, l};
                V_values = V_dataset{i, j, k, l};
                
                % Add the vector values to the line
                for m = 1:numel(C_values.Capacity)
                    fprintf(fileID_C, ',%.27f', C_values.Capacity(m));
                    fprintf(fileID_U, ',%.21f', U_values.Capacity(m));
                    fprintf(fileID_V, ',%.21f', V_values.Capacity(m));
                end
     
                % Add a newline character at the end of the line
                fprintf(fileID_C, '\n');
                fprintf(fileID_U, '\n');
                fprintf(fileID_V, '\n');
            end
        end
    end
end

% Close the Capacity CSV file
fclose(fileID_C);
fclose(fileID_U);
fclose(fileID_V);


