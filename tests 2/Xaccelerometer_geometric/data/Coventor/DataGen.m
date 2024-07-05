%In order to avoid possible errors due to changes in the state of the MATLAB 
%workspace, it is recommended to clean the workspace in the beginning of each
%script using clearvars command.
%clearvars

clear all
clc %clears the MATLAB command window and closes all open figures 

warning('off','all')

%% Parameters
g = 9.81;
ts = 1e-5; % Acquisition Time Step 
tf = 1.5e-3; % Final Time
    
% Load data and set initial parameters
input_file = readtable("2023_10_04_GA98_twins_data.xlsx");
initialSimulation = 131;
totalSimulation = 500;

% Function to generate Input Voltage Profile 
amplitude_val = 1.8;
Tx_val = 0.4e-3;
VoltageProfile = @(t) 0.5*amplitude_val*(1+(sin((t/Tx_val-1/4)*2*pi))).*(t<2*Tx_val);
 
% Data Generation
num_training_samples = 500; % Number of saples to generate for training dataset
num_test_overetch = 11; % Number of elements for Overetch range in test 
num_test_offset = 11; % Number of elements for Offset range in test
num_test_thickness = 11; % Number of elements for Thickness range in test

% Range of Parameters [um]
min_overetch = 0.1;
max_overetch = 0.5;
min_offset = -0.5;
max_offset = 0.5;
min_thickness = 29.0;
max_thickenss = 31.0;

% Name of Ouput Files
training_output_file = "training.mat";
testing_output_file = "testing.mat";

% Choose if generate Training Data or Testing Data
Do_testing = false;
Do_training = ~Do_testing;

save_file = "testing_lhs.mat";

%% Training / Testing Dataset Parameters Generation

if Do_training
    % Latin Hypercube generation of Device Parameters
    samples = latinHypercubeSampling(3, [min_overetch,max_overetch; ... % Overetch Range
                                    min_offset, max_offset; ... % Offset Range
                                    min_thickness, max_thickenss], ... % Thickness Range
                                    num_training_samples);
    overetch_values = samples(:,1);
    offset_values =samples(:,2);
    thickness_values = samples(:,3);
    save_file = training_output_file;

elseif Do_testing
    % Linearly spaced generation of Device Parameters
    overetch_values = linspace(min_overetch, max_overetch, num_test_overetch);
    offset_values = linspace(min_offset, max_offset, num_test_offset);
    thickness_values = linspace(min_thickness, max_thickenss, num_test_thickness);
    samples = combvec(overetch_values, offset_values, thickness_values)';
    save_file = testing_output_file;

end

%% Parameter to Set if simulation needs to be restarted

if initialSimulation > 1
    load("samples.mat")
    load(save_file)
    num_samples = length(samples);
    dC_dV = S_dataset.dC_dV;
    dC_1g = S_dataset.dC_1g;
else 
    % Cell array to store CapacityTable objects for each simulation
    save("samples.mat","samples")
    num_samples = length(samples);
    totalSimulation = num_samples;
    C_dataset = cell(num_samples,1);
    dC_dV = zeros(totalSimulation,1);
    dC_1g = zeros(totalSimulation,1);
end

% Generate the capacity values for each parameter combination
totalSimulations = num_samples;
h = cov.memsplus.Simulation('Accelerometer.3dsch');

% %% Rows to store
% 
% overetch_values = input_file.OE_um_(1:totalSimulation);
% offset_values = input_file.x0_un_(1:totalSimulation);
% thickness_values = input_file.t_um_(1:totalSimulation);
% dC_dV_ref = input_file.dCdV_pF_(1:totalSimulation);
% dC_1g_ref = input_file.S_DC_1g_pF_(1:totalSimulation);
% 
% if initialSimulation == 1
%     dC_dV = zeros(totalSimulation,1);
%     dC_dV_anal = zeros(totalSimulation,1);
%     dC_dV = zeros(totalSimulation,1);
%     dC_1g_anal = zeros(totalSimulation,1);
%     U_dataset = cell(totalSimulation,1);
%     C1_dataset = cell(totalSimulation,1);
%     C2_dataset = cell(totalSimulation,1);
% elseif initialSimulation > 1
%     load(save_file)
%     dC_dV = output.dC_dV;
%     dC_dV_anal = output.dC_dV_anal;
%     dC_1g = output.dC_1g;
%     dC_1g_anal = output.dC_1g_anal;
% end
% 
% global capacity
% capacity = @(x, offset, thickness, overetch) ...
%     10 * 8.854e-12 * thickness * 1e-6 * (101 - 2 * overetch) * ...
%     (1 ./ (1.2 + 2 * overetch - offset - x) - 1 ./ (1.2 + 2 * overetch + offset + x));
%%
for currentSimulation=initialSimulation:totalSimulation

    % Extract parameters from the input file
    overetch = samples(currentSimulation,1);
    offset = samples(currentSimulation,2);
    thickness = samples(currentSimulation,3);

    % overetch = samples(currentSimulation,1);
    disp(strcat('Overetch: ',num2str(overetch)))
    % offset = samples(currentSimulation,2);
    disp(strcat('Offset: ',num2str(offset)))
    % thickness = samples(currentSimulation,3);
    disp(strcat('Thickness: ',num2str(thickness)))
    % save(save_file,"U_dataset","C1_dataset","C2_dataset");

    % Set Overetch
    h.Variables.Overetch = overetch;
    h.Variables.offset = offset;
    h.Variables.Thickness = thickness;
    
    % Set damping
    h.Variables.alpha = 31400;
    h.Variables.beta = 0;
    
    % Configure the alternative inputs based on the current case
    Voltage = @(t) VoltageProfile(t);
    
    % Create and compute a DC solution
    dc = h.Analyses.add('DC');
    dc.Properties.ExposedConnectorsValues.E_UP_right = 0.0;
    dc.Properties.ExposedConnectorsValues.E_DOWN_right = 0.0;
    % dc.Properties.ExposedConnectorsValues.E_UP_left = 0.0;
    % dc.Properties.ExposedConnectorsValues.E_DOWN_left = 0.0;
    dc.run();

    % Create and set up a transient analysis
    tran = dc.add('Transient');
    tran.Properties.TimeSpan.Values = 0:ts:tf;
    tran.Properties.Tolerances.RelativeTolerance = 1e-12;
    tran.Properties.ExposedConnectorsValues.E_UP_right = Voltage;
    tran.Properties.ExposedConnectorsValues.E_DOWN_right = Voltage;
    % tran.Properties.ExposedConnectorsValues.E_UP_left = 0.0;
    % tran.Properties.ExposedConnectorsValues.E_DOWN_left = 0.0;
    tran.runtimeSettings.Plotter.doPlotting = false;
    tran.run();
    
    capacity_C1 = 2*tran.Result.Outputs.Cap17.Values;
    capacity_C2 = 2*tran.Result.Outputs.Cap18.Values;
    capacity = capacity_C1 - capacity_C2;

    % Add the capacity values to the CapacityTable
    timeSteps = 0:ts:tf;
    U_table = MEMS_Table(overetch, ...
                         offset,  ...
                         thickness, ...
                         timeSteps, ....
                         capacity);

    C_dataset{currentSimulation} = U_table;

    % [Simulation Analysis Code]
    % Use functions or subroutines to avoid repeating code chunks for different DC analyses
    V_left=0.0;
    V_right=0.0;
    tax=0.0;
    [~, dC1_0V_temp] = simulation(h,V_left,V_right,tax,offset,thickness,overetch);

    V_left=0.0;
    V_right=1.8;
    tax=0.0;
    [~, dC1_dV_temp] = simulation(h,V_left,V_right,tax,offset,thickness,overetch);

    dC_dV(currentSimulation) = dC1_dV_temp - dC1_0V_temp;

    V_left=0.69;
    V_right=0.69;
    tax=0.0;
    [dC_0g_temp, ~] = simulation(h,V_left,V_right,tax,offset,thickness,overetch);

    V_left=0.69;
    V_right=0.69;
    tax=-9.81;
    [dC_1g_temp, ~] = simulation(h,V_left,V_right,tax,offset,thickness,overetch);

    dC_1g(currentSimulation) = dC_1g_temp - dC_0g_temp;

    % Save data periodically
    if mod(currentSimulation, 10) == 0

        S_dataset = table(overetch_values, offset_values, thickness_values, dC_dV, dC_1g, ...
            'VariableNames',{'overetch','offset','thickness','dC_dV','dC_1g'});
        
        % Save final data
        save(save_file,"S_dataset","C_dataset");

    end  

    % Display progress update
    fprintf('Simulation %d of %d\n', currentSimulation, totalSimulation);
    % Update loading bar
    progress = currentSimulation / totalSimulation;
    loadingBarWidth = 30;
    numCompleted = floor(loadingBarWidth * progress);
    numRemaining = loadingBarWidth - numCompleted;
    loadingBar = [repmat('#', 1, numCompleted), repmat('-', 1, numRemaining)];
    fprintf('[%s] %.1f%%\n', loadingBar, progress * 100);
    tran.delete
    dc.delete
end

%%

S_dataset = table(overetch_values, offset_values, thickness_values, dC_dV, dC_1g, ...
            'VariableNames',{'overetch','offset','thickness','dC_dV','dC_1g'});
        
% Save final data
save(save_file,"S_dataset","C_dataset");



