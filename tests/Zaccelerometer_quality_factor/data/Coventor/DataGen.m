%In order to avoid possible errors due to changes in the state of the MATLAB 
%workspace, it is recommended to clean the workspace in the beginning of each
%script using clearvars command.
%clearvars

clear all
clc %clears the MATLAB command window and closes all open figures 

warning('off','all')

%% Parameters
g  = 9.81;
ts = 1e-5; % Acquisition Time Step 
tf = 1.5e-3; % Final Time
    
% Load data and set initial parameters
initialSimulation = 900;
totalSimulation = 2500;

% Function to generate Input Voltage Profile 
amplitude_val = 1.8;
Tx_val = 0.4e-3;
VoltageProfile = @(t) 0.5*amplitude_val*(1+(sin((t/Tx_val-1/4)*2*pi))).*(t<2*Tx_val);
 
% Data Generation
num_training_samples = 2500; 
num_test_overetch    = 5;  
num_test_offset      = 5; 
num_test_thickness   = 5; 
num_test_qFactor     = 5;

% Range of Parameters [um]
min_overetch  = 0.1;
max_overetch  = 0.5;
min_offset    = -0.4;
max_offset    = 0.4;
min_thickness = 29.0;
max_thickenss = 31.0;
min_qFactor   = 0.4;
max_qFactor   = 0.6;

% Name of Ouput Files
training_output_file = "training_qFactor_all.mat";
testing_output_file = "testing_qFactor.mat";

% Choose if generate Training Data or Testing Data
Do_testing = false;
Do_training = ~Do_testing;

%% Training / Testing Dataset Parameters Generation

if Do_training
    % Latin Hypercube generation of Device Parameters
    samples = latinHypercubeSampling(4, [min_overetch,max_overetch; ... % Overetch Range
                                    min_offset, max_offset; ... % Offset Range
                                    min_thickness, max_thickenss; ... % Thickness Range
                                    min_qFactor, max_qFactor], ... % Q factor Range
                                    num_training_samples);
    overetch_values  = samples(:,1);
    offset_values    = samples(:,2);
    thickness_values = samples(:,3);
    qFactor_values   = samples(:,4);
    save_file = training_output_file;

elseif Do_testing
    % Linearly spaced generation of Device Parameters
    overetch_values  = linspace(min_overetch, max_overetch, num_test_overetch);
    offset_values    = linspace(min_offset, max_offset, num_test_offset);
    thickness_values = linspace(min_thickness, max_thickenss, num_test_thickness);
    qFactor_values   = linspace(min_qFactor, max_qFactor, num_test_qFactor);
    samples = combvec(overetch_values, offset_values, thickness_values, qFactor_values)';
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

%%
overetch_vec  = samples(:,1);
offset_vec    = samples(:,2);
thickness_vec = samples(:,3);
qFactor_vec   = samples(:,4);

for currentSimulation=initialSimulation:totalSimulation

    % Extract parameters from the input file
    overetch  = samples(currentSimulation,1);
    offset    = samples(currentSimulation,2);
    thickness = samples(currentSimulation,3);
    qFactor   = samples(currentSimulation,4);

    disp(strcat('Overetch: ',num2str(overetch)))
    disp(strcat('Offset: ',num2str(offset)))
    disp(strcat('Thickness: ',num2str(thickness)))
    disp(strcat('Q Factor: ',num2str(qFactor)))

    % Set Overetch
    h.Variables.Overetch = overetch;
    h.Variables.Offset = offset;
    h.Variables.Thickness = thickness;

    mass = thickness*84000*2320*1e-18;
    width = 2.8e-6;
    Young = 160e9;
    length1 = 211.4e-6;
    length2 = 100.4e-6;

    alpha = alpha_qualityFactor(width,thickness*1e-6,Young,length1, length2, overetch*1e-6, mass, qFactor);

    % Set damping
    h.Variables.alpha = alpha;
    h.Variables.beta = 0;
    
    % Configure the alternative inputs based on the current case
    Voltage = @(t) VoltageProfile(t);
    
    % Create and compute a DC solution
    dc = h.Analyses.add('DC');
    dc.Properties.ExposedConnectorsValues.E_UP_right = 0.0;
    dc.Properties.ExposedConnectorsValues.E_DOWN_right = 0.0;
    dc.run();

    % Create and set up a transient analysis
    tran = dc.add('Transient');
    tran.Properties.TimeSpan.Values = 0:ts:tf;
    tran.Properties.Tolerances.RelativeTolerance = 1e-12;
    tran.Properties.ExposedConnectorsValues.E_UP_right = Voltage;
    tran.Properties.ExposedConnectorsValues.E_DOWN_right = Voltage;
    tran.runtimeSettings.Plotter.doPlotting = true;
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
    [~, dC1_0V_temp] = simulation(h,0.0,0.0,0.0,offset,thickness,overetch);
    [~, dC1_dV_temp] = simulation(h,0.0,1.8,0.0,offset,thickness,overetch);
    dC_dV(currentSimulation) = dC1_dV_temp - dC1_0V_temp;

    [dC_0g_temp, ~] = simulation(h,0.69,0.69,0.0,offset,thickness,overetch);
    [dC_1g_temp, ~] = simulation(h,0.69,0.69,-9.81,offset,thickness,overetch);
    dC_1g(currentSimulation) = dC_1g_temp - dC_0g_temp;

    % Save data periodically
    if mod(currentSimulation, 2) == 0

        S_dataset = table(overetch_vec, offset_vec, thickness_vec, qFactor_vec, dC_dV, dC_1g, ...
            'VariableNames',{'overetch','offset','thickness','qFactor','dC_dV','dC_1g'});
        
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

S_dataset = table(overetch_vec, offset_vec, thickness_vec, qFactor_vec, dC_dV, dC_1g, ...
    'VariableNames',{'overetch','offset','thickness','qFactor','dC_dV','dC_1g'});
        
% Save final data
save(save_file,"S_dataset","C_dataset");



