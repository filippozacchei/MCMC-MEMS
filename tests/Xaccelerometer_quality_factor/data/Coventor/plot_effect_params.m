% Define the parameter ranges and mean values
num_test_points = 5;
min_overetch = 0.1; max_overetch = 0.5;
min_offset = -0.4; max_offset = 0.4;
min_thickness = 29; max_thickness = 31;
min_qFactor = 0.4; max_qFactor = 0.6;

% Generate the values for each parameter
overetch_values = linspace(min_overetch, max_overetch, num_test_points);
offset_values = linspace(min_offset, max_offset, num_test_points);
thickness_values = linspace(min_thickness, max_thickness, num_test_points);
qFactor_values = linspace(min_qFactor, max_qFactor, num_test_points);

% Fixed mean values
mean_overetch = table2array(S_dataset(3,1));
mean_offset = 0.0;
mean_thickness = 30.0;
mean_qFactor = 0.5;

% Function to plot the effect of varying each parameter


% Call the plot function for each parameter
plot_effect(overetch_values, {mean_overetch, mean_offset, mean_thickness, mean_qFactor}, 'overetch', C_dataset, S_dataset);
plot_effect(offset_values, {mean_overetch, mean_offset, mean_thickness, mean_qFactor}, 'offset', C_dataset, S_dataset);
plot_effect(thickness_values, {mean_overetch, mean_offset, mean_thickness, mean_qFactor}, 'thickness', C_dataset, S_dataset);
plot_effect(qFactor_values, {mean_overetch, mean_offset, mean_thickness, mean_qFactor}, 'qFactor', C_dataset, S_dataset);


function plot_effect(param_values, fixed_values, param_name, C_dataset, S_dataset)
    figure('Position', [100, 100, 1024, 768]); % Large figure size for presentation
    hold on;
    title(['Effect of Varying ', param_name, ' on Displacement Time Series'], 'FontSize', 16);
    xlabel('Time', 'FontSize', 14);
    ylabel('Displacement (units)', 'FontSize', 14);
    
    colors = lines(length(param_values)); % Get a set of colors for the plots
    
    for i = 1:length(param_values)
        % Create a temporary copy of fixed_values to modify
        temp_values = fixed_values;
        temp_values{strcmp(param_name, S_dataset.Properties.VariableNames)} = param_values(i);
        
        % Find the index of the row in S_dataset that matches temp_values
        index = find(all(table2array(S_dataset(:,1:4)) == cell2mat(temp_values), 2));
        
        % Retrieve the corresponding time series data from C_dataset
        time_series = C_dataset{index}.Displacement;
        
        plot(time_series, 'DisplayName', [param_name ' = ' num2str(param_values(i))], ...
             'LineWidth', 2, 'Color', colors(i,:));
    end
    
    legend show;
    legend('Location', 'best', 'FontSize',18);
    grid on;
    hold off;
end