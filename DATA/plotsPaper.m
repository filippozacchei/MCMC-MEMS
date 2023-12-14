index = 1:151;
figure;
plot(dx25(index,2), dx25(index,4), '-ok', 'LineWidth', 1.5, 'MarkerFaceColor', 'r'); % Red circles
hold on;
plot(dx10(index,2), dx10(index,4), ':*b', 'LineWidth', 1.5, 'MarkerFaceColor', 'g'); % Blue stars
plot(dx5(index,2), dx5(index,4), '--xk', 'LineWidth', 1.5, 'MarkerFaceColor', 'm'); % Cyan crosses
legend('Mesh I', 'Mesh II', 'Mesh III');
set(gca, 'FontSize', 14); % Increase font size
xlabel('t [s]', 'FontSize', 14); % Replace with your label
ylabel('X-displacement [{\mu}m]', 'FontSize', 14); % Replace with your label
title('COMSOL, mesh convergence, Configuration I', 'FontSize', 16)
grid on;
set(gca, 'FontSize', 12); % Adjust font size as needed

%%
index = 1:151;

% Create a figure
figure;
hold on;

% First dataset plot
plot(simC(index, 1), dx5(index, 4), '-ok', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');

% Second dataset plot
plot(U_dataset{1}.Time, 1e6 * (X_dataset{676}.Displacement), '--', 'LineWidth', 2.5, 'Color', 'b');

% Improve the appearance
set(gca, 'FontSize', 14); % Increase font size
xlabel('t [s]', 'FontSize', 14); % Replace with your label
ylabel('X-displacement [{\mu}m]', 'FontSize', 14); % Replace with your label
title('Displacement Numerical Comparison, Configuration I', 'FontSize', 16); % Replace with your title
legend('Comsol', 'Coventor', 'Location', 'best'); % Update legend labels as needed
grid on; % Add grid
box on; % Add box around the plot

hold off;

%% Sensitiivity Check
ConfigurationA = -2.672523572947665E-16	+4.5400254029049525E-15;
ConfigurationB =  1.0611195398533726E-14-6.264622552052287E-15
ConfigurationC = -6.186726583716623E-16+4.830293551725285E-15
ConfigurationD = 1.1700186329390646E-14-7.409207321344535E-15
ConfigurationE = 3.5697558375954156E-14-3.107481452629578E-14
ConfigurationF = -1.3048479669875268E-14+1.7364324580214383E-14;
ConfigurationG = 1.099652342536774E-14-6.548367569683695E-15
ConfigurationH = 1.0915178038385785E-14-6.5642674331050284E-15;
ConfigurationI = 2.296846542068043E-14-1.8585254582860554E-14;
ConfigurationL = 8.616158336974257E-15-4.338660060278766E-15;

comsol_data = 1e15*[ConfigurationA, ConfigurationB, ConfigurationC, ConfigurationD, ConfigurationE, ...
                    ConfigurationF, ConfigurationG, ConfigurationH, ConfigurationI, ConfigurationL];
coventor_data = 1e3*output.dC_1g([1, 5, 50, 100, 150, 200, 250, 300, 350, 400]);

% Create a figure
figure;
hold on;

% Plot data
plot(coventor_data, comsol_data, 'o', 'MarkerSize', 8, 'LineWidth', 1.5); % Change marker style and size
plot(coventor_data, coventor_data, '-', 'LineWidth', 1.5); % Line plot for reference

% Annotations with adjusted positions
labels = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L'};
label_offsets = [-5e-3, 0; ... % Offset for each label, adjust as needed
                 -5e-3, -5e-3; ...
                 0, 0; 0, 0; 0, 0; 0, 0; 0, 0; 0, 0; 0, 0;
                 5e-3, 5e-3 % Add offsets for each label here
                 ];

for i = 1:length(comsol_data)
    % Adjust text position by adding a small offset
    text_x = coventor_data(i) + label_offsets(i, 1);
    text_y = comsol_data(i) + label_offsets(i, 2);
    text(text_x, text_y, labels{i}, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right','FontSize',14);
end

% Set axis limits
ylim([4.1, 4.75]);

% Improve the appearance
set(gca, 'FontSize', 14); % Increase font size
xlabel('Sensitivity, Coventor-data, [fF/N]', 'FontSize', 14);
ylabel('Sensitivity [fF/N]', 'FontSize', 14);
title('Sensitivity Numerical Comparison, Configuration II', 'FontSize', 16);
legend('COMSOL','Coventor')
grid on; % Add grid
box on; % Add box around the plot
