%%
clc
clear all
close all

load Capacity.mat

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
overetch_values = linspace(0.3, 0.5, 5);      % Modify overetch value as desired

% Plot the capacity variation over time for different parameter values
figure('Position', [100, 100, 800, 500], 'Renderer', 'painters'); % Adjust figure size and resolution
hold on;
colors = jet(numel(overetch_values));
time = 0:ts:tf;
index=2:201;
% Line and marker styles
lineWidth = 1.5;
markerSize = 6;

% Generate the capacity values for each parameter combination
for i = 1:10:300
    cap = Capacity_dataset{i};
    plot(time(index), 1e15*(cap.Capacity(index)-cap.Capacity(index-1)), 'LineWidth', lineWidth, 'LineStyle','-.');
end

hold off;
xlabel('Time [s]');
ylabel('Capacity Variation {\Delta}C [fF]');
title('Capacity Variation for Different Overetch and Acceleration');
% legend(cellstr(num2str(numel(overetch_values)', 'Overetch = %.2f')), 'Location', 'best');
set(gca, 'FontSize', 12); % Adjust font size of axes labels and ticks

% Customize the plot appearance for publication
box on; % Add a box around the plot
grid on; % Add grid lines
colormap(colors); % Set the colormap to match the line colors
colorbar('Ticks', 0.3:0.05:0.5, 'TickLabels', cellstr(num2str(overetch_values', '%.2f'))); % Add colorbar with overetch values
caxis([0.3, 0.5]); % Set the color axis limits
xlim([0, tf]); % Adjust x-axis limits
% ylim([min(Ca(:)), max(C(:))]); % Adjust y-axis limits
set(gca, 'TickDir', 'out'); % Set tick direction
set(gca, 'LineWidth', 1); % Adjust axis linewidth
set(gca, 'FontName', 'Arial'); % Set font name
set(gca, 'FontWeight', 'bold'); % Set font weight

%% Plot the capacity variation over time for different overetch values
figure('Position', [100, 100, 800, 500], 'Renderer', 'painters'); % Adjust figure size and resolution
hold on;
colors = hsv(numel(overetch_values));
time = 0:ts:tf;

% Prepare data for surface plot
[X, Y] = meshgrid(time, amplitudeX/9.81);

% Surface plot
for l = 1:numel(overetch_values)
    cap = [];
    for j = 1:11
        temp = Capacity_dataset{j, 1, 1, l}.Capacity(index) - Capacity_dataset{j, 1, 1, l}.Capacity(index-1);
        cap = [cap; temp'];
    end
    h = surf(X(:,index-1), Y(:,index-1), 1e15*cap, 'EdgeColor', 'none', 'FaceColor', colors(l, :), 'FaceAlpha', '1'); % Add transparency
    set(h, 'EdgeColor', 'k', 'LineStyle', ":");
end
light("Style","Infinite","Position",[-1e180 0 0])
light("Style","Infinite","Position",[1e18 0 0])
light("Style","Infinite","Position",[1 -1 0])
light("Style","Infinite","Position",[0 0 1e18])
light("Style","Infinite","Position",[0 0 -11e18])
material dull

% 
% % Highlight intersections
% for l = 1:2:numel(overetch_values)-1
%     cap1 = [];
%     cap2 = [];
%     for j = 1:11
%         temp1 = Capacity_dataset{j, 1, 1, l}.Capacity(index) - Capacity_dataset{j, 1, 1, l}.Capacity(index-1);
%         temp2 = Capacity_dataset{j, 1, 1, l+1}.Capacity(index) - Capacity_dataset{j, 1, 1, l+1}.Capacity(index-1);
%         cap1 = [cap1; temp1'];
%         cap2 = [cap2; temp2'];
%     end
%     intersection = cap1 .* cap2; % Find intersection by element-wise multiplication
%     surf(X(:,index-1), Y(:,index-1), intersection, 'EdgeColor', 'none', 'FaceColor', 'red', 'FaceAlpha', 0.8); % Highlight intersection
% end

hold off;
xlabel('Time [s]');
ylabel('Acceleration [g]');
zlabel('Capacity Variation {\Delta}C [fF]');
title('Capacity Variation for Different Overetch and Acceleration');
set(gca, 'FontSize', 12); % Adjust font size of axes labels and ticks

% Customize the plot appearance for publication
box on; % Add a box around the plot
grid on; % Add grid lines
colormap(colors); % Set the colormap to enhance visibility
colorbar('Ticks', linspace(0.3, 0.5, numel(overetch_values)), 'TickLabels', cellstr(num2str(overetch_values', '%.2f')), 'FontSize', 12); % Add colorbar with overetch values
caxis([0.3, 0.5]); % Set the color axis limits
xlim([0, 1e-3]); % Adjust x-axis limits
ylim([0, max(amplitudeX/g)]); % Adjust y-axis limits
zlim([-3e-1,3e-1]); % Adjust y-axis limits
set(gca, 'TickDir', 'out'); % Set tick direction
set(gca, 'LineWidth', 1); % Adjust axis linewidth
set(gca, 'FontName', 'Arial'); % Set font name
set(gca, 'FontWeight', 'bold'); % Set font weight
view(-150,20); % Set 3D view





%%
cap = CapacityTable(numel(timeSteps), amplitudeX(i), amplitudeY(j), amplitudeZ(k), overetch_values(l));
cap.addData(timeSteps, capacity);
Capacity_dataset{i,j,k,l} = cap;  % Update the corresponding element in Capacity_dataset

% Print the Capacity attribute within the loop
disp(Capacity_dataset{i,j,k,l}.Capacity);
plot(time, Capacity_dataset{i,j,k,l}.Capacity);

