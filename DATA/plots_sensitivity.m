load('S_training_dataset.mat');

% Scatter plot for overetch vs. dC_1g
figure;
scatter(S_training_dataset.overetch, 1e3*S_training_dataset.dC_1g, 'filled');
xlabel('Overetch (\mu m)', 'FontSize', 14);
ylabel('Sensitivity [fF/N]', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

% Save the figure
saveas(gcf, 'overetch_vs_dC_1g.png'); % Save as a PNG file

% Scatter plot for offset vs. dC_1g
figure;
scatter(S_training_dataset.offset, 1e3*S_training_dataset.dC_1g, 'filled');
xlabel('Offset (\mu m)', 'FontSize', 14);
ylabel('Sensitivity [fF/N]', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

% Save the figure
saveas(gcf, 'offset_vs_dC_1g.png'); % Save as a PNG file

% Scatter plot for thickness vs. dC_1g
figure;
scatter(S_training_dataset.thickness, 1e3*S_training_dataset.dC_1g, 'filled');
xlabel('Thickness (\mu m)', 'FontSize', 14);
ylabel('Sensitivity [fF/N]', 'FontSize', 14);
grid on;
set(gca, 'FontSize', 12);

% Save the figure
saveas(gcf, 'thickness_vs_dC_1g.png'); % Save as a PNG file
