function a = alpha_qualityFactor(w, h, E, l1, l2, o, mass, Qfactor)
    % This function evaluates alpha based on the provided parameters
    % and a target fundamental frequency f0_target.

    % Constants
    k = stiffness(w,h,E,l1,l2,o);

    % Angular frequency
    omega = sqrt(k / mass); 

    % Damping ratio, delta, and Q-factor as anonymous functions
    xi = @(alpha) alpha / (2 * omega);
    delta = @(alpha) 2 * pi * xi(alpha) / sqrt(1 - (xi(alpha)^2));
    Q = @(alpha) 1 / delta(alpha);
    % Alpha evaluation using fsolve
    a = fzero(@(alpha) sqrt(1 - (xi(alpha)^2)) - (Qfactor * 2 * pi * xi(alpha)), 20000);
end
