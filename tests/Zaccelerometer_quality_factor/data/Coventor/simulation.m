%% Create and compute a DC solution
function [dC, dC1] = simulation(h,V_left,V_right,tax,offset,thickness,overetch)

    dc = h.Analyses.add('DC');
    dc.Properties.ExposedConnectorsValues.E_UP_right = V_right;
    dc.Properties.ExposedConnectorsValues.E_DOWN_right = V_right;
    dc.Properties.ExposedConnectorsValues.E_UP_left = V_left;
    dc.Properties.ExposedConnectorsValues.E_DOWN_left = V_left;
    dc.Properties.ExposedConnectorsValues.tax = tax;
    dc.run();
    Caps = dc.Result.Outputs;
    dC = 2*(Caps.Cap17.Values-Caps.Cap18.Values)*1e12;
    displacement = dc.Result.States.ProofMass_x.Values;

    dC1 = 1e12*2*Caps.Cap17.Values;
    dc.delete
end