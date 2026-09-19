%% ===============================================================
%  FAST PARALLEL DATASET GENERATION
%  - Uses 'parsim' to run on all CPU cores
%  - Eliminates re-compilation time
%  - Fixes 'distcomp' deprecation warning
% ===============================================================
clear; clc;

% Check for Parallel Computing Toolbox (Modern Check)
if isempty(ver('parallel'))
    warning('Parallel Computing Toolbox not found. Simulations will run sequentially.');
else
    % Optional: Start pool now to avoid delay in the loop
    if isempty(gcp('nocreate'))
        parpool; 
    end
end

%% 1. USER SETTINGS
models = {
    "motor_healthy", ...
    "motor_brb", ...
    "motor_bearing_fault", ...
    "motor_sws", ...
    "motor_eccentricity_fault", ...
    "motor_VI"
};

% Matches your variable naming convention
base_names = {
    "Ia", "Ib", "Ic";
    "Ia1", "Ib1", "Ic1";
    "Ia2", "Ib2", "Ic2";
    "Ia3", "Ib3", "Ic3";
    "Ia4", "Ib4", "Ic4";
    "Ia5", "Ib5", "Ic5"
};

num_samples = 4;           % Samples per class
V_min = 350; V_max = 450;
T_min = 10;   T_max = 60;
f_min = 49.5; f_max = 50.5;
Fs = 10000;
trim_samples = 10000;       % Remove first 1 sec

disp("Starting FAST dataset generation...");

%% 2. MODEL LOOP (Sequential Setup, Parallel Execution)
for m = 1:length(models)
    mdl = models{m};
    Ia_base = base_names{m,1};
    Ib_base = base_names{m,2};
    Ic_base = base_names{m,3};
    
    fprintf('\nPreparing Model: %s ...\n', mdl);
    load_system(mdl); % Load once into memory
    
    % --- PREPARE PARALLEL INPUTS ---
    % Instead of running immediately, we build a "Job List"
    in = Simulink.SimulationInput.empty(0, num_samples);
    
    for i = 1:num_samples
        in(i) = Simulink.SimulationInput(mdl);
        
        % Randomize Parameters
        V_in = V_min + rand() * (V_max - V_min);
        Torque_in = T_min + rand() * (T_max - T_min);
        f_supply = f_min + rand() * (f_max - f_min);
        
        % Calculate Severity based on model index
        if m == 6 % VI
            fault_severity = 0.05 + rand() * (0.15 - 0.05);
        elseif m == 4 % SWS
            fault_severity = 0.05 + rand() * (0.45 - 0.05);
        elseif m == 5 % Eccentricity
            fault_severity = 0.15 + rand() * (0.70 - 0.15);
        elseif m == 3 % Bearing
            fault_severity = 0.20 + rand() * (0.80 - 0.20);
        elseif m == 2 % BRB
            fault_severity = 0.20 + rand() * (0.80 - 0.20);
        else % Healthy
            fault_severity = 0;
        end
        
        % Set Variables directly on the input object
        in(i) = in(i).setVariable('V_in', V_in);
        in(i) = in(i).setVariable('Torque_in', Torque_in);
        in(i) = in(i).setVariable('fault_severity', fault_severity);
        in(i) = in(i).setVariable('f_supply', f_supply);

        
        in(i) = in(i).setModelParameter('FastRestart', 'off');
    end
    
    % --- RUN PARALLEL SIMULATIONS ---
    fprintf('  Running %d simulations in parallel...\n', num_samples);
    
    % 'parsim' manages the workers automatically.
    % 'TransferBaseWorkspaceVariables' = off speeds it up significantly
    simOut = parsim(in, 'ShowProgress', 'on', ...
        'TransferBaseWorkspaceVariables', 'off');
    
    % --- PROCESS & SAVE RESULTS ---
    fprintf('  Processing and Saving data...\n');
    
    for i = 1:num_samples
        if isempty(simOut(i).ErrorMessage)
            % Extract Data
            Ia_raw = simOut(i).get('Ia');
            Ib_raw = simOut(i).get('Ib');
            Ic_raw = simOut(i).get('Ic');
            
            % Trim transient (Safe check)
            if length(Ia_raw) > trim_samples
                Ia_cut = Ia_raw(trim_samples+1:end);
                Ib_cut = Ib_raw(trim_samples+1:end);
                Ic_cut = Ic_raw(trim_samples+1:end);
            else
                Ia_cut = Ia_raw; Ib_cut = Ib_raw; Ic_cut = Ic_raw;
            end
            
            % Generate Names
            Ia_name = sprintf('%s_%d', Ia_base, i);
            Ib_name = sprintf('%s_%d', Ib_base, i);
            Ic_name = sprintf('%s_%d', Ic_base, i);
            
            % Save to Base Workspace
            assignin('base', Ia_name, Ia_cut);
            assignin('base', Ib_name, Ib_cut);
            assignin('base', Ic_name, Ic_cut);
        else
            fprintf('  Failed simulation %d: %s\n', i, simOut(i).ErrorMessage);
        end
    end
    
    % Cleanup
    close_system(mdl, 0); 
end

disp("====================================================");
disp(" Fast Dataset generation complete!");
disp("====================================================");