%% batch_run.m
% Start COMSOL server manually via:
%   "C:\Program Files\COMSOL63\bin\win64\comsolmphserver.exe"
%
% ============================================================
% Full batch pipeline: parameter sampling → κ(x,y) generation → COMSOL simulation
% Author: Rino M. Albertin
% Date: 2025-10-16
%
% DESCRIPTION
%   Executes a complete batch of synthetic permeability-field simulations:
%     1. Generate parameter samples (sample_parameters.m)
%     2. Generate κ(x,y) fields (V1_gen_permeability.m)
%     3. Run Darcy–Brinkman COMSOL simulations (run_comsol_case.m)
%
%   Output structure:
%       data/raw/<batch_name>/       → generated κ(x,y) fields
%       data/processed/<batch_name>/ → exported COMSOL results
%
% REQUIREMENTS
%   • COMSOL Multiphysics with LiveLink for MATLAB
%   • Functions:
%       - sample_parameters.m
%       - gen_material_fields.m
%       - run_comsol_case.m
% ============================================================

clear; clc;

%% --- Settings ----------------------------------------------------------
debug = true;              % enable/disable debug output
dbstop if error            % stop on errors for debugging

%% --- Configuration -----------------------------------------------------
this_file  = mfilename('fullpath');
script_dir = fileparts(this_file);

project_root = fullfile(script_dir, '..');
project_root = char(java.io.File(project_root).getCanonicalPath());
addpath(genpath(fullfile(project_root, 'matlab', 'functions')));

% === SAVE COMSOL WITH SOLUTION ===
save_model = false;

% === SAMPLING PARAMETERS ===
method     = 'lhs';       % 'uniform', 'lhs', or 'sobol'
variation  = 0.8;         % relative parameter variation
N          = 1000;        % number of samples
seed       = 3001;        % reproducibility seed

batch_name = sprintf('%s_var%.0f_seed%.0f', ...
    method, variation*100, seed);

% === PATHS ===
meta_dir      = fullfile(project_root, 'data', 'meta');
raw_dir       = fullfile(project_root, 'data', 'raw', batch_name);
processed_dir = fullfile(project_root, 'data', 'processed', batch_name);
template_path = fullfile(project_root, 'comsol', 'template_brinkman.mph');

if ~isfolder(meta_dir), mkdir(meta_dir); end
if ~isfolder(raw_dir), mkdir(raw_dir); end
if ~isfolder(processed_dir), mkdir(processed_dir); end

%% --- Generate or load parameter samples -------------------------------
sample_csv = fullfile(meta_dir, [batch_name '.csv']);

if ~isfile(sample_csv)
    disp("🧩 No existing sample found — generating new parameter set...");
    sample_parameters(method, variation, N, seed, meta_dir);
else
    disp("📂 Using existing parameter sample.");
end

%% --- COMSOL Connection -------------------------------------------------
addpath('C:\Program Files\COMSOL63\mli');
try
    v = mphversion;
    disp("✅ Connected to COMSOL Server: " + v);
catch
    disp("🔄 Connecting to COMSOL Server on port 2036...");
    mphstart(2036);
    pause(2);
    v = mphversion;
    disp("✅ Connected to COMSOL Server: " + v);
end

%% --- Load sample parameters -------------------------------------------
T = readtable(sample_csv, 'Delimiter', ';');
n_cases = height(T);

disp("------------------------------------------------------------");
disp("🚀 Starting batch with " + n_cases + " cases (" + batch_name + ")");
disp("------------------------------------------------------------");

%% --- Fixed geometry parameters ----------------------------------------
Lx = 1.2;
Ly = 0.75;
res = 0.003;

%% --- Start total timer -------------------------------------------------
t_batch_start = tic;

%% --- Main batch loop ---------------------------------------------------
for i = 1:n_cases
    case_id  = T.case_id(i);
    case_tag = sprintf('case_%04d', case_id);

    % ============================================================
    % RESUME LOGIC
    % Skip if sol.csv exists AND no .mph exists (completed case)
    % Recompute if sol.csv + .mph both exist (inconsistent crash state)
    % ============================================================
    sol_file = fullfile(processed_dir, sprintf('%s_sol.csv', case_tag));
    mph_file = fullfile(processed_dir, sprintf('%s.mph', case_tag));

    if isfile(sol_file)
        if isfile(mph_file)
            fprintf('[%4d/%4d] ⚠️ Inconsistent state: sol.csv + mph present → recomputing (%s)\n', ...
                i, n_cases, case_tag);
            delete(mph_file);
        else
            fprintf('[%4d/%4d] ⏩ Skip: solution already exists (%s)\n', ...
                i, n_cases, case_tag);
            continue;
        end
    end

    % ============================================================
    % SOBOL HYBRID LOGIC: skip non-simulated cases
    % ============================================================
    if ismember('simulate', T.Properties.VariableNames)
        if ~logical(T.simulate(i))
            fprintf('[%4d/%4d] ⏭️  Meta-only Sobol case (no COMSOL): %s\n', ...
                i, n_cases, case_tag);
            continue;
        end
    end

    % --- Build option struct -------------------------------------------
    opts = struct( ...
        'k_mean',            T.k_mean(i), ...
        'var_rel',           T.var_rel(i), ...
        'base_len_rel',      T.base_len_rel(i), ...
        'smooth_len_rel',    T.smooth_len_rel(i), ...
        'ms_weight',         [T.msW_c(i), T.msW_f(i)], ...
        'anisotropy',        [T.ani_x(i), T.ani_y(i)], ...
        'coupling',          T.coupling(i), ...
        'noise_level',       T.noise_level(i), ...
        'noise_granularity', T.noise_granularity(i), ...
        'noise_bias',        T.noise_bias(i), ...
        'a_max',             T.a_max(i), ...
        'a_gamma',           T.a_gamma(i), ...
        'tensor_strength',   T.tensor_strength(i), ...
        'theta_jitter',      T.theta_jitter(i), ...
        'theta_smooth_rel',  T.theta_smooth_rel(i), ...
        ...
        'A_rel',             T.A_rel(i), ...
        'phi_smooth_rel',    T.phi_smooth_rel(i), ...
        'texture_amp',       T.texture_amp(i), ...
        ...
        'p_inlet_mean',      T.p_inlet_mean(i), ...
        'a_sin',             T.a_sin(i), ...
        'f_sin',             T.f_sin(i), ...
        'phi_sin',           T.phi_sin(i), ...
        'k_gauss',           T.k_gauss(i), ...
        'a_gauss',           T.a_gauss(i), ...
        'sigma_gauss',       T.sigma_gauss(i), ...
        'gauss_jitter',      T.gauss_jitter(i), ...
        'a_lin',             T.a_lin(i), ...
        ...
        'save',              true, ...
        'save_dir',          raw_dir, ...
        'file_tag',          case_tag ...
    );

    seed_case    = seed + case_id;

    %% --- Debug info ----------------------------------------------------
    if debug
        fprintf('\n[DEBUG] Case %d/%d (%s)\n', i, n_cases, case_tag);
    
        % --- global -----------------------------------------------------
        fprintf('  global: k_mean=%.2e | var_rel=%.2f | seed=%d\n', ...
            opts.k_mean, opts.var_rel, seed_case);
    
        % --- background -------------------------------------------------
        fprintf('  background: base_len=%.3f | smooth_len=%.3f | ms_weight=[%.2f %.2f] | anisotropy=[%.2f %.2f] | coupling=%.2f\n', ...
            opts.base_len_rel, opts.smooth_len_rel, ...
            opts.ms_weight(1), opts.ms_weight(2), ...
            opts.anisotropy(1), opts.anisotropy(2), ...
            opts.coupling);
    
        % --- noise ------------------------------------------------------
        fprintf('  noise: noise_level=%.2f | noise_granularity=%.2f | noise_bias=%.2f\n', ...
            opts.noise_level, opts.noise_granularity, opts.noise_bias);
    
        % --- tensor -----------------------------------------------------
        fprintf('  tensor: a_max=%.2f | a_gamma=%.2f | tensor_strength=%.2f | theta_jitter=%.3f | theta_smooth=%.3f\n', ...
            opts.a_max, opts.a_gamma, opts.tensor_strength, ...
            opts.theta_jitter, opts.theta_smooth_rel);

        % --- porosity ---------------------------------------------------
        fprintf('  porosity: A_rel=%.3f | phi_smooth_rel=%.3f | texture_amp=%.4f\n', ...
            opts.A_rel, opts.phi_smooth_rel, opts.texture_amp);

        % --- pressure BC --------------------------------------------
        fprintf('  p_bc: mean=%.1f | a_sin=%.3f | f_sin=%.2f | phi_sin=%.2f\n', ...
            opts.p_inlet_mean, opts.a_sin, opts.f_sin, opts.phi_sin);
        
        fprintf('  k_gauss=%d | a_gauss=%.3f | sigma_gauss=%.3f | jitter=%.2f | a_lin=%.3f\n', ...
            opts.k_gauss, opts.a_gauss, opts.sigma_gauss, opts.gauss_jitter, opts.a_lin);

        % --- io ---------------------------------------------------------
        fprintf('\n  io: save_dir=%s | file_tag=%s\n', ...
            opts.save_dir, opts.file_tag);
    end

    %% --- Step 1: Generate permeability field ---------------------------
    try
        [fields, info] = gen_material_fields(Lx, Ly, res, seed_case, opts);
        if debug
            fprintf('  → Fields exported: %s\n', info.export.paths.csv);
        end
    catch ME
        fprintf('[%4d/%4d] ❌ Error in gen_material_fields: %s\n', ...
            i, n_cases, ME.message);
        continue;
    end

    %% --- Step 2: Run COMSOL simulation --------------------------------
    field_path = info.export.paths.csv;;

    try
        [model, results] = run_comsol_case(field_path, template_path, processed_dir, save_model);
        fprintf('[%4d/%4d] ✅ COMSOL completed: %s (%.1f s)\n', ...
            i, n_cases, opts.file_tag, results.time_s);
    catch ME
        fprintf('[%4d/%4d] ❌ Error in COMSOL: %s\n', ...
            i, n_cases, ME.message);
        continue;
    end
end

%% --- End total timer ---------------------------------------------------
t_batch_end = toc(t_batch_start);
t_min = t_batch_end / 60;
t_hr  = t_batch_end / 3600;

disp("------------------------------------------------------------");
fprintf("🏁 Batch completed successfully.\n");
fprintf("⏱️ Total time: %.1f s (%.2f min | %.2f h)\n", ...
    t_batch_end, t_min, t_hr);
disp("------------------------------------------------------------");

if debug
    dbclear if error
end
