% ============================================================
% Visualize all intermediate fields of gen_structure_field,
% gen_permeability_field and gen_porosity_field using hooks
%
% Author: Rino M. Albertin
% Date:   2026-01-04
% ============================================================

function test_gen_field()
clear; clc; close all;

%% --- Path setup ---------------------------------------------
this_file = mfilename('fullpath');
this_dir  = fileparts(this_file);

functions_dir = fileparts(this_dir);
core_dir      = fullfile(functions_dir,'core');

addpath(genpath(core_dir));

%% === DOMAIN & GLOBAL PARAMETERS =============================
Lx  = 1.2;
Ly  = 0.75;
res = 0.003;
seed    = 1;

%% === OPTIONS ===============================================
opts = struct();

% --- Background heterogeneity -------------------------------
opts.base_len_rel   = 0.10;
opts.smooth_len_rel = 0.05;
opts.ms_weight      = [0.3, 0.7];
opts.anisotropy     = [3.0, 1.0];
opts.coupling       = 0.5;

% --- Localized noises (structure-space) ---------------------
opts.noise_level       = 0.2;
opts.noise_granularity = 0.5;
opts.noise_bias        = 0.5;

% --- Global permeability statistics -------------------------
opts.k_mean  = 5e-9;
opts.var_rel = 0.5;

% --- Tensor construction ------------------------------------
opts.a_max            = 2.0;
opts.a_gamma          = 2.0;
opts.tensor_strength  = 1.0;
opts.theta_jitter     = 0.05;
opts.theta_smooth_rel = 0.05;

% --- Porosity parameters ------------------------------------
opts.A_rel = 2.0;
opts.phi_smooth_rel = 0.05;
opts.texture_amp = 0.005;

% --- Pressure BC parameters ---------------------------------
opts.p_inlet_mean = 350;      % [Pa]   | var=0.8: [196, 623] | var=1.6: [133, 920]
opts.a_sin        = 0.06;     % [-]    | var=0.8: [-0.11, +0.11] | var=1.6: [-0.16, +0.16]
opts.f_sin        = 0.75;     % [-]    | var=0.8: [-1.33, +1.33] | var=1.6: [-1.97, +1.97]
opts.phi_sin      = pi;       % [rad]  | var=0.8: [1.76, 5.59] | var=1.6: [1.19, 8.26]
opts.k_gauss      = 2;        % {1-5}
opts.a_gauss      = 0.10;     % [-]    | var=0.8: [-0.18, +0.18] | var=1.6: [-0.26, +0.26]
opts.sigma_gauss  = 0.05;     % [-]    | var=0.8: [0.028, 0.089] | var=1.6: [0.019, 0.132]
opts.gauss_jitter = 0.25;     % [-]    | var=0.8: [0.14, 0.45] | var=1.6: [0.095, 0.66]
opts.a_lin        = 0.05;     % [-]    | var=0.8: [-0.09, +0.09] | var=1.6: [-0.13, +0.13]

%% === HOOK STORAGE ==========================================
H = struct();
opts.enable_hooks = true;

% --- Structure ----------------------------------------------
opts.hooks.filtered_fields     = @(d) store_hook('filtered_fields', d);
opts.hooks.structure_field_bg  = @(d) store_hook('structure_field_bg', d);
opts.hooks.structure_field     = @(d) store_hook('structure_field', d);
opts.hooks.noises_scaled       = @(d) store_hook('noises_scaled', d);

% --- Permeability -------------------------------------------
opts.hooks.kappa_final         = @(d) store_hook('kappa_final', d);

% --- Tensor -------------------------------------------------
opts.hooks.anisotropy_ratio    = @(d) store_hook('anisotropy_ratio', d);
opts.hooks.principal_k         = @(d) store_hook('principal_k', d);
opts.hooks.tensor              = @(d) store_hook('tensor', d);

% --- Porosity -----------------------------------------------
opts.hooks.phi_input    = @(d) store_hook('phi_input', d);
opts.hooks.phi_smoothed = @(d) store_hook('phi_smoothed', d);
opts.hooks.phi_level    = @(d) store_hook('phi_level', d);
opts.hooks.phi_final    = @(d) store_hook('phi_final', d);

% --- Pressure BC --------------------------------------------
opts.hooks.p_inlet      = @(d) store_hook('p_inlet', d);

%% === RUN PIPELINE ==========================================
fields = struct();
info   = struct();

% --- 1) Structure field -------------------------------------
[fields, info.structure] = gen_structure_field( ...
    Lx, Ly, res, seed, opts);

% --- 2) Permeability + tensor -------------------------------
[fields, info.permeability] = gen_permeability_field(fields, opts);

% --- 3) Porosity --------------------------------------------
[fields, info.porosity] = gen_porosity_field(fields, opts);

% --- 4) Pressure BC -----------------------------------------
[fields, info.bc] = gen_pressure_bc(fields, opts);

%% === PIPELINE VISUALIZATION ================================

% === STAGE 1: STRUCTURE =====================================
figure('Name','Stage 1: Multiscale structure');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile; plot_field(H.filtered_fields.base_field,   'Base field');
nexttile; plot_field(H.filtered_fields.smooth_field, 'Smooth field');
nexttile; plot_field(H.structure_field_bg.z_bg,      'z_{bg}');

sgtitle('Stage 1: Multiscale Gaussian structure');

% === STAGE 2: STRUCTURE + NOISES =============================
if opts.noise_level > 0
    figure('Name','Stage 2: Structure noises');
    tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

    nexttile; plot_field(H.noises_scaled.field, 'z_{noises}');
    nexttile; plot_field(H.structure_field_bg.z_bg, 'z_{bg}');
    nexttile; plot_field(H.structure_field.z, 'z');

    sgtitle('Stage 2: Final structure field');
end

% === STAGE 3: PERMEABILITY ==================================
figure('Name','Stage 3: Permeability');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
plot_field(log10(H.kappa_final.kappa), 'log10(kappa)');

nexttile;
histogram(log10(H.kappa_final.kappa(:)),80);
grid on;
xlabel('log10(kappa)');
ylabel('count');

sgtitle('Stage 3: Permeability field');

% === STAGE 4: TENSOR ========================================
figure('Name','Stage 4: Tensor components');
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

nexttile; plot_field(log10(H.principal_k.k1), 'log10(k1)');
nexttile; plot_field(log10(H.principal_k.k2), 'log10(k2)');
nexttile; plot_field(log10(H.anisotropy_ratio.a), 'log10(a)');

nexttile; plot_field(log10(abs(H.tensor.K.Kxx)), 'log10(|Kxx|)');
nexttile; plot_field(log10(abs(H.tensor.K.Kyy)), 'log10(|Kyy|)');
nexttile; plot_field(log10(abs(H.tensor.K.Kxy)), 'log10(|Kxy|)');

sgtitle('Stage 4: Tensor components');

% === STAGE 5: POROSITY + PERMEABILITY =======================
figure('Name','Stage 5: Porosity and permeability');
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% (1) Input backbone
nexttile;
plot_field(H.phi_input.z_bg, 'Input backbone z_{bg}');
% (2) Final permeability (reference)
nexttile;
plot_field(log10(H.kappa_final.kappa), 'log_{10}(\kappa)');
% (3) Smoothed porosity backbone
nexttile;
plot_field(H.phi_smoothed.z_phi, 'Smoothed backbone z_{\phi}');
% (4) Final porosity field (large)
nexttile;
plot_field(H.phi_final.phi, '\phi(x,y) final');

% --- KC reference annotation (global, not a tile) ------------
annotation('textbox', [0.30 0.01 0.40 0.07], ...
    'String', sprintf('KC reference:  \\phi_{ref} = %.4f   |   k_{ref} = %.2e m^2', ...
        H.phi_level.phi_ref, H.phi_level.k_ref), ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'Interpreter','tex', ...
    'FontSize',11);

sgtitle('Stage 5: Porosity from structure, KC-anchored to permeability');

% === STAGE 6: PRESSURE BOUNDARY =============================
figure('Name','Stage 6: Pressure boundary condition');
p = H.p_inlet.p_inlet;

plot(p,'LineWidth',2);
grid on;
xlabel('x-index');
ylabel('p_{inlet} [Pa]');
title('Inlet pressure at y = 0');


disp('✔ Structure → permeability → porosity → pressure BC pipeline OK.');

%% === LOCAL FUNCTIONS =======================================
    function ok = store_hook(name, data)
        H.(name) = data;
        ok = true;
    end

    function plot_field(A, title_str)
        imagesc(A);
        axis equal tight;
        colorbar;
        title(title_str,'Interpreter','none');
    end
end
