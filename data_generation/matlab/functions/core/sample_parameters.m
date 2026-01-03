%% sample_parameters.m
% ============================================================
% Generate sampled parameter sets for κ(x,y) field generation
% Author: Rino M. Albertin
% Date: 2025-10-16
%
% DESCRIPTION
%   Generates random parameter combinations for synthetic permeability
%   field generation using different sampling strategies:
%
%       - 'uniform' : purely random sampling
%       - 'lhs'     : Latin Hypercube Sampling (space-filling)
%       - 'sobol'   : Sobol low-discrepancy sequence
%
%   Variation is applied multiplicatively in log-space for positive parameters
%   and in logit-space for [0,1] parameters to maintain stable distributions
%   even for large variation values (e.g. 0.8 or 1.6).
%
%   SPECIAL BEHAVIOR FOR SOBOL:
%   ---------------------------
%   When method == 'sobol', an additional boolean column `simulate` is added
%   to the output table. This column flags which Sobol samples should be
%   physically simulated (e.g. with COMSOL).
%
%   The first X% of Sobol samples are marked as:
%       simulate == true
%
%   The remaining samples are marked as:
%       simulate == false
%
%   This enables hybrid workflows:
%       - expensive physics simulations on a Sobol subset
%       - cheap NN forward passes on the full Sobol design
%
%   For 'uniform' and 'lhs', no `simulate` column is added and ALL cases
%   are assumed to be simulated.
%
% ------------------------------------------------------------------------
% USAGE
%   sample_parameters(method, variation, N, seed, p_log, output_dir)
%
% INPUTS
%   method      – Sampling method: 'uniform', 'lhs', 'sobol'
%   variation   – Variation factor: max multiplicative factor = (1 + variation)
%   N           – Number of parameter sets to generate
%   seed        – Random seed for reproducibility
%   p_log       – Fraction of lognormal cases (optional)
%   output_dir  – Output directory for saving CSV/JSON
%
% OUTPUTS
%   Saves .csv and .json in output_dir
% ============================================================

function sample_parameters(method, variation, N, seed, p_log, output_dir)

%% --- Defaults ----------------------------------------------------------
if nargin < 1, method = 'lhs'; end
if nargin < 2, variation = 0.10; end
if nargin < 3, N = 200; end
if nargin < 4, seed = 42; end
if nargin < 5, p_log = 1.0; end
if nargin < 6
    this_file  = mfilename('fullpath');
    script_dir = fileparts(this_file);
    project_root = fullfile(script_dir, '..', '..');
    project_root = char(java.io.File(project_root).getCanonicalPath());
    output_dir = fullfile(project_root, 'data', 'meta');
end

if ~isfolder(output_dir), mkdir(output_dir); end
rng(seed);

valid_methods = ["uniform","lhs","sobol"];
assert(any(strcmpi(method, valid_methods)), ...
    'Invalid method. Use ''uniform'', ''lhs'', or ''sobol''.');

%% --- Base parameters ---------------------------------------------------
base = struct( ...
    'k_mean',          5e-9, ...
    'var_rel',         0.5, ...
    'corr_len_rel',    0.05, ...
    'anisotropy',      [3.0, 1.0], ...
    'volume_fraction', 0.8, ...
    'ms_weight',       [0.7, 0.3], ...
    'ms_scale',        0.1, ...
    'coupling',        0.5 ...
);

param_names = ["k_mean","var_rel","corr_len_rel","anisotropy_x","anisotropy_y", ...
               "volume_fraction","ms_weight_c","ms_weight_f", ...
               "ms_scale","coupling"];
n_params = numel(param_names);

%% --- Sampling setup ----------------------------------------------------
switch lower(method)
    case 'uniform'
        X = rand(N, n_params);
    case 'lhs'
        X = lhsdesign(N, n_params, 'Criterion','maximin','Iterations',50);
    case 'sobol'
        p = sobolset(n_params, 'Skip', 1000, 'Leap', 200);
        p = scramble(p,'MatousekAffineOwen');
        X = net(p, N);
end

%% --- Variation helpers -------------------------------------------------
span = log(1 + variation);      % multiplicative factor range in log-space
Z = 2*X - 1;                    % map to [-1, 1]

logit = @(x) log(x./(1-x));
inv_logit = @(z) 1 ./ (1 + exp(-z));

%% --- Apply variation (log-space for >0 parameters) --------------------
k_mean       = base.k_mean       .* exp(span * Z(:,1));
var_rel      = base.var_rel      .* exp(span * Z(:,2));
corr_len_rel = base.corr_len_rel .* exp(span * Z(:,3));
anisotropy_x = base.anisotropy(1).* exp(span * Z(:,4));
anisotropy_y = base.anisotropy(2).* exp(span * Z(:,5));
ms_scale     = base.ms_scale     .* exp(span * Z(:,9));

%% --- [0,1] parameters via logit-space ---------------------------------
vol_frac_logit = logit(base.volume_fraction) + span * Z(:,6);
volume_fraction = inv_logit(vol_frac_logit);

coupling_logit = logit(base.coupling) + span * Z(:,10);
coupling = inv_logit(coupling_logit);

%% --- ms-weight sampling (softmax ensures [0,1], sum=1) ----------------
w_c_base = log(base.ms_weight(1));
w_f_base = log(base.ms_weight(2));

w_c = w_c_base + span * Z(:,7);
w_f = w_f_base + span * Z(:,8);

w_exp = exp([w_c w_f]);
ms_weight_c = w_exp(:,1) ./ sum(w_exp,2);
ms_weight_f = w_exp(:,2) ./ sum(w_exp,2);

%% --- Lognormal flag sampling ------------------------------------------
lognormal = rand(N,1) < p_log;

%% --- Assemble table ----------------------------------------------------
T = table((1:N)', k_mean, var_rel, corr_len_rel, anisotropy_x, anisotropy_y, ...
          volume_fraction, ms_weight_c, ms_weight_f, ms_scale, ...
          coupling, lognormal, ...
    'VariableNames', {'case_id','k_mean','var_rel','corr_len_rel', ...
                      'ani_x','ani_y','vol_frac', ...
                      'msW_c','msW_f','ms_scale','coupling','lognormal'});

%% --- Sobol-specific simulation flag -----------------------------------
if strcmpi(method,'sobol')
    n_sim = 1000;                              % FIX number of GT/COMSOL cases

    simulate = false(N,1);
    simulate(1:n_sim) = true;                  % take first Sobol points

    T.simulate = simulate;

    % store as info only (not used for selection)
    simulate_frac = n_sim / N;
end

%% --- Metadata ----------------------------------------------------------
meta = struct();
meta.method      = method;
meta.variation   = variation;
meta.N           = N;
meta.seed        = seed;
meta.base        = base;
meta.lognormal_p = p_log;
meta.param_names = param_names;
meta.output_dir  = output_dir;
meta.timestamp   = datestr(now,'yyyy-mm-dd HH:MM:SS');

if strcmpi(method,'sobol')
    meta.sobol_n_sim = n_sim;
    meta.sobol_simulate_fraction = simulate_frac;
end

%% --- Output paths ------------------------------------------------------
fname_base = sprintf('%s_var%.0f_plog%.0f_seed%.0f', ...
    method, variation*100, p_log*100, seed);

path_csv  = fullfile(output_dir, [fname_base '.csv']);
path_json = fullfile(output_dir, [fname_base '.json']);

%% --- Export ------------------------------------------------------------
writetable(T, path_csv, 'Delimiter',';');

fid_json = fopen(path_json,'w');
fprintf(fid_json,'%s', jsonencode(struct('meta',meta,'n_cases',N), ...
    'PrettyPrint',true));
fclose(fid_json);

disp("Parameter sampling completed:");
disp("   → " + path_csv);

end
