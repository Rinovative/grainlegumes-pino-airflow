%% gen_permeability_field.m
% ============================================================
% Generate synthetic 2D permeability fields κ(x,y) for porous media
%
% Author: Rino M. Albertin
% Date:   2026-01-04
%
% DESCRIPTION
%   Maps a previously generated dimensionless structure field z(x,y)
%   to physically meaningful permeability fields.
%
%   This function performs ONLY the physical interpretation step:
%     - lognormal mapping to scalar permeability κ(x,y)
%     - derivation of anisotropy magnitude a(x,y)
%     - derivation of smooth orientation field θ(x,y)
%     - construction of a symmetric permeability tensor K(x,y)
%
%   The underlying structure field is generated externally by
%   gen_structure_field.m and passed explicitly via fields.structure.
%
% ------------------------------------------------------------------------
% USAGE
%   [fields, info] = gen_permeability_field(fields, k_mean, var_rel, opts)
%
% INPUTS
%   fields        – Struct containing:
%                   fields.grid.X, fields.grid.Y
%                   fields.structure.z, fields.structure.z_bg
%   k_mean        – Mean permeability [m²]
%   var_rel       – Relative standard deviation (e.g. 0.5 = 50 %)
%
% OPTIONAL STRUCT opts
%   Same option structure as gen_structure_field.m plus:
%
%   (TENSOR CONSTRUCTION)
%   .a_max
%   .a_gamma
%   .tensor_strength
%   .theta_jitter
%   .theta_smooth_rel
%
% OUTPUTS
%   fields.material.kappa
%   fields.material.K.Kxx, Kyy, Kxy
%   fields.grid.X, fields.grid.Y
%   info – geometry, parameters, statistics
% ============================================================

function [fields, info] = gen_permeability_field(fields, opts)
%% === Default options & hooks ================================
if nargin < 2 || isempty(opts)
    opts = struct();
end

if ~isfield(opts,'enable_hooks'), opts.enable_hooks = false; end
if ~isfield(opts,'hooks'),        opts.hooks = struct(); end

call_hook = @(name,data) ...
    (opts.enable_hooks ...
     && isfield(opts.hooks,name) ...
     && isa(opts.hooks.(name),'function_handle')) ...
     && opts.hooks.(name)(data);

%% === Tensor defaults ========================================
if ~isfield(opts,'a_max'),            opts.a_max = 2.0; end
if ~isfield(opts,'a_gamma'),          opts.a_gamma = 2.0; end
if ~isfield(opts,'tensor_strength'),  opts.tensor_strength = 1.0; end
if ~isfield(opts,'theta_jitter'),     opts.theta_jitter = 0.2; end
if ~isfield(opts,'theta_smooth_rel'), opts.theta_smooth_rel = 0.02; end

%% === Structure field generation =============================
k_mean  = opts.k_mean;
var_rel = opts.var_rel;

X    = fields.grid.X;
Y    = fields.grid.Y;

z    = fields.structure.z;
z_bg = fields.structure.z_bg;

dx = X(1,2) - X(1,1);
dy = Y(2,1) - Y(1,1);

%% === Lognormal permeability mapping =========================
s_logn = sqrt(log(1 + var_rel^2));
logk   = s_logn * z - 0.5 * s_logn^2;

kappa = k_mean .* exp(logk);
fields.material.kappa = kappa;
call_hook('kappa_final', struct('kappa', kappa));

%% === Tensor construction ====================================
% (A) smoothing kernel for orientation
Lx = X(1,end) - X(1,1);
sigma_t = (opts.theta_smooth_rel * Lx) / dx;
sigma_t = max(sigma_t, 1.0);

kt = ceil(6 * sigma_t);
[xkt, ykt] = meshgrid(-kt:kt, -kt:kt);

Gt = exp(-(xkt.^2 + ykt.^2) / (2 * sigma_t^2));
Gt = Gt / sum(Gt(:));

% (B) anisotropy magnitude from backbone structure
z_abs = abs(z_bg) / max(abs(z_bg(:)));
a = 1 + opts.tensor_strength * ((opts.a_max - 1) * (z_abs .^ opts.a_gamma));
call_hook('anisotropy_ratio', struct('a', a));

% (C) orientation from structure gradients
[dzdy, dzdx] = gradient(z_bg, dy, dx);
theta_raw = atan2(dzdy, dzdx);

% director formulation (π-periodic)
dir_x = cos(2 * theta_raw);
dir_y = sin(2 * theta_raw);

dir_x = conv2(dir_x, Gt, 'same');
dir_y = conv2(dir_y, Gt, 'same');

dir_n = sqrt(dir_x.^2 + dir_y.^2);
dir_x = dir_x ./ max(dir_n, eps);
dir_y = dir_y ./ max(dir_n, eps);

% optional jitter
if opts.theta_jitter > 0
    dir_x = dir_x + opts.theta_jitter * conv2(randn(size(dir_x)), Gt, 'same');
    dir_y = dir_y + opts.theta_jitter * conv2(randn(size(dir_y)), Gt, 'same');

    dir_n = sqrt(dir_x.^2 + dir_y.^2);
    dir_x = dir_x ./ max(dir_n, eps);
    dir_y = dir_y ./ max(dir_n, eps);
end

theta = 0.5 * atan2(dir_y, dir_x);

% (D) principal permeabilities
k1 = kappa .* a;
k2 = kappa ./ a;
call_hook('principal_k', struct('k1', k1, 'k2', k2));

% (E) rotate into tensor
c = cos(theta); s = sin(theta);

K.Kxx = k1 .* c.^2 + k2 .* s.^2;
K.Kyy = k1 .* s.^2 + k2 .* c.^2;
K.Kxy = (k1 - k2) .* s .* c;

fields.material.K = K;
call_hook('tensor', struct('K', K, 'a', a, 'theta', theta));

%% === Statistics =============================================
info.statistics.kappa.mean = mean(kappa(:));
info.statistics.kappa.std  = std(kappa(:));
info.statistics.kappa.min  = min(kappa(:));
info.statistics.kappa.max  = max(kappa(:));

traceK = K.Kxx + K.Kyy;
detK   = K.Kxx .* K.Kyy - K.Kxy.^2;
call_hook('tensor_checks', struct( ...
    'trace', K.Kxx + K.Kyy, ...
    'det',   K.Kxx .* K.Kyy - K.Kxy.^2 ));

info.statistics.tensor.trace.mean = mean(traceK(:));
info.statistics.tensor.det.mean   = mean(detK(:));

%% === Metadata ===============================================
info.parameters = struct( ...
    'permeability', struct( ...
        'k_mean',  opts.k_mean, ...
        'var_rel', opts.var_rel, ...
        's_logn',  s_logn ), ...
    'tensor', struct( ...
        'a_max',           opts.a_max, ...
        'a_gamma',         opts.a_gamma, ...
        'tensor_strength', opts.tensor_strength ), ...
    'orientation', struct( ...
        'theta_jitter',     opts.theta_jitter, ...
        'theta_smooth_rel', opts.theta_smooth_rel ) ...
);

end
