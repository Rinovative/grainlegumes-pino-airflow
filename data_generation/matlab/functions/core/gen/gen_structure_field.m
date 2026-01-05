%% gen_structure_field.m
% ============================================================
% Generate synthetic 2D multiscale structure fields z(x,y)
% for heterogeneous porous or granular media.
%
% Author: Rino M. Albertin
% Date:   2026-01-04
%
% DESCRIPTION
%   Generates stochastic two-dimensional, dimensionless structure fields
%   z(x,y) representing the geometric and statistical backbone of a
%   heterogeneous porous medium.
%
%   The structure field encodes spatial organization, connectivity,
%   dominant regions, and sub-scale variability, but carries NO physical
%   meaning by itself. It serves as a shared latent material descriptor
%   from which different physical fields (e.g. permeability, porosity)
%   can later be derived in a physically consistent but decoupled manner.
%
%   The generator is based on correlated Gaussian random fields and models
%   multiscale heterogeneity through a clear separation of roles:
%     - a BASE-scale correlated field
%         defines the large-scale backbone of the medium, controlling
%         connectivity, dominant regions, and global organization
%     - a SMOOTH-scale correlated field
%         introduces finer-scale modulation and texture, smoothing
%         transitions and representing sub-dominant heterogeneity
%
%   Both fields are normalized independently and combined by weighted
%   superposition to form a background structure field z_bg(x,y).
%
%   Additional localized heterogeneities (e.g. clusters, lenses, or
%   particle-scale variability) are introduced as stochastic Gaussian
%   perturbations directly in structure space and added to z_bg.
%
%   The final structure field z(x,y) is normalized once globally to ensure
%   statistical consistency across realizations.
%
%   No physical mapping (e.g. permeability, porosity, tensors) is applied
%   in this function.
%
%   Algorithm overview:
%       1. Generate Gaussian noise fields for base and smooth scales
%       2. Introduce optional cross-scale coupling
%       3. Apply Gaussian smoothing kernels
%            - smooth scale: anisotropic
%            - base scale: isotropic
%       4. Normalize each scale independently
%       5. Combine base and smooth scales by weighted superposition
%       6. Optionally add localized Gaussian perturbations in structure space
%       7. Normalize the final structure field
%
% ------------------------------------------------------------------------
% USAGE
%   [fields, info] = gen_structure_field(Lx, Ly, res, seed, opts)
%
% INPUTS
%   Lx, Ly        – Domain size [m]
%   res           – Grid spacing [m]
%   seed          – Random seed for reproducibility
%
% OPTIONAL STRUCT opts (BACKGROUND HETEROGENEITY)
%   .base_len_rel     scalar
%       Relative correlation length of the BASE-scale field
%   .smooth_len_rel   scalar
%       Relative correlation length of the SMOOTH-scale field
%   .ms_weight        [w_base, w_smooth]
%       Weights for combining base and smooth scales (sum = 1)
%   .anisotropy       [ax, ay]
%       Correlation stretch factors applied ONLY to the smooth-scale field
%   .coupling         scalar
%       Cross-scale coupling coefficient (0 = independent, 1 = identical)
%
% OPTIONAL STRUCT opts (LOCALIZED NOISES)
%   .noise_level         scalar in [0,1]
%       Relative strength of sub-scale structural perturbations
%   .noise_granularity   scalar in [0,1]
%       Morphology control for localized heterogeneities
%   .noise_bias          scalar in [0,1]
%       Probability of positive vs. negative perturbations
%
% OUTPUTS
%   fields.structure.z_base
%   fields.structure.z_smooth
%   fields.structure.z_bg
%   fields.structure.z_noises
%   fields.structure.z
%
%   fields.grid.X, fields.grid.Y
%   info – struct containing geometry, parameters, and statistics
% ============================================================

function [fields, info] = gen_structure_field(Lx, Ly, res, seed, opts)

%% === RNG setup ==============================================
rng(seed, 'twister');
rng_state = rng;

%% === Default options & hooks ================================
if nargin < 5
    opts = struct();
end

% --- Hooks ---------------------------------------------------
if ~isfield(opts,'enable_hooks'), opts.enable_hooks = false; end
if ~isfield(opts,'hooks'),        opts.hooks = struct(); end

call_hook = @(name,data) ...
    (opts.enable_hooks ...
     && isfield(opts.hooks,name) ...
     && isa(opts.hooks.(name),'function_handle')) ...
     && opts.hooks.(name)(data);

%% === Background heterogeneity ===============================
if ~isfield(opts,'base_len_rel'),    opts.base_len_rel   = 0.10; end
if ~isfield(opts,'smooth_len_rel'),  opts.smooth_len_rel = 0.05; end
if ~isfield(opts,'ms_weight'),       opts.ms_weight      = [0.4, 0.6]; end
if ~isfield(opts,'anisotropy'),      opts.anisotropy     = [3, 1]; end
if ~isfield(opts,'coupling'),        opts.coupling       = 0.5; end

%% === Localized Gaussian noises ===============================
if ~isfield(opts,'noise_level'),       opts.noise_level = 0.2; end
if ~isfield(opts,'noise_granularity'), opts.noise_granularity = 0.5; end
if ~isfield(opts,'noise_bias'),        opts.noise_bias  = 0.5; end

%% === Initialize info ========================================
info = struct();
info.statistics = struct();
info.statistics.noise = struct('max_abs',0,'l2_norm',0);

%% === Spatial grid ==========================================
dx = res; dy = res;
nx = round(Lx/dx) + 1;
ny = round(Ly/dy) + 1;

x = linspace(0, Lx, nx);
y = linspace(0, Ly, ny);
[X, Y] = meshgrid(x, y);

fields.grid.X = X;
fields.grid.Y = Y;

%% === Correlation kernels ===================================
len_base   = opts.base_len_rel   * Lx;
len_smooth = opts.smooth_len_rel * Lx;

ax = opts.anisotropy(1);
ay = opts.anisotropy(2);

sigma_smooth_x = (len_smooth / sqrt(8*log(2))) / dx * ax;
sigma_smooth_y = (len_smooth / sqrt(8*log(2))) / dy * ay;
sigma_base     = (len_base   / sqrt(8*log(2))) / dx;

ks = ceil(6 * max([sigma_smooth_x, sigma_smooth_y, sigma_base]));
[xk, yk] = meshgrid(-ks:ks, -ks:ks);

G_smooth = exp(-((xk.^2)/(2*sigma_smooth_x^2) + (yk.^2)/(2*sigma_smooth_y^2)));
G_base   = exp(-((xk.^2 + yk.^2)/(2*sigma_base^2)));

G_smooth = G_smooth / sum(G_smooth(:));
G_base   = G_base   / sum(G_base(:));

%% === Multiscale structure field =============================
z_seed_base   = randn(ny,nx);
z_uncorr      = randn(ny,nx);
z_seed_smooth = opts.coupling * z_seed_base ...
              + sqrt(1 - opts.coupling^2) * z_uncorr;

z_base   = conv2(z_seed_base,   G_base,   'same');
z_smooth = conv2(z_seed_smooth, G_smooth, 'same');
call_hook('filtered_fields', struct( ...
    'base_field',   z_base, ...
    'smooth_field', z_smooth ));

z_base   = (z_base   - mean(z_base(:)))   / std(z_base(:));
z_smooth = (z_smooth - mean(z_smooth(:))) / std(z_smooth(:));

z_bg = opts.ms_weight(1)*z_base + opts.ms_weight(2)*z_smooth;
call_hook('structure_field_bg', struct('z_bg', z_bg));

%% === Sub-scale localized noises =============================
z_noises = zeros(size(z_bg));

if opts.noise_level > 0

    level = opts.noise_level;
    chi   = opts.noise_granularity;
    bias  = opts.noise_bias;

    Xn = X / Lx;
    Yn = Y / Ly;

    len0 = opts.base_len_rel;
    sigma_min = len0 / 10;
    sigma_max = len0;
    sigma_char = sigma_min * (sigma_max / sigma_min)^(1 - chi);

    area_dom   = 1.0;
    area_noise = pi * sigma_char^2;

    n_mean   = level * area_dom / max(area_noise, eps);
    n_noises = poissrnd(n_mean);

    noise_field = zeros(size(z_bg));
    s_spread = log(2);

    for i = 1:n_noises

        cx = rand;
        cy = rand;

        sigma  = sigma_char * exp(0.5 * s_spread * randn);
        aspect = exp(s_spread * randn);

        if rand < 0.5
            sx = sigma * aspect;
            sy = sigma;
        else
            sx = sigma;
            sy = sigma * aspect;
        end

        phi = 2*pi*rand;
        c = cos(phi); s_ = sin(phi);

        Xr =  c*(Xn-cx) + s_*(Yn-cy);
        Yr = -s_*(Xn-cx) + c*(Yn-cy);

        sign_amp = (rand < bias)*2 - 1;

        noise_field = noise_field + sign_amp * exp( ...
            -(Xr.^2./(2*sx^2) + Yr.^2./(2*sy^2)) );
    end

    noise_field = noise_field - mean(noise_field(:));
    noise_field = noise_field / max(rms(noise_field(:)), eps);

    z_noises = level * noise_field;
    call_hook('noises_scaled', struct('field', z_noises));
end

%% === Final structure field ==================================
z = z_bg + z_noises;
z = (z - mean(z(:))) / std(z(:));
call_hook('structure_field', struct( ...
    'z', z, 'z_bg', z_bg, 'z_noises', z_noises ));

fields.structure.z_base   = z_base;
fields.structure.z_smooth = z_smooth;
fields.structure.z_bg     = z_bg;
fields.structure.z_noises = z_noises;
fields.structure.z        = z;

%% === Statistics ==============================================
info.statistics.structure = struct();
info.statistics.structure.z.mean = mean(z(:));
info.statistics.structure.z.std  = std(z(:));
info.statistics.structure.z.min  = min(z(:));
info.statistics.structure.z.max  = max(z(:));

info.statistics.structure.z_bg.mean = mean(z_bg(:));
info.statistics.structure.z_bg.std = std(z_bg(:));
info.statistics.structure.z_noises.rms = rms(z_noises(:));

if opts.noise_level > 0
    info.statistics.noise.max_abs = max(abs(z_noises(:)));
    info.statistics.noise.l2_norm = norm(z_noises(:)) / numel(z_noises);
end

%% === Metadata ==============================================
info.parameters = struct( ...
    'seed', seed, ...
    'rng_state', rng_state, ...
    'background', struct( ...
        'base_len_rel',   opts.base_len_rel, ...
        'smooth_len_rel', opts.smooth_len_rel, ...
        'ms_weight',      opts.ms_weight, ...
        'anisotropy',     opts.anisotropy, ...
        'coupling',       opts.coupling ), ...
    'noise', struct( ...
        'level',       opts.noise_level, ...
        'granularity', opts.noise_granularity, ...
        'bias',        opts.noise_bias ) ...
);

end
