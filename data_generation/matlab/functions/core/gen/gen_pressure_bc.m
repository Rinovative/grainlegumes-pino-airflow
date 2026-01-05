%% gen_pressure_bc.m
% ============================================================
% Generate synthetic inlet pressure boundary condition p(y=0)
%
% Author: Rino M. Albertin
% Date:   2026-01-05
%
% DESCRIPTION
%   Generates a deterministic pressure boundary condition applied
%   exclusively at the inlet boundary y = 0.
%
%   The inlet pressure is constructed from a small set of global
%   shape parameters, enabling a wide variety of boundary "figures"
%   while keeping the DoE dimensionality low.
%
%   Uniform pressure is obtained automatically when all amplitudes
%   are set to zero.
%
% ------------------------------------------------------------------------
% FUNCTION FORM
%
%   p(x) = p_mean · ( 1
%                    + a_sin · sin(2π·f_sin·x̂)
%                    + Σ_i a_i · exp(-(x̂-μ_i)^2/(2σ_i^2))
%                    + a_lin · (2x̂ - 1) )
%
%   with:
%     μ_i   equally spaced centers
%     σ_i   = σ · (1 + jitter · ξ_i),  ξ_i ~ N(0,1)
%
% ------------------------------------------------------------------------
% USAGE
%   [fields, info] = gen_pressure_bc(fields, opts)
%
% REQUIRED OPTS (SAMPLED UPSTREAM)
%   .p_inlet_mean     scalar > 0
%
% OPTIONAL OPTS (LOW-DIMENSIONAL)
%   .a_sin            scalar, default 0
%   .f_sin            scalar >= 0, default 1
%
%   .k_gauss          integer >= 0, default 0
%   .a_gauss          scalar, default 0
%   .sigma_gauss      scalar > 0, default 0.12
%   .gauss_jitter     scalar >= 0, default 0.25
%
%   .a_lin            scalar, default 0
%
% OUTPUTS
%   fields.bc.p_inlet
%   info.parameters
%   info.statistics
% ============================================================

function [fields, info] = gen_pressure_bc(fields, opts)

%% === Defaults & hooks ======================================
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

%% === Required parameter ====================================
if ~isfield(opts,'p_inlet_mean')
    error('gen_pressure_bc:MissingMeanPressure', ...
        'opts.p_inlet_mean must be provided (sampled inlet pressure).');
end

%% === Fixed safety bound ====================================
p_min = 0.0;   % FIXED, internal

%% === Defaults (shape parameters) ===========================
if ~isfield(opts,'a_sin'),        opts.a_sin = 0.0; end
if ~isfield(opts,'f_sin'),        opts.f_sin = 1.0; end

if ~isfield(opts,'k_gauss'),      opts.k_gauss = 0; end
if ~isfield(opts,'a_gauss'),      opts.a_gauss = 0.0; end
if ~isfield(opts,'sigma_gauss'),  opts.sigma_gauss = 0.12; end
if ~isfield(opts,'gauss_jitter'), opts.gauss_jitter = 0.25; end

if ~isfield(opts,'a_lin'),        opts.a_lin = 0.0; end

%% === Extract grid ==========================================
X = fields.grid.X;
Y = fields.grid.Y;

nx = size(X,2);

% robust y = 0 detection
y0_mask = abs(Y(:,1) - min(Y(:))) < 1e-12;
if ~any(y0_mask)
    error('gen_pressure_bc:NoY0Boundary', ...
        'No grid row corresponds to y = 0.');
end

x = X(y0_mask, :);
x_hat = (x - min(x)) / max(max(x) - min(x), eps);

%% === Shape construction ====================================
shape = zeros(1, nx);

% --- Sinus --------------------------------------------------
if opts.a_sin ~= 0
    shape = shape + opts.a_sin * ...
        sin(2*pi*opts.f_sin*x_hat + opts.phi_sin);
end

% --- Gaussian bumps -----------------------------------------
if opts.k_gauss > 0 && opts.a_gauss ~= 0

    k = opts.k_gauss;

    % equally spaced centers in (0,1)
    mu = linspace(0, 1, k+2);
    mu = mu(2:end-1);

    % per-bump sigma jitter
    sigma0 = opts.sigma_gauss;
    jitter = opts.gauss_jitter;

    sigma_i = sigma0 * (1 + jitter * randn(1,k));
    sigma_i = max(sigma_i, 0.05 * sigma0);

    % normalized amplitudes
    a_i = opts.a_gauss / max(k,1);

    for i = 1:k
        shape = shape + a_i * ...
            exp(-(x_hat - mu(i)).^2 ./ (2*sigma_i(i)^2));
    end
end

% --- Linear gradient ----------------------------------------
if opts.a_lin ~= 0
    shape = shape + opts.a_lin * (2*x_hat - 1);
end

%% === Final inlet pressure ==================================
p_mean  = opts.p_inlet_mean;
p_inlet = p_mean * (1 + shape);

% safety clipping
p_inlet = max(p_inlet, p_min);

call_hook('p_inlet', struct('p_inlet', p_inlet));

%% === Store in fields =======================================
fields.bc = struct();
fields.bc.p_inlet = p_inlet;

%% === Statistics ============================================
info.statistics.p_inlet.mean = mean(p_inlet);
info.statistics.p_inlet.std  = std(p_inlet);
info.statistics.p_inlet.min  = min(p_inlet);
info.statistics.p_inlet.max  = max(p_inlet);

%% === Metadata ==============================================
info.parameters = struct( ...
    'p_inlet_mean', opts.p_inlet_mean, ...
    'a_sin',        opts.a_sin, ...
    'f_sin',        opts.f_sin, ...
    'k_gauss',      opts.k_gauss, ...
    'a_gauss',      opts.a_gauss, ...
    'sigma_gauss',  opts.sigma_gauss, ...
    'gauss_jitter', opts.gauss_jitter, ...
    'a_lin',        opts.a_lin );

end
