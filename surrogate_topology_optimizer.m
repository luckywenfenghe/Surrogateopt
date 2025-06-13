%% COMPLETE SURROGATE HPO FOR TOPOLOGY OPTIMIZATION
% Implementation following suggest.txt code review recommendations
% Addresses all identified bugs and performance issues

clear; close all; clc;

% Declare global variables first
global FAST_ITERATIONS COARSE_MESH_SIZE;

fprintf('=== ROBUST SURROGATE HPO FOR TOPOLOGY OPTIMIZATION ===\n');

%% OPTIMIZED PARALLEL POOL MANAGEMENT (NO REPEATED RESTART)
% Only create pool if none exists - avoid expensive restart
try
    p = gcp('nocreate');
    if isempty(p)
        fprintf('Starting parallel pool with 4 workers (one-time setup)...\n');
        p = parpool('local', 4);
        % FIXED: Prevent IdleTimeout pool closure (suggest_for_adjust.txt §1)
        p.IdleTimeout = Inf;
        fprintf('Parallel pool ready with %d workers (IdleTimeout=Inf)\n', p.NumWorkers);
    else
        fprintf('Reusing existing parallel pool with %d workers\n', p.NumWorkers);
        % Ensure IdleTimeout is set
        if p.IdleTimeout ~= Inf
            p.IdleTimeout = Inf;
            fprintf('Set IdleTimeout=Inf to prevent pool closure\n');
        end
    end
catch ME
    fprintf('Warning: Parallel pool setup failed: %s\n', ME.message);
    fprintf('Continuing with serial computation...\n');
end

%% OPTIMIZED CONFIGURATION (FAST HPO MODE)
LEGACY_WARMUP_RUNS = 8;      % Reduced from 20 → 8 for speed
SURROGATE_MAX_EVALS = 30;    % Reduced from 50 → 30 for speed  
BATCH_SIZE = 3;              % Parallel batch size for surrogate
SAVE_RESULTS = true;         % Save all results
TIMEOUT_MINUTES = 3;         % Reduced timeout for faster failure detection

% PERFORMANCE SETTINGS (MAKE GLOBAL FOR WRAPPER ACCESS)
COARSE_MESH_SIZE = 40;       % Coarse mesh for HPO phase (was 80)
FINE_MESH_SIZE = 80;         % Fine mesh for final validation
FAST_ITERATIONS = 15;        % Fast iterations for HPO (was 40)
FULL_ITERATIONS = 120;       % Full iterations for final validation

% Set global values for wrapper functions
FAST_ITERATIONS = 15;
COARSE_MESH_SIZE = 40;
fprintf('Performance settings: Mesh %dx%d → %dx%d, Iterations %d → %d\n', ...
    COARSE_MESH_SIZE, COARSE_MESH_SIZE, FINE_MESH_SIZE, FINE_MESH_SIZE, ...
    FAST_ITERATIONS, FULL_ITERATIONS);

% Parameter bounds: [beta_init, qa_factor, mv_factor, rmin_factor]
lb = [0.5, 0.7, 0.7, 0.7];
ub = [3.0, 1.4, 1.4, 1.4];
param_names = {'beta_init', 'qa_growth_factor', 'mv_adaptation_rate', 'rmin_decay_rate'};

fprintf('Parameter space: [%.1f-%.1f, %.1f-%.1f, %.1f-%.1f, %.1f-%.1f]\n', ...
    lb(1), ub(1), lb(2), ub(2), lb(3), ub(3), lb(4), ub(4));
fprintf('Legacy warm-start: %d evaluations\n', LEGACY_WARMUP_RUNS);
fprintf('Surrogate exploration: %d evaluations\n', SURROGATE_MAX_EVALS);

%% PHASE 1: LEGACY WARM-START DATA GENERATION
fprintf('\n--- Phase 1: Legacy Warm-Start Generation ---\n');

% Initialize storage
X_warmstart = zeros(LEGACY_WARMUP_RUNS, 4);
F_warmstart = zeros(LEGACY_WARMUP_RUNS, 1);
metrics_history = cell(LEGACY_WARMUP_RUNS, 1);

% Starting parameters (reasonable defaults)
current_params = [2.0, 1.0, 1.0, 1.0];

% Timing
warmstart_time = tic;

% Generate legacy warm-start data
for i = 1:LEGACY_WARMUP_RUNS
    fprintf('Legacy %d/%d: [%.2f, %.2f, %.2f, %.2f]', ...
        i, LEGACY_WARMUP_RUNS, current_params(1), current_params(2), ...
        current_params(3), current_params(4));
    
    % Evaluate with current parameters (with timeout protection)
    eval_start = tic;
    try
        [obj, metrics] = topology_wrapper_with_metrics(current_params);
        eval_time = toc(eval_start);
        
        % Check for reasonable evaluation time
        if eval_time > TIMEOUT_MINUTES * 60
            fprintf(' [TIMEOUT: %.1f min]', eval_time/60);
            obj = 1e6;  % Penalty for slow evaluations
            metrics.converged = false;
        end
    catch ME
        fprintf(' [ERROR: %s]', ME.message);
        obj = 1e6;
        metrics = struct('obj_raw', 1e6, 'gray', 100, 'vol', 0, ...
                        'converged', false, 'iterations', 0, 'time', 0, 'change', 0);
    end
    
    % Store results
    X_warmstart(i, :) = current_params;
    F_warmstart(i) = obj;
    metrics_history{i} = metrics;
    
    if metrics.converged
        conv_str = 'Y';
    else
        conv_str = 'N';
    end
    fprintf(' → Obj: %.3e, Gray: %.1f%%, Conv: %s\n', ...
        obj, metrics.gray, conv_str);
    
    % Update using legacy logic
    if i < LEGACY_WARMUP_RUNS
        current_params = legacy_parameter_update(current_params, metrics, i);
        % Enforce bounds
        current_params = max(current_params, lb);
        current_params = min(current_params, ub);
    end
end

warmstart_time = toc(warmstart_time);

% Find best warm-start result
[best_warmstart_f, best_idx] = min(F_warmstart);
best_warmstart_x = X_warmstart(best_idx, :);

fprintf('Warm-start completed in %.2f min\n', warmstart_time/60);
fprintf('Best legacy result: F=%.3e at [%.2f, %.2f, %.2f, %.2f]\n', ...
    best_warmstart_f, best_warmstart_x(1), best_warmstart_x(2), ...
    best_warmstart_x(3), best_warmstart_x(4));

%% PHASE 2: SURROGATE OPTIMIZATION
fprintf('\n--- Phase 2: Surrogate Optimization ---\n');

% Check if Global Optimization Toolbox is available
if license('test', 'GADS_Toolbox')
    try
        % Configure surrogate optimization (following suggest.txt §4.1)
        try
            % FIXED: Proper table construction (suggest_for_adjust.txt §2)
            initial_table = array2table(X_warmstart, ...
                'VariableNames', {'x1','x2','x3','x4'});
            initial_table.objective = F_warmstart;
            
            options = optimoptions('surrogateopt', ...
                'InitialPoints', initial_table, ...
                'MaxFunctionEvaluations', SURROGATE_MAX_EVALS, ...
                'Display', 'iter', ...
                'UseParallel', false, ...  % Keep false to avoid nested parallel issues
                'MinSampleDistance', 0.08);
            fprintf('Using InitialPoints table for warm-start\n');
            
        catch
            % Fallback to basic options without initial points
            options = optimoptions('surrogateopt', ...
                'MaxFunctionEvaluations', SURROGATE_MAX_EVALS + LEGACY_WARMUP_RUNS, ...
                'Display', 'iter', ...
                'UseParallel', false, ...
                'MinSampleDistance', 0.08);
            fprintf('Using basic surrogateopt (will pre-evaluate warm points)\n');
            
            % Pre-evaluate warm-start points
            fprintf('Pre-evaluating %d warm-start points...\n', LEGACY_WARMUP_RUNS);
            for i = 1:LEGACY_WARMUP_RUNS
                f_check = topology_wrapper(X_warmstart(i,:));
                fprintf('  Warm %d: f=%.3e\n', i, f_check);
            end
        end
        
        % Run surrogate optimization
        surrogate_time = tic;
        [x_optimal, f_optimal, exitflag, output] = surrogateopt(@topology_wrapper, lb, ub, options);
        surrogate_time = toc(surrogate_time);
        
        fprintf('Surrogate optimization completed\n');
        fprintf('Exit flag: %d, Total evaluations: %d\n', exitflag, output.funccount);
        
    catch ME
        fprintf('Surrogate optimization failed: %s\n', ME.message);
        fprintf('Using best legacy result as optimal\n');
        x_optimal = best_warmstart_x;
        f_optimal = best_warmstart_f;
        surrogate_time = 0;
    end
else
    fprintf('Global Optimization Toolbox not available\n');
    fprintf('Using best legacy result as optimal\n');
    x_optimal = best_warmstart_x;
    f_optimal = best_warmstart_f;
    surrogate_time = 0;
end

%% PHASE 3: FINAL HIGH-FIDELITY VALIDATION
fprintf('\n--- Phase 3: High-Fidelity Validation ---\n');

% Run final validation with FINE MESH and FULL ITERATIONS
params_hifi = struct();
params_hifi.enable_surrogate_mode = true;
params_hifi.beta_init = x_optimal(1);
params_hifi.qa_growth_factor = x_optimal(2);
params_hifi.mv_adaptation_rate = x_optimal(3);
params_hifi.rmin_decay_rate = x_optimal(4);
params_hifi.max_iterations = FULL_ITERATIONS;  % Full iterations
params_hifi.nely = FINE_MESH_SIZE;            % Fine mesh for accuracy
params_hifi.force_serial = true;              % Keep serial for stability
params_hifi.disable_plotting = false;         % Enable plotting for final result

fprintf('Running high-fidelity validation (%dx%d mesh, %d iterations)...\n', ...
    FINE_MESH_SIZE, FINE_MESH_SIZE, FULL_ITERATIONS);
final_result = topFlow_mpi_robust(params_hifi);

%% RESULTS ANALYSIS AND SUMMARY
fprintf('\n=== OPTIMIZATION RESULTS SUMMARY ===\n');

% Parameter results
fprintf('Optimal parameters:\n');
for i = 1:length(param_names)
    fprintf('  %s: %.3f\n', param_names{i}, x_optimal(i));
end

% Objective comparison
fprintf('\nObjective comparison:\n');
fprintf('  Legacy best: %.3e\n', best_warmstart_f);
fprintf('  Surrogate best: %.3e\n', f_optimal);
fprintf('  High-fidelity final: %.3e\n', final_result.obj);

% Performance metrics
if f_optimal < best_warmstart_f
    improvement = (best_warmstart_f - f_optimal) / best_warmstart_f * 100;
    fprintf('  Improvement: +%.2f%%\n', improvement);
else
    degradation = (f_optimal - best_warmstart_f) / best_warmstart_f * 100;
    fprintf('  Change: %.2f%% (exploration)\n', degradation);
end

% Final metrics
fprintf('\nFinal design metrics:\n');
fprintf('  Objective: %.3e\n', final_result.obj);
fprintf('  Grayscale: %.1f%%\n', final_result.gray);
fprintf('  Volume fraction: %.3f\n', final_result.vol);
if final_result.converged
    fprintf('  Convergence: Yes\n');
else
    fprintf('  Convergence: No\n');
end
fprintf('  Iterations: %d\n', final_result.iterations);
fprintf('  Time: %.2f sec\n', final_result.time);

% Timing summary
total_time = warmstart_time + surrogate_time + final_result.time;
fprintf('\nTiming summary:\n');
fprintf('  Warm-start phase: %.2f min\n', warmstart_time/60);
if surrogate_time > 0
    fprintf('  Surrogate phase: %.2f min\n', surrogate_time/60);
end
fprintf('  Final validation: %.2f min\n', final_result.time/60);
fprintf('  Total time: %.2f min\n', total_time/60);

%% SAVE RESULTS
if SAVE_RESULTS
    results = struct();
    results.X_warmstart = X_warmstart;
    results.F_warmstart = F_warmstart;
    results.metrics_history = metrics_history;
    results.x_optimal = x_optimal;
    results.f_optimal = f_optimal;
    results.final_result = final_result;
    results.best_warmstart_x = best_warmstart_x;
    results.best_warmstart_f = best_warmstart_f;
    results.warmstart_time = warmstart_time;
    results.surrogate_time = surrogate_time;
    results.total_time = total_time;
    results.param_names = param_names;
    results.bounds = [lb; ub];
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filename = sprintf('surrogate_results_%s.mat', timestamp);
    save(filename, 'results');
    fprintf('\nResults saved to %s\n', filename);
end

fprintf('\n=== SURROGATE HPO COMPLETED ===\n');

%% PARALLEL POOL MANAGEMENT
% Keep pool alive for potential future runs
try
    p = gcp('nocreate');
    if ~isempty(p)
        fprintf('Parallel pool remains active for future use (%d workers)\n', p.NumWorkers);
        fprintf('Use "delete(gcp)" to manually close if needed\n');
    end
catch
    % Ignore pool status errors
end

%% TOPOLOGY OPTIMIZATION WRAPPER (§4.2 from suggest.txt)
function f = topology_wrapper(x)
    % Robust wrapper following code review recommendations
    
    % Get global performance settings
    global FAST_ITERATIONS COARSE_MESH_SIZE;
    if isempty(FAST_ITERATIONS); FAST_ITERATIONS = 15; end
    if isempty(COARSE_MESH_SIZE); COARSE_MESH_SIZE = 40; end
    
    % Parameter validation
    x = max(x, [0.5, 0.7, 0.7, 0.7]);  % Lower bounds
    x = min(x, [3.0, 1.4, 1.4, 1.4]);  % Upper bounds
    
    % Build parameter structure (FAST HPO MODE)
    params = struct();
    params.enable_surrogate_mode = true;
    params.beta_init = x(1);
    params.qa_growth_factor = x(2);
    params.mv_adaptation_rate = x(3);
    params.rmin_decay_rate = x(4);
    params.max_iterations = FAST_ITERATIONS;  % Fast evaluation for HPO
    params.nely = COARSE_MESH_SIZE;          % Coarse mesh for speed
    params.force_serial = true;              % Avoid pool conflicts
    params.disable_plotting = true;         % No plotting during HPO
    
    try
        % Call robust topology optimizer
        result = topFlow_mpi_robust(params);
        
        % FIXED: Rebalanced objective weights (suggest_for_adjust.txt §2)
        % Normalize physical objective to [0,1] range
        obj_normalized = result.obj / 1000.0;  % Assuming ~1000 is typical range
        
        f = obj_normalized ...                      % Normalized physical objective
          + 0.5 * (result.gray/100)^2 ...          % Reduced grayscale penalty
          + (~result.converged) * 0.1;             % Reduced convergence penalty
        
        % Ensure finite result
        if ~isfinite(f)
            f = 1e6;
        end
        
    catch ME
        fprintf('  ERROR in topology evaluation: %s\n', ME.message);
        f = 1e6;  % Large penalty for failed evaluations
    end
end

%% TOPOLOGY WRAPPER WITH DETAILED METRICS (FOR WARM-START PHASE)
function [obj, metrics] = topology_wrapper_with_metrics(x)
    % Extended wrapper that returns both objective and detailed metrics
    
    % Get global performance settings
    global FAST_ITERATIONS COARSE_MESH_SIZE;
    if isempty(FAST_ITERATIONS); FAST_ITERATIONS = 15; end
    if isempty(COARSE_MESH_SIZE); COARSE_MESH_SIZE = 40; end
    
    % Parameter validation
    x = max(x, [0.5, 0.7, 0.7, 0.7]);  % Lower bounds
    x = min(x, [3.0, 1.4, 1.4, 1.4]);  % Upper bounds
    
    % Build parameter structure (FAST HPO MODE)
    params = struct();
    params.enable_surrogate_mode = true;
    params.beta_init = x(1);
    params.qa_growth_factor = x(2);
    params.mv_adaptation_rate = x(3);
    params.rmin_decay_rate = x(4);
    params.max_iterations = FAST_ITERATIONS;  % Fast evaluation for HPO
    params.nely = COARSE_MESH_SIZE;          % Coarse mesh for speed
    params.force_serial = true;              % Avoid pool conflicts
    params.disable_plotting = true;         % No plotting during HPO
    
    try
        % Call robust topology optimizer
        result = topFlow_mpi_robust(params);
        
        % FIXED: Rebalanced objective weights (suggest_for_adjust.txt §2)
        % Normalize physical objective to [0,1] range
        obj_normalized = result.obj / 1000.0;  % Assuming ~1000 is typical range
        
        obj = obj_normalized ...                    % Normalized physical objective
            + 0.5 * (result.gray/100)^2 ...        % Reduced grayscale penalty
            + (~result.converged) * 0.1;           % Reduced convergence penalty
        
        % Extract detailed metrics for legacy update logic
        metrics = struct();
        metrics.obj_raw = result.obj;
        metrics.gray = result.gray;
        metrics.vol = result.vol;
        metrics.converged = result.converged;
        metrics.iterations = result.iterations;
        metrics.time = result.time;
        
        % Calculate change metric (approximation for legacy logic)
        persistent prev_obj;
        if isempty(prev_obj)
            metrics.change = 0.1;  % Initial change
        else
            metrics.change = abs(result.obj - prev_obj) / max(abs(prev_obj), 1e-6);
        end
        prev_obj = result.obj;
        
        % Ensure finite result
        if ~isfinite(obj)
            obj = 1e6;
            metrics.converged = false;
        end
        
    catch ME
        fprintf('  ERROR in topology evaluation: %s\n', ME.message);
        obj = 1e6;
        metrics = struct('obj_raw', 1e6, 'gray', 100, 'vol', 0, ...
                        'converged', false, 'iterations', 0, 'time', 0, 'change', 0);
    end
end

%% LEGACY PARAMETER UPDATE LOGIC (EXTRACTED FROM ORIGINAL)
function params_next = legacy_parameter_update(params, metrics, iteration)
    % Legacy logic extracted from original topology optimization
    params_next = params;
    
    % Beta adaptation based on convergence and grayscale
    if iteration > 8 && metrics.change < 0.01
        params_next(1) = min(params(1) * 1.08, 3.0);
    end
    if metrics.gray > 25
        params_next(1) = min(params(1) * 1.05, 3.0);
    end
    
    % QA factor based on progress
    if metrics.gray > 20
        params_next(2) = min(params(2) * 1.15, 1.4);
    elseif metrics.gray < 10 && metrics.converged
        params_next(2) = max(params(2) * 0.95, 0.7);
    end
    
    % Move limit factor based on convergence rate
    if metrics.change > 0.008
        params_next(3) = min(params(3) * 1.12, 1.4);
    elseif metrics.change < 0.002
        params_next(3) = max(params(3) * 0.92, 0.7);
    end
    
    % Filter factor gradual refinement
    if iteration > 5 && mod(iteration, 3) == 0
        params_next(4) = max(params(4) * 0.97, 0.7);
    end
    
    % Add exploration noise for diversity
    if iteration > 10
        noise = 0.02 * randn(1, 4);
        params_next = params_next + noise;
    end
end

fprintf('Surrogate topology optimizer ready. Run this script to execute.\n'); 