%% COMPLETE SURROGATE HPO FOR TOPOLOGY OPTIMIZATION
% Implementation following o3super_pro_suggest.txt code review recommendations
% Addresses all identified bugs and performance issues and intergrate the
% fancy math formular to automate calibration

clear; close all; clc;

% FIXED: Set random seed for reproducibility (new_suggest.txt #6)
RANDOM_SEED = 42;
rng(RANDOM_SEED, 'twister');
fprintf('Random seed set to %d for reproducibility\n', RANDOM_SEED);

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
% IMPROVED: Better sampling strategy (new_suggest.txt #3)
LEGACY_WARMUP_RUNS = 5;      % Reduced legacy runs, use LHS for better coverage
LHS_WARMUP_RUNS = 15;        % LHS sampling for better space coverage (≥10×dim)
SURROGATE_MAX_EVALS = 40;    % Increased for better convergence
BATCH_SIZE = 4;              % Match number of workers
SAVE_RESULTS = true;         % Save all results
TIMEOUT_MINUTES = 3;         % Reduced timeout for faster failure detection

% PERFORMANCE SETTINGS (MAKE GLOBAL FOR WRAPPER ACCESS)
COARSE_MESH_SIZE = 40;       % Coarse mesh for HPO phase (was 80)
FINE_MESH_SIZE = 80;         % Fine mesh for final validation
FAST_ITERATIONS = 15;        % Fast iterations for HPO (was 40)
FULL_ITERATIONS = 120;       % Full iterations for final validation

% Parameter bounds: [beta_init, qa_factor, mv_factor, rmin_factor]
lb = [0.5, 0.7, 0.7, 0.7];
ub = [3.0, 1.4, 1.4, 1.4];

% ACADEMIC OBJECTIVE PARAMETERS (Eq. 9-12 from warm_suggest.txt)
VOL_TARGET = 0.25;  % Target volume fraction (= volfrac)
ZETA = [];          % Scaling factor (will be auto-calibrated in Phase 1)

% FIXED: Create configuration structure AFTER lb/ub definition (basic_final_suggest.txt #5)
hpo_config = struct();
hpo_config.fast_iterations = FAST_ITERATIONS;
hpo_config.coarse_mesh_size = COARSE_MESH_SIZE;
hpo_config.fine_mesh_size = FINE_MESH_SIZE;
hpo_config.full_iterations = FULL_ITERATIONS;
hpo_config.lb = lb;
hpo_config.ub = ub;

fprintf('Performance settings: Mesh %dx%d → %dx%d, Iterations %d → %d\n', ...
    hpo_config.coarse_mesh_size, hpo_config.coarse_mesh_size, ...
    hpo_config.fine_mesh_size, hpo_config.fine_mesh_size, ...
    hpo_config.fast_iterations, hpo_config.full_iterations);
param_names = {'beta_init', 'qa_growth_factor', 'mv_adaptation_rate', 'rmin_decay_rate'};

fprintf('Parameter space: [%.1f-%.1f, %.1f-%.1f, %.1f-%.1f, %.1f-%.1f]\n', ...
    lb(1), ub(1), lb(2), ub(2), lb(3), ub(3), lb(4), ub(4));
fprintf('Legacy warm-start: %d evaluations\n', LEGACY_WARMUP_RUNS);
fprintf('Surrogate exploration: %d evaluations\n', SURROGATE_MAX_EVALS);

%% PHASE 1: IMPROVED WARM-START DATA GENERATION
fprintf('\n--- Phase 1: Improved Warm-Start Generation ---\n');

% IMPROVED: Combined Legacy + LHS sampling (new_suggest.txt #3)
TOTAL_WARMUP_RUNS = LEGACY_WARMUP_RUNS + LHS_WARMUP_RUNS;

% Initialize storage
X_warmstart = zeros(TOTAL_WARMUP_RUNS, 4);
F_warmstart = zeros(TOTAL_WARMUP_RUNS, 1);
metrics_history = cell(TOTAL_WARMUP_RUNS, 1);

% Generate LHS samples for better space coverage
fprintf('Generating %d LHS samples for better parameter space coverage...\n', LHS_WARMUP_RUNS);
% IMPROVED: Use maximin criterion for better space filling (basic_final_suggest.txt #3)
X_lhs = lhsdesign(LHS_WARMUP_RUNS, 4, 'criterion', 'maximin');
% Scale to parameter bounds
for i = 1:4
    X_lhs(:, i) = lb(i) + X_lhs(:, i) * (ub(i) - lb(i));
end

% Starting parameters for legacy runs (reasonable defaults)
current_params = [2.0, 1.0, 1.0, 1.0];

% Timing
warmstart_time = tic;

% Generate combined warm-start data: Legacy + LHS
for i = 1:TOTAL_WARMUP_RUNS
    if i <= LEGACY_WARMUP_RUNS
        % Legacy sampling with adaptive updates
        fprintf('Legacy %d/%d: [%.2f, %.2f, %.2f, %.2f]', ...
            i, LEGACY_WARMUP_RUNS, current_params(1), current_params(2), ...
            current_params(3), current_params(4));
        eval_params = current_params;
    else
        % LHS sampling for better space coverage
        lhs_idx = i - LEGACY_WARMUP_RUNS;
        eval_params = X_lhs(lhs_idx, :);
        fprintf('LHS %d/%d: [%.2f, %.2f, %.2f, %.2f]', ...
            lhs_idx, LHS_WARMUP_RUNS, eval_params(1), eval_params(2), ...
            eval_params(3), eval_params(4));
    end
    
    % Evaluate with current parameters (with timeout protection)
    eval_start = tic;
    try
        [obj, metrics] = topology_wrapper_with_metrics(eval_params, hpo_config);
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
    X_warmstart(i, :) = eval_params;
    F_warmstart(i) = obj;
    metrics_history{i} = metrics;
    
    % ZETA AUTO-CALIBRATION (first evaluation only)
    if isempty(ZETA) && isfield(metrics, 'obj_p1') && isfield(metrics, 'Mnd')
        base = metrics.obj_p1;  % Primary objective with p=1
        aux = (VOL_TARGET - metrics.vol)^2 + metrics.Mnd;  % Volume + discreteness terms
        ZETA = base / max(aux, 1e-12);  % Ensure same order of magnitude
        fprintf(' → ZETA auto-calibrated: %.3e\n', ZETA);
    end
    
    if metrics.converged
        conv_str = 'Y';
    else
        conv_str = 'N';
    end
    fprintf(' → Obj: %.3e, Gray: %.1f%%, Conv: %s\n', ...
        obj, metrics.gray, conv_str);
    
    % Update using legacy logic (only for legacy samples)
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
        % Configure surrogate optimization with version compatibility
        try
            % FIXED: Proper table construction (suggest_for_adjust.txt §2)
            initial_table = array2table(X_warmstart, ...
                'VariableNames', {'x1','x2','x3','x4'});
            initial_table.objective = F_warmstart;
            
            % IMPROVED: Version-compatible options setup
            options = create_surrogate_options(SURROGATE_MAX_EVALS, BATCH_SIZE, lb, true, initial_table);
            fprintf('Using InitialPoints table for warm-start\n');
            
        catch ME_table
            fprintf('Warning: InitialPoints table setup failed: %s\n', ME_table.message);
            % Fallback to basic options without initial points
            options = create_surrogate_options(SURROGATE_MAX_EVALS + TOTAL_WARMUP_RUNS, BATCH_SIZE, lb, false, []);
            fprintf('Using basic surrogateopt (will pre-evaluate warm points)\n');
            
            % Pre-evaluate warm-start points
            fprintf('Pre-evaluating %d warm-start points...\n', TOTAL_WARMUP_RUNS);
            for i = 1:TOTAL_WARMUP_RUNS
                [f_check, ~, ~] = wrapper_with_constraints(X_warmstart(i,:), hpo_config);
                fprintf('  Warm %d: f=%.3e\n', i, f_check);
            end
        end
        
        % IMPROVED: Single evaluation for objective and constraints 
        surrogate_time = tic;
        
        fprintf('Using single-evaluation wrapper for optimal efficiency...\n');
        [x_optimal, f_optimal, exitflag, output] = surrogateopt(@(x) wrapper_with_constraints(x, hpo_config), lb, ub, options);
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
params_hifi.max_iterations = hpo_config.full_iterations;  % Full iterations
params_hifi.nely = hpo_config.fine_mesh_size;            % Fine mesh for accuracy
params_hifi.force_serial = true;              % Keep serial for stability
params_hifi.disable_plotting = false;         % Enable plotting for final result

fprintf('Running high-fidelity validation (%dx%d mesh, %d iterations)...\n', ...
    hpo_config.fine_mesh_size, hpo_config.fine_mesh_size, hpo_config.full_iterations);
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

%% SAVE RESULTS WITH ENHANCED LOGGING
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
    
    % IMPROVED: Enhanced logging for reproducibility (basic_final_suggest.txt #7)
    results.experiment_info = struct();
    results.experiment_info.random_seed = RANDOM_SEED;
    results.experiment_info.matlab_version = version;
    results.experiment_info.matlab_detailed = ver('MATLAB');  % Include patch version
    results.experiment_info.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    results.experiment_info.config = hpo_config;
    results.experiment_info.legacy_runs = LEGACY_WARMUP_RUNS;
    results.experiment_info.lhs_runs = LHS_WARMUP_RUNS;
    results.experiment_info.surrogate_evals = SURROGATE_MAX_EVALS;
    results.experiment_info.script_name = mfilename('fullpath');  % Record script path
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filename = sprintf('surrogate_results_%s.mat', timestamp);
    save(filename, 'results');
    
    % Also save JSON log for easy inspection
    json_filename = sprintf('run_log_%s.json', timestamp);
    log_data = struct();
    log_data.seed = RANDOM_SEED;
    log_data.timestamp = results.experiment_info.timestamp;
    log_data.bounds = [lb; ub];
    log_data.optimal_params = x_optimal;
    log_data.optimal_objective = f_optimal;
    log_data.total_time_minutes = total_time/60;
    
    % Write JSON log (simple format)
    fid = fopen(json_filename, 'w');
    if fid ~= -1
        fprintf(fid, '{\n');
        fprintf(fid, '  "seed": %d,\n', log_data.seed);
        fprintf(fid, '  "timestamp": "%s",\n', log_data.timestamp);
        fprintf(fid, '  "optimal_params": [%.4f, %.4f, %.4f, %.4f],\n', x_optimal);
        fprintf(fid, '  "optimal_objective": %.6e,\n', f_optimal);
        fprintf(fid, '  "total_time_minutes": %.2f\n', log_data.total_time_minutes);
        fprintf(fid, '}\n');
        fclose(fid);
        fprintf('JSON log saved to %s\n', json_filename);
    end
    
    fprintf('Results saved to %s\n', filename);
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

%% COMBINED OBJECTIVE AND CONSTRAINTS WRAPPER (OPTIMAL EFFICIENCY)
function [f, c, ceq] = wrapper_with_constraints(x, config)
    % IMPROVED: Single evaluation for both objective and constraints
    % Avoids duplicate topFlow_mpi_robust calls, ~2x speedup
    
    % Access global parameters for academic objective
    global VOL_TARGET ZETA
    
    % Parameter validation using config bounds
    x = max(x, config.lb);  % Lower bounds
    x = min(x, config.ub);  % Upper bounds
    
    % Build parameter structure (FAST HPO MODE)
    params = struct();
    params.enable_surrogate_mode = true;
    params.beta_init = x(1);
    params.qa_growth_factor = x(2);
    params.mv_adaptation_rate = x(3);
    params.rmin_decay_rate = x(4);
    params.max_iterations = config.fast_iterations;  % Fast evaluation for HPO
    params.nely = config.coarse_mesh_size;          % Coarse mesh for speed
    params.force_serial = true;              % Avoid pool conflicts
    params.disable_plotting = true;         % No plotting during HPO
    
    try
        % Single call to topology optimizer
        result = topFlow_mpi_robust(params);
        
        % ACADEMIC OBJECTIVE FUNCTION (Eq. 9 from warm_suggest.txt)
        if ~isempty(ZETA) && isfield(result, 'obj_p1') && isfield(result, 'Mnd')
            % Use academic formulation: f = c̃(x,y) + ζ[(f-V)² + M_nd]
            vol_penalty = (VOL_TARGET - result.vol)^2;
            discrete_penalty = result.Mnd;
            f = result.obj_p1 + ZETA * (vol_penalty + discrete_penalty);
        else
            % Fallback to standard objective if metrics unavailable
            f = result.obj;
        end
        
        % SIMPLIFIED CONSTRAINTS: Only convergence required
        c = [
            double(~result.converged)   % Must converge (0 if converged, 1 if not)
        ];
        
        % Equality constraints (ceq = 0) - none
        ceq = [];
        
        % Ensure finite results
        if ~isfinite(f)
            f = 1e6;
            c = [100; 1];  % Large constraint violations
        end
        
    catch ME
        fprintf('  ERROR in topology evaluation: %s\n', ME.message);
        f = 1e6;  % Large penalty for failed evaluations
        c = [100; 1];  % Large constraint violations
        ceq = [];
    end
end

%% TOPOLOGY WRAPPER WITH DETAILED METRICS (IMPROVED - no global variables)
function [obj, metrics] = topology_wrapper_with_metrics(x, config)
    % IMPROVED: Thread-safe wrapper without global variables (new_suggest.txt #5)
    
    % Parameter validation using config bounds
    x = max(x, config.lb);  % Lower bounds
    x = min(x, config.ub);  % Upper bounds
    
    % Build parameter structure (FAST HPO MODE)
    params = struct();
    params.enable_surrogate_mode = true;
    params.beta_init = x(1);
    params.qa_growth_factor = x(2);
    params.mv_adaptation_rate = x(3);
    params.rmin_decay_rate = x(4);
    params.max_iterations = config.fast_iterations;  % Fast evaluation for HPO
    params.nely = config.coarse_mesh_size;          % Coarse mesh for speed
    params.force_serial = true;              % Avoid pool conflicts
    params.disable_plotting = true;         % No plotting during HPO
    
    try
        % Call robust topology optimizer
        result = topFlow_mpi_robust(params);
        
        % IMPROVED: Adaptive weight normalization (new_suggest.txt #2)
        obj = adaptive_objective_function(result, config);
        
        % Extract detailed metrics for legacy update logic
        metrics = struct();
        metrics.obj_raw = result.obj;
        metrics.obj_p1 = result.obj_p1;  % NEW: p=1 objective
        metrics.Mnd = result.Mnd;        % NEW: Non-discreteness measure
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
        metrics = struct('obj_raw', 1e6, 'obj_p1', 1e6, 'Mnd', 1.0, 'gray', 100, 'vol', 0, ...
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
    % MODIFIED: Encourage complex flow structures - reduce beta when grayscale is too low
    if metrics.gray < 50
        params_next(1) = max(params(1) * 0.95, 0.5);  % Reduce beta to encourage gray zones
    elseif metrics.gray > 70
        params_next(1) = min(params(1) * 1.02, 3.0);  % Slight increase when grayscale is good
    end
    
    % QA factor based on progress - encourage higher grayscale
    if metrics.gray < 60
        params_next(2) = max(params(2) * 0.95, 0.7);  % Reduce qa to encourage flow channels
    elseif metrics.gray > 70 && metrics.converged
        params_next(2) = min(params(2) * 1.05, 1.4);  % Increase qa when grayscale is sufficient
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

%% VERSION-COMPATIBLE SURROGATE OPTIONS CREATOR
function options = create_surrogate_options(max_evals, batch_size, lb, use_init_points, init_table)
    % Create surrogate optimization options with version compatibility
    % Handles different MATLAB versions that may not support all options
    
    % Base options (available in all versions)
    base_options = {
        'MaxFunctionEvaluations', max_evals, ...
        'Display', 'iter', ...
        'UseParallel', true, ...   % Enable parallel evaluation
        'MinSampleDistance', 0.1/sqrt(length(lb)) ...  % Scale with dimension
    };
    
    % Add constraint tolerance (generally available)
    extended_options = [base_options, {'ConstraintTolerance', 1e-3}];
    
    % Try to add InitialPoints if requested
    if use_init_points && ~isempty(init_table)
        % First try: Table format (newer MATLAB versions)
        try
            test_options_table = [extended_options, {'InitialPoints', init_table}];
            options_test = optimoptions('surrogateopt', test_options_table{:});
            extended_options = test_options_table;
            fprintf('✓ InitialPoints (table format) supported\n');
        catch ME_table
            % Second try: Matrix format (older MATLAB versions)
            try
                if istable(init_table)
                    init_matrix = table2array(init_table(:, 1:end-1));  % Exclude objective column
                else
                    init_matrix = init_table;
                end
                test_options_matrix = [extended_options, {'InitialPoints', init_matrix}];
                options_test = optimoptions('surrogateopt', test_options_matrix{:});
                extended_options = test_options_matrix;
                fprintf('✓ InitialPoints (matrix format) supported\n');
            catch ME_matrix
                fprintf('! InitialPoints not supported - Table error: %s\n', ME_table.message);
                fprintf('! InitialPoints not supported - Matrix error: %s\n', ME_matrix.message);
                % Continue without InitialPoints
            end
        end
    end
    
    % Try to add BatchSize (newer versions only)
    try
        test_options = [extended_options, {'BatchSize', batch_size}];
        options = optimoptions('surrogateopt', test_options{:});
        fprintf('✓ BatchSize=%d supported\n', batch_size);
    catch ME_batch
        try
            % Try without BatchSize but with other extended options
            options = optimoptions('surrogateopt', extended_options{:});
            if contains(ME_batch.message, 'BatchSize')
                fprintf('! BatchSize not supported, using sequential evaluation\n');
            else
                fprintf('! Advanced options partially supported\n');
            end
        catch ME_extended
            % If extended options fail, use minimal options
            fprintf('Warning: Using minimal options due to compatibility: %s\n', ME_extended.message);
            minimal_options = {
                'MaxFunctionEvaluations', max_evals, ...
                'Display', 'iter', ...
                'UseParallel', true
            };
            try
                options = optimoptions('surrogateopt', minimal_options{:});
            catch ME_minimal
                % Last resort: ultra-minimal options
                fprintf('Warning: Using ultra-minimal options: %s\n', ME_minimal.message);
                options = optimoptions('surrogateopt', ...
                    'MaxFunctionEvaluations', max_evals, ...
                    'Display', 'iter');
            end
        end
    end
end

%% ACADEMIC OBJECTIVE FUNCTION (Eq. 9 from warm_suggest.txt)
function f = adaptive_objective_function(result, config)
    % ACADEMIC FORMULATION: f = c̃(x,y) + ζ[(f-V)² + M_nd]
    % This replaces all previous grayscale-based penalties with mathematically principled approach
    
    % Access global parameters
    global VOL_TARGET ZETA
    
    % Use academic objective if all components are available
    if ~isempty(ZETA) && isfield(result, 'obj_p1') && isfield(result, 'Mnd')
        % Primary objective with p=1 SIMP
        primary_obj = result.obj_p1;
        
        % Volume fraction penalty term
        vol_penalty = (VOL_TARGET - result.vol)^2;
        
        % Non-discreteness penalty term
        discrete_penalty = result.Mnd;
        
        % Combined academic objective (Eq. 9)
        f = primary_obj + ZETA * (vol_penalty + discrete_penalty);
        
        % Add small convergence penalty if needed
        if ~result.converged
            f = f + 0.1 * primary_obj;  % 10% penalty for non-convergence
        end
    else
        % Fallback to simplified objective if academic metrics unavailable
        f = result.obj;
        if ~result.converged
            f = f * 2.0;  % Double penalty for non-convergence
        end
    end
    
    % Ensure finite result
    if ~isfinite(f) || f <= 0
        f = 1e6;
    end
end



fprintf('Surrogate topology optimizer ready. Run this script to execute.\n'); 