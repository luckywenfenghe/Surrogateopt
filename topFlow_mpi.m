%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    NAVIER-STOKES TOPOLOGY OPTIMISATION CODE WITH MPI     %
%    PARALLEL COMPUTING FOR THERMAL PROBLEMS, 2024         %
% COPYRIGHT (c) 2022, J ALEXANDERSEN. BSD 3-CLAUSE LICENSE %
% MPI PARALLELIZATION: Enhanced for thermal problems       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;

%% PARALLEL COMPUTING INITIALIZATION
% Configure number of workers (you can modify this value)
desired_workers = 8; % Set to desired number of workers (or 'auto' for maximum)

try
    % Check if parallel pool is running
    p = gcp('nocreate');
    if ~isempty(p)
        % Close existing pool if worker count doesn't match
        if p.NumWorkers ~= desired_workers
            delete(p);
            p = [];
        end
    end
    
    if isempty(p)
        % Get maximum available workers
        max_workers = feature('numcores');
        fprintf('System has %d CPU cores available\n', max_workers);
        
        % Determine actual number of workers to use
        if ischar(desired_workers) && strcmp(desired_workers, 'auto')
            num_workers = max_workers;
        else
            num_workers = min(desired_workers, max_workers);
        end
        
        % Start parallel pool with specified number of workers
        fprintf('Starting parallel pool with %d workers...\n', num_workers);
        parpool('local', num_workers);
        p = gcp;
    end
    
    fprintf('Parallel Computing Initialized with %d workers\n', p.NumWorkers);
    fprintf('CPU cores: %d, Workers used: %d\n', feature('numcores'), p.NumWorkers);
    
    % Get worker information
    nprocs = p.NumWorkers + 1; % Include main process
    fprintf('Total processes (including main): %d\n', nprocs);
catch ME
    fprintf('Parallel Computing Toolbox initialization failed: %s\n', ME.message);
    fprintf('Continuing with serial computation...\n');
    nprocs = 1; % Single process for serial computation
end

%% DEFINITION OF INPUT PARAMETERS
% PROBLEM TO SOLVE (Fixed to 3 for thermal problem)
probtype = 3;
% DOMAIN SIZE
Lx = 1.0; Ly = 1.0;
% DOMAIN DISCRETISATION
nely = 80; nelx = nely*Lx/Ly;
% ALLOWABLE FLUID VOLUME FRACTION
volfrac = 1/4; xinit = volfrac;

% PROGRESSIVE DENSITY FILTERING PARAMETERS
rmin_init = 1.5; % Initial filtering radius (2*dx as suggested)
rmin_final = 0.6; % Final filtering radius for fine details
r_decay = 0.98; % Decay factor (0.98 as suggested in filter_adjust.txt)
filter_update_freq = 2; % Update filtering radius every 2 iterations
rmin = rmin_init; % Current filtering radius

% PHYSICAL PARAMETERS
Uin = 1e1; rho = 1e1; mu = 1e3;
% THERMAL PARAMETERS
kappa = 0.8; % Thermal conductivity (W/m·K)
Cp = 4180; % Specific heat capacity (J/kg·K)
dt_thermal = 0.01; % Time step for transient thermal analysis
% BRINKMAN PENALISATION
alphamax = 2.5*mu/(0.01^2); alphamin = 2.5*mu/(100^2);
% CONTINUATION STRATEGY - PROGRESSIVE QA GROWTH
ainit = 2.5*mu/(0.1^2);
qinit = 0.02;
qa_max = qinit / 0.01; % Maximum qa value (equivalent to smallest divisor)
qa_growth_rate = 1.05; % Progressive growth rate for qa
qa_warmup_iterations = 20; % Iterations before qa starts growing
% OPTIMISATION PARAMETERS ADAPTIVE - PROGRESSIVE STEP SIZE
chlim = 1e-3; chnum = 3;
mv_max = 0.05; mv_min = 0.001;  % Expanded range for better control
mv_init = 0.01; % Start with small step size
tau_up = 1.15; tau_dn = 0.85;   % More conservative adjustment rates
eps_hi = 1.5*chlim; eps_lo = 0.3*chlim; % Tighter thresholds
mvlim = mv_init;   % START WITH SMALL STEP SIZE FOR STABILITY

% PROGRESSIVE STEP SIZE PARAMETERS
warmup_iterations = 50; % Number of iterations for gradual increase
step_growth_rate = 1.015; % Factor for gradual step size increase (more gradual)

% HEAVISIDE β-PROJECTION PARAMETERS - PROGRESSIVE GROWTH
beta_init = 1.0; beta_max = 8; % Reduced max for more gradual growth
beta_growth_rate = 1.01; % Progressive growth rate for beta
beta_warmup_iterations = 50; % Iterations before beta starts growing
beta = beta_init;
eta = 0.5; % Threshold parameter for projection

% OPTIMISATION PARAMETERS
maxiter = 150;  plotdes = 1; % Total iterations for progressive strategies

% NEWTON SOLVER PARAMETERS
nltol = 1e-5; nlmax = 25; plotres = 0;
% EXPORT FILE
filename='output_mpi'; exportdxf = 0;

%% PREPARE FINITE ELEMENT ANALYSIS
dx = Lx/nelx; dy = Ly/nely;
nodx = nelx+1; nody = nely+1; nodtot = nodx*nody;
neltot = nelx*nely; 
doftot = 4*nodtot; % u(2) + p(1) + T(1) = 4 DOFs per node

% NODAL CONNECTIVITY
nodenrs = reshape(1:nodtot,nody,nodx);
edofVecU = reshape(2*nodenrs(1:end-1,1:end-1)+1,neltot,1);
edofMatU = repmat(edofVecU,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],neltot,1);
edofVecP = reshape(nodenrs(1:end-1,1:end-1),neltot,1);
edofMatP = repmat(edofVecP,1,4)+repmat([1 nely+[2 1] 0],neltot,1);
edofVecT = reshape(nodenrs(1:end-1,1:end-1),neltot,1);
edofMatT = repmat(edofVecT,1,4)+repmat([1 nely+[2 1] 0],neltot,1);
edofMat = [edofMatU 2*nodtot+edofMatP 3*nodtot+edofMatT];

% Sparse matrix setup
iJ = reshape(kron(edofMat,ones(16,1))',256*neltot,1);
jJ = reshape(kron(edofMat,ones(1,16))',256*neltot,1);
iR = reshape(edofMat',16*neltot,1); jR = ones(16*neltot,1); 
jE = repmat(1:neltot,16,1);

%% PREPARE EXPLICIT DENSITY FILTERING 
% Pre-calculate filtering weights for efficiency (Sigmund 2007)
dx = Lx/nelx; dy = Ly/nely;
rmin = rmin_init * max(dx, dy); % Scale by element size
[H, Hs] = filter_setup(nelx, nely, rmin);
fprintf('      Initial filter radius: %4.2f (elements), %4.3f (physical)\n', rmin_init, rmin);

%% DEFINE BOUNDARY CONDITIONS
% DEFINE THE PROBLEMS IN SEPARATE MATLAB FILE
run('problems.m');
% NULLSPACE MATRICES FOR IMPOSING BOUNDARY CONDITIONS
EN=speye(doftot); ND=EN; ND(fixedDofs,fixedDofs)=0.0; EN=EN-ND;
% VECTORS FOR FREE DOFS
alldofs = 1:doftot; freedofs = setdiff(alldofs,fixedDofs);

%% INITIALISATION
% SOLUTION VECTOR
S = zeros(doftot,1); dS = S; L = S; 
S(fixedDofs) = DIR(fixedDofs);
% PREVIOUS TEMPERATURE FIELD (for transient analysis)
Sold = S; % Initialize previous solution
% DESIGN FIELD
xPhys = xinit*ones(nely,nelx); 
% COUNTERS
loop = 0; nlittot = 0; chcnt = 0;
% CHANGE
change = Inf; objOld = Inf;
% CONTINUATION
qa = qinit; % Initialize qa with initial value
% VECTORISED CONSTANTS
dxv = dx*ones(1,neltot); dyv = dy*ones(1,neltot);
muv = mu*ones(1,neltot); rhov = rho*ones(1,neltot);
dtv = dt_thermal*ones(1,neltot); % Time step vector
kv = kappa*ones(1,neltot); Cpv = Cp*ones(1,neltot);
Qv = Qsource*ones(1,neltot);

%% OUTPUT PROBLEM INFORMATION
fprintf('=========================================================\n');
fprintf('      Problem number: %2i - Reynolds number: %3.2e\n',probtype,Renum);
fprintf('      Heat transfer problem with uniform heat source\n');
fprintf('      Parallel Computing Enabled\n');
fprintf('      Number of workers: %d\n', p.NumWorkers);
fprintf('      Progressive strategies enabled:\n');
fprintf('      • Beta: %3.1f → %3.1f (rate: %4.2f, warmup: %d its)\n', beta_init, beta_max, beta_growth_rate, beta_warmup_iterations);
fprintf('      • Move-limit: %4.3f → [%4.3f, %4.3f] (warmup: %d its)\n', mv_init, mv_min, mv_max, warmup_iterations);
fprintf('      • QA continuation: %4.3f → %4.1f (rate: %4.2f, warmup: %d its)\n', qinit, qa_max, qa_growth_rate, qa_warmup_iterations);
fprintf('      • Density filtering: %3.1f → %3.1f (decay: %4.2f, freq: %d its)\n', rmin_init, rmin_final, r_decay, filter_update_freq);
fprintf('=========================================================\n');
fprintf('      Design it.:   0\n');

%% START ITERATION
destime = tic; ittime = tic;
while (loop <= maxiter)
    if (plotdes); figure(1); imagesc(xPhys); colorbar; caxis([0 1]); axis equal; axis off; drawnow; end
    
    %% GREYSCALE INDICATOR
    Md = 100*full(4*sum(xPhys(:).*(1-xPhys(:)))/neltot); 
    
    %% MATERIAL INTERPOLATION
    alpha = alphamin + (alphamax-alphamin)*(1-xPhys(:))./(1+qa*xPhys(:));
    dalpha = (qa*(alphamax - alphamin)*(xPhys(:) - 1))./(xPhys(:)*qa + 1).^2 - (alphamax - alphamin)./(xPhys(:)*qa + 1);
    
    %% NON-LINEAR NEWTON SOLVER WITH PARALLEL COMPUTING
    normR = 1; nlit = 0; fail = -1; nltime = tic;
    while (fail ~= 1)
        nlit = nlit+1; nlittot = nlittot+1;
        
        %% BUILD RESIDUAL AND JACOBIAN WITH PARALLEL COMPUTATION
        % Extract state variables for thermal problem (16 DOFs per element)
        uVars = S(edofMat(:,1:8)); % Velocity components u1-u8
        pVars = S(edofMat(:,9:12)); % Pressure components p1-p4  
        TVars = S(edofMat(:,13:16)); % Temperature components T1-T4
        
        % Parallel residual computation using parfor
        sR = zeros(16, neltot);
        parfor i = 1:neltot
            sR(:,i) = RES(dxv(i),dyv(i),muv(i),rhov(i),kv(i),Cpv(i),Qv(i),alpha(i),...
                uVars(i,1),uVars(i,2),uVars(i,3),uVars(i,4),uVars(i,5),uVars(i,6),uVars(i,7),uVars(i,8),...
                pVars(i,1),pVars(i,2),pVars(i,3),pVars(i,4),...
                TVars(i,1),TVars(i,2),TVars(i,3),TVars(i,4));
        end
        
        R = sparse(iR,jR,sR(:)); R(fixedDofs) = 0;
        if (nlit == 1); r0 = norm(R); end
        r1 = norm(R); normR = r1/r0;
        if (plotres); figure(6); semilogy(nlittot,normR,'x'); axis square; grid on; hold on; end
        if (normR < nltol); break; end
        
        % Parallel Jacobian computation using parfor
        sJ = zeros(256, neltot);
        parfor i = 1:neltot
            sJ(:,i) = JAC(dxv(i),dyv(i),muv(i),rhov(i),kv(i),Cpv(i),Qv(i),alpha(i),...
                uVars(i,1),uVars(i,2),uVars(i,3),uVars(i,4),uVars(i,5),uVars(i,6),uVars(i,7),uVars(i,8),...
                pVars(i,1),pVars(i,2),pVars(i,3),pVars(i,4),...
                TVars(i,1),TVars(i,2),TVars(i,3),TVars(i,4));
        end
        
        J = sparse(iJ,jJ,sJ(:)); J = (ND'*J*ND+EN);  
        
        % CALCULATE NEWTON STEP
        dS = -J\R;
        
        % L2-NORM LINE SEARCH (optimized with parallel computation)
        Sp = S + 0.5*dS;
        uVars_p = Sp(edofMat(:,1:8));
        pVars_p = Sp(edofMat(:,9:12));
        TVars_p = Sp(edofMat(:,13:16));

        sR_p = zeros(16, neltot);
        parfor i = 1:neltot
            sR_p(:,i) = RES(dxv(i),dyv(i),muv(i),rhov(i),kv(i),Cpv(i),Qv(i),alpha(i),...
                uVars_p(i,1),uVars_p(i,2),uVars_p(i,3),uVars_p(i,4),uVars_p(i,5),uVars_p(i,6),uVars_p(i,7),uVars_p(i,8),...
                pVars_p(i,1),pVars_p(i,2),pVars_p(i,3),pVars_p(i,4),...
                TVars_p(i,1),TVars_p(i,2),TVars_p(i,3),TVars_p(i,4));
        end
        
        R = sparse(iR,jR,sR_p(:)); R(fixedDofs) = 0; r2 = norm(R);
        
        Sp = S + 1.0*dS;
        uVars_p = Sp(edofMat(:,1:8));
        pVars_p = Sp(edofMat(:,9:12));
        TVars_p = Sp(edofMat(:,13:16));

        sR_p = zeros(16, neltot);
        parfor i = 1:neltot
            sR_p(:,i) = RES(dxv(i),dyv(i),muv(i),rhov(i),kv(i),Cpv(i),Qv(i),alpha(i),...
                uVars_p(i,1),uVars_p(i,2),uVars_p(i,3),uVars_p(i,4),uVars_p(i,5),uVars_p(i,6),uVars_p(i,7),uVars_p(i,8),...
                pVars_p(i,1),pVars_p(i,2),pVars_p(i,3),pVars_p(i,4),...
                TVars_p(i,1),TVars_p(i,2),TVars_p(i,3),TVars_p(i,4));
        end
        
        R = sparse(iR,jR,sR_p(:)); R(fixedDofs) = 0; r3 = norm(R);
        
        % SOLUTION UPDATE WITH "OPTIMAL" DAMPING
        lambda = max(0.01,min(1.0,(3*r1 + r3 - 4*r2)/(4*r1 + 4*r3 - 8*r2)));
        S = S + lambda*dS;
        
        % IF FAIL, RETRY FROM ZERO SOLUTION
        if (nlit == nlmax && fail < 0); nlit = 0; S(freedofs) = 0.0; normR=1; fail = fail+1; end
        if (nlit == nlmax && fail < 1); fail = fail+1; end
    end
    nltime=toc(nltime);
    
    fprintf('      Newton it.: %2i - Res. norm: %3.2e - Sol. time: %6.3f sec\n',nlit,normR,nltime);
    if (fail == 1); error('ERROR: Solver did not converge after retry from zero!\n      Stopping optimisation.\n'); end
    
    %% OBJECTIVE EVALUATION WITH PARALLEL COMPUTING - AVERAGE TEMPERATURE
    % Parallel objective computation for average temperature
    phiVals = zeros(1, neltot);
    parfor i = 1:neltot
        phiVals(i) = PHI(dxv(i),dyv(i),...
            TVars(i,1),TVars(i,2),TVars(i,3),TVars(i,4));
    end
    obj = sum(phiVals);
    
    change = abs(objOld-obj)/objOld; objOld = obj;
    
    %% PROGRESSIVE ADAPTIVE MOVE-LIMIT UPDATE
    % Progressive step size strategy: start small, gradually increase
    mvlim_old = mvlim;
    
    % Phase 1: Warmup phase - gradual increase regardless of performance
    if (loop <= warmup_iterations)
        % Gradual increase during warmup
        target_mvlim = mv_init * (step_growth_rate ^ loop);
        mvlim = min(target_mvlim, mv_max * 0.5); % Cap at half of max during warmup
        fprintf('      Warmup phase (%d/%d): mvlim = %5.3f\n', loop, warmup_iterations, mvlim);
    else
        % Phase 2: Performance-based adaptive adjustment
        if (change > eps_hi)
            % Good progress - can increase step size more aggressively
            mvlim = min(mvlim * tau_up, mv_max);
        elseif (change < eps_lo && loop > warmup_iterations + 5)
            % Poor progress after warmup - reduce step size conservatively
            mvlim = max(mvlim * tau_dn, mv_min);
        else
            % Moderate progress - small increase or maintain
            mvlim = min(mvlim * 1.02, mv_max); % Very small increase
        end
    end
    
    % No more step-wise qa continuation - using progressive qa instead
    
    % Special handling for high beta values - reduce step size for stability
    if (beta > 8.0 && change < eps_lo && mvlim > mv_min * 2.0)
        mvlim = max(mvlim * 0.8, mv_min * 1.5); % More conservative with high beta
    end
    
    %% VOLUME CONSTRAINT
    V = mean(xPhys(:));
    
    %% PRINT RESULTS
    ittime = toc(ittime);
    fprintf('      Obj.: %3.2e - Constr.: %3.2e - Md: %3.2f\n',obj,V,Md);
    fprintf('      Change: %4.3e - It. time: %6.3f sec\n',change,ittime);
    fprintf('      QA: %4.3e - mvlim: %5.3f', qa, mvlim);
    
    % Show step size progression info
    if (mvlim ~= mvlim_old)
        if (mvlim > mvlim_old)
            fprintf(' ↗️');
        else
            fprintf(' ↘️');
        end
    end
    fprintf('\n');
    
    fprintf('      Beta: %3.1f - Filter radius: %4.3f\n', beta, rmin);
    ittime = tic;
    
    %% EVALUATE CURRENT ITERATE - CONTINUE UNLESS CONSIDERED CONVERGED
    if (change < chlim); chcnt = chcnt + 1; else; chcnt = 0; end
    if (chcnt == chnum && beta >= beta_max); break; end % Stop when converged and beta is at maximum
    
    %% PROGRESSIVE QA UPDATE
    % Progressive qa growth: start after warmup iterations
    qa_old = qa;
    if (loop > qa_warmup_iterations)
        % Gradual qa increase based on iteration
        growth_factor = qa_growth_rate ^ (loop - qa_warmup_iterations);
        qa = min(qinit * growth_factor, qa_max);
        
        if (abs(qa - qa_old) / qa_old > 0.05) % Only show significant changes (>5%)
            fprintf('      QA updated: %4.3e → %4.3e\n', qa_old, qa);
        end
    end
    
    %% PROGRESSIVE BETA UPDATE
    % Progressive beta growth: start after warmup iterations
    beta_old = beta;
    if (loop > beta_warmup_iterations)
        % Gradual beta increase based on iteration
        growth_factor = beta_growth_rate ^ (loop - beta_warmup_iterations);
        beta = min(beta_init * growth_factor, beta_max);
        
        if (beta ~= beta_old)
            fprintf('      Beta updated: %3.1f → %3.1f\n', beta_old, beta);
        end
    end
    
    %% PRINT HEADER FOR ITERATION
    loop = loop + 1;
    fprintf('---------------------------------------------------------\n');
    fprintf('      Design it.:%4i\n',loop);
    
    %% ADJOINT SOLVER WITH PARALLEL COMPUTING - TEMPERATURE DERIVATIVES
    % Parallel adjoint derivative computation for temperature objective
    dphiVals = zeros(16, neltot);
    parfor i = 1:neltot
        dphiVals(:,i) = dPHIds(dxv(i),dyv(i),...
            TVars(i,1),TVars(i,2),TVars(i,3),TVars(i,4));
    end
    
    sR = [dphiVals; zeros(0,neltot)]; % No padding needed
    RHS = sparse(iR,jR,sR(:)); RHS(fixedDofs) = 0;
    L = J'\RHS;
    
    %% COMPUTE SENSITIVITIES WITH PARALLEL COMPUTING
    % Parallel sensitivity computation
    drdgVals = zeros(16, neltot);
    dphidgVals = zeros(1, neltot);
    parfor i = 1:neltot
        drdgVals(:,i) = dRESdg(dxv(i),dyv(i),muv(i),rhov(i),kv(i),Cpv(i),Qv(i),alpha(i),dalpha(i),...
            uVars(i,1),uVars(i,2),uVars(i,3),uVars(i,4),uVars(i,5),uVars(i,6),uVars(i,7),uVars(i,8),...
            pVars(i,1),pVars(i,2),pVars(i,3),pVars(i,4),...
            TVars(i,1),TVars(i,2),TVars(i,3),TVars(i,4));
        dphidgVals(i) = dPHIdg(dxv(i),dyv(i),alpha(i),dalpha(i),...
            TVars(i,1),TVars(i,2),TVars(i,3),TVars(i,4));
    end
    
    dRdg = sparse(iR(:),jE(:),drdgVals(:));
    dphidg = dphidgVals;
    
    % Add temperature equation contribution to sensitivity
    try
        sR_temp = zeros(4, neltot);
        parfor i = 1:neltot
            sR_temp(:,i) = dRESTemperature(dxv(i),dyv(i),muv(i),rhov(i),kv(i),Cpv(i),Qv(i),alpha(i),...
                uVars(i,1),uVars(i,2),uVars(i,3),uVars(i,4),uVars(i,5),uVars(i,6),uVars(i,7),uVars(i,8),...
                pVars(i,1),pVars(i,2),pVars(i,3),pVars(i,4),...
                TVars(i,1),TVars(i,2),TVars(i,3),TVars(i,4));
        end
        % Extract temperature adjoint variables (last 4 entries per element)
        L_temp_elem = L(edofMat(:,13:16)); % Extract 4 temperature adjoint values per element (neltot×4)
        L_temp = L_temp_elem.'; % Transpose to 4×neltot to match sR_temp dimensions
        temp_contrib = sum(L_temp .* sR_temp, 1); % Calculate Σ λ_T · ∂R_T/∂ρ per element (1 × neltot)
        % Add temperature contribution to design sensitivity
        sens = reshape(dphidg - L'*dRdg,nely,nelx) + reshape(temp_contrib, nely, nelx);
    catch ME
        % If dRESTemperature doesn't exist yet, use standard sensitivity
        sens = reshape(dphidg - L'*dRdg,nely,nelx);
        fprintf('      Warning: dRESTemperature.m call failed (%s), using standard sensitivity\n', ME.message);
    end
    
    % VOLUME CONSTRAINT
    dV = ones(nely,nelx)/neltot;
    
    %% OPTIONAL: SENSITIVITY FILTERING (more robust for fluid/thermal coupling)
    % Uncomment next line to use sensitivity filtering instead of density filtering
    % sens = sensitivity_filter(sens, H, Hs);
    
    %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
    xnew = xPhys; xlow = xPhys(:)-mvlim; xupp = xPhys(:)+mvlim;
    ocfac = xPhys(:).*max(1e-10,(-sens(:)./dV(:))).^(1/3);
    l1 = 0; l2 = ( 1/(neltot*volfrac)*sum( ocfac ) )^3;
    while (l2-l1)/(l1+l2) > 1e-3
        lmid = 0.5*(l2+l1);
        xnew(:) = max(0,max(xlow,min(1,min(xupp,ocfac/(lmid^(1/3))))));
        if mean(xnew(:)) > volfrac; l1 = lmid; else; l2 = lmid; end
    end
    %% EXPLICIT DENSITY FILTERING (Sigmund 2007)
    % Step 1: OC update gives design variables
    rho_tmp = xnew(:);                          % 优化得出的密度
    
    % Step 2: Apply spatial density filtering 
    rho_flt = (H * rho_tmp) ./ Hs;             % 空间滤波: ρ_filtered = (H*ρ)./Hs
    
    % Step 3: Apply β-projection to filtered density
    xPhys = reshape(rho_flt, nely, nelx);       % Reshape back to 2D
    xPhys = (tanh(beta*eta) + tanh(beta*(xPhys-eta)))./(tanh(beta*eta) + tanh(beta*(1-eta))); % β-投影
    
    %% PROGRESSIVE STRATEGIES COMPLETE
    % All parameters (qa, beta, mvlim) now use progressive growth
    
    %% PROGRESSIVE FILTERING RADIUS UPDATE 
    rmin_old = rmin;
    if (mod(loop, filter_update_freq) == 0 && rmin > rmin_final * max(dx, dy))
        % Update filtering radius progressively (分阶段 rmin 递减)
        rmin = max(rmin * r_decay, rmin_final * max(dx, dy));
        if (abs(rmin - rmin_old) > 1e-6)
            % Recalculate filter weights when radius changes significantly
            [H, Hs] = filter_setup(nelx, nely, rmin);
            fprintf('      Filter radius updated: %4.3f → %4.3f (physical units)\n', rmin_old, rmin);
        end
    end
    
end

%% PRINT FINAL INFORMATION
destime = toc(destime);
fprintf('=========================================================\n');
fprintf('      Number of design iterations: %4i\n',loop);
fprintf('      Final objective: %4.3e\n',obj);
fprintf('      Final beta value: %3.1f (max: %3.1f)\n', beta, beta_max);
fprintf('      Total time taken: %6.2f min\n',destime/60);
fprintf('      Parallel Performance:\n');
fprintf('      Workers used: %d\n', p.NumWorkers);
fprintf('      Total processes: %d\n', nprocs);
fprintf('=========================================================\n');

%% PLOT RESULTS
run('postproc.m');

if exist('exportdxf','var') && exportdxf
    run('export.m');
end
%% EXPLICIT DENSITY FILTERING FUNCTIONS (Sigmund 2007)
function [H, Hs] = filter_setup(nelx, nely, rmin)
% Standard density filter setup with linear weight decay
    iH = ones(nelx*nely*(2*(ceil(rmin)-1)+1)^2,1);
    jH = ones(size(iH));
    sH = zeros(size(iH));
    k = 0;
    for i1 = 1:nelx
        for j1 = 1:nely
            e1 = (i1-1)*nely+j1;
            for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
                for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                    e2 = (i2-1)*nely+j2;
                    k = k+1;
                    iH(k) = e1;
                    jH(k) = e2;
                    sH(k) = max(0, rmin - sqrt((i1-i2)^2+(j1-j2)^2));
                end
            end
        end
    end
    % Remove zero entries and create sparse matrix
    H = sparse(iH(1:k), jH(1:k), sH(1:k));
    Hs = sum(H, 2); % Normalization weights: Σ H_ij for each row i
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code was initial written by: Joe Alexandersen                      %
%                           Department of Mechanical and                  %
%                                         Electrical Engineering          %
%                           University of Southern Denmark                %
%                           DK-5230 Odense M, Denmark.                    % 
% Parallel Computing enhanced by: https://github.com/luckywenfenghe       %
% Features:                                                               %
% - Parallel Computing using MATLAB Parallel Computing Toolbox           %
% - Element-Level Parallelization with parfor loops                       %
% - Adaptive Move-Limit via Trust-Region MMA                              %
% - Automatic β-Projection Continuation                                   %
% - Optimized for Thermal Problems (Problem Type 3)                       %
%                                                                         %
% Please send your comments and questions to: joal@sdu.dk                 %
%                                                                         %
% The code is intended for educational purposes and theoretical details   %
% are discussed in the paper: "A detailed introduction to density-based   %
% topology optimisation of fluid flow problems including implementation   %
% in MATLAB", J. Alexandersen, SMO 2022, doi:                             %                          
%                                                                         %
% Disclaimer:                                                             %
% The author does not guarantee that the code is free from errors.        %
% Furthermore, the author shall not be liable in any event caused by the  %
% use of the program.                                                     %      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

