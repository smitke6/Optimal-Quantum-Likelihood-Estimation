%% Background
% This script implements the optimized version of QLE developed in the
% article "Optimal Quantum Likelihood Estimation". This version finds the
% optimal parameters for each iteration by searching over a grid.
% The configurable pieces are denoted by comments starting with CONFIGURE.

clc;
clearvars;

%% Setup
% Pauli matrices
sigX = [0, 1; 1, 0];
sigY = [0, -1i; 1i, 0];
sigZ = [1, 0; 0, -1];
sigX2 = 2 * sigX;
sigZ2 = 2 * sigZ;

% Candidate Hamiltonians
H1 = sigX+sigZ;
H2 = 1.2 * [1, 0; 0, 0] - 0.8 * [0, 0; 0, 1];  
H3 = 2*sigY+[1, 0; 0, 2];
H4 = 4*(1/sqrt(2)) * [1, 1; 1, -1];
H5 = sigX+H4;
H6= [1, 2; 2, -1];
H7 = 0.5 * sigX + 0.8 * sigY;
H8 = [2, 0; 0, -1];
H9 = [1, 1i; -1i, 1];
H10 = [3, 2; 2, 0];
H11 = [0, 2 + 1i; 2 - 1i, 1];
H12 = 0.6 * sigZ + 0.4 * sigY;
A = randn(2) + 1i * randn(2);
H13 = (A + A') / 2; 

%CONFIGURE: The user may compose any list of candidate Hamiltonians out of
%the 2*2 matrices above, or use their own matrices. The paper studies the
%following two lists.
%H_list = {H1, H4, H3, H2, H5, H6};
H_list = {sigX, sigX2, sigZ, sigZ2};

N = length(H_list);

convergence_iters = NaN(1, N);
success_flags = false(1, N);

%% Algorithm
% CONFIGURE: The user may choose the convergence threshold
threshold = 0.99;

for idx = 1:N
    disp('**********************')
    disp('the right H is:'); disp(idx);
    H_true = H_list{idx};
    weights = ones(1, N) / N;
    num_iterations = 10;

    iter = 1;
    while iter <= num_iterations+1
        if max(weights) >= threshold
            convergence_iters(idx) = iter;
            break;
        end

        %t = PGH(H_list, weights);
        best_params = optimize_entropy(weights,H_list);
        t     = best_params(1);
        theta = best_params(2);
        phi   = best_params(3);
        alpha = best_params(4);
        beta  = best_params(5);
        disp("ideal t")
        disp(t);

        psi0   = [cos(alpha); exp(1i*beta)*sin(alpha)];
        psi0 = psi0 / norm(psi0);

        psi_all = repmat(psi0, 1, N);
        rho_tilda_F = diag(weights);
        rho_total = zeros(2*N);
        for j = 1:N
            proj = psi_all(:,j) * psi_all(:,j)';
            block = kron(proj, rho_tilda_F);
            rho_total = rho_total + block;
        end

        U_true = expm(-1i * H_true * t);
        psi_f  = U_true * psi0;
        psi_f = psi_f / norm(psi_f);
        u1=[cos(theta/2); exp(1i*phi)*sin(theta/2)];
        u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
        W=[u1,u2];
        psi_final=W*psi_f;
        psi_final = psi_final / norm(psi_final);
        p_u1   = abs(psi_final(1))^2;
        outcome = sample_from_distribution([p_u1, 1 - p_u1]);

        likelihoods = zeros(1, N);
        psi_all = zeros(2, N);
        rho_Y=zeros(2,2);
        H_YF=0;
        for j = 1:N
            U_j = expm(-1i * H_list{j} * t);
            psi_j = U_j * psi0;
            psi_j=W*psi_j;
            psi_j = psi_j / norm(psi_j);
            psi_all(:,j) = psi_j;
            pj = abs(psi_j(1))^2;
            pj = min(max(real(pj), 1e-12), 1 - 1e-12);
            likelihoods(j) = (outcome==0)*pj + (outcome==1)*(1-pj);
            rho_Y=rho_Y+weights(j)*psi_j*psi_j';
            H_YF=H_YF+weights(j)*(-pj*log2(pj)-(1-pj)*log2(1-pj));
        end

        ent_y=Entropy(diag(diag(rho_Y)));
        disp(['Iteration ', num2str(iter), ':']);
        disp('H(Y)='); disp(ent_y);
        disp('H(Y|F)='); disp(H_YF);
        disp('I(FY)='); disp(ent_y-H_YF);

        weights = weights .* likelihoods;
        weights(weights < 0) = 0;
        weights = weights / sum(weights);

        if max(weights) >= 0.95
            [~, guess_H] = max(weights);
            success_flags(idx) = (guess_H == idx);
        end

        disp('weights ='); disp(weights);
        iter = iter + 1;
    end
end

disp('Convergence iterations:');
disp(convergence_iters-1);
disp('Did each H converge to correct hypothesis?');
disp(success_flags);
disp(['All correct: ', mat2str(all(success_flags))]);

%% Helper functions
function t = PGH(H_list, weights)
    [~, I] = sort(weights, 'descend');
    i1 = I(1); i2 = I(2);
    deltaH = H_list{i1} - H_list{i2};
    normH = norm(deltaH, 'fro');
    t = pi / (2 * normH);
end

function outcome = sample_from_distribution(p)
    if rand() < p(1), outcome = 0; else, outcome = 1; end
end

%Entropy - Von Neumann entropy of a quantum state
% S = Entropy(rho) returns the von Neumann entropy of the density matrix
% rho.
function S = Entropy(rho)
    % Force Hermitian: symmetric average
    rho = (rho + rho') / 2;

    % Check if it's still significantly non-Hermitian
    if norm(rho - rho', 'fro') > 1e-10
        warning('Entropy: input not Hermitian even after symmetrization.');
    end

    [~, D] = eig(rho);
    eigvals = real(diag(D));
    eigvals = eigvals(eigvals > 1e-12); % filter near-zero

    S = -sum(eigvals .* log2(eigvals));
end

%optimize_entropy - Find the best t and parameters minimizing H(Y|F)
%   INPUT:  weights - 1x4 array of weights for the four hypotheses
%   OUTPUT: best_params - [t, theta, phi, alpha, beta]
function best_params = optimize_entropy(weights, H_list)
    N=length(H_list);

    t_vals = linspace(0.1, 2*pi, 10);
    sample = 24;
    theta_vals = linspace(0, pi/2, sample/4);
    phi_vals   = linspace(0, 2*pi, sample);
    alpha_vals = linspace(0, pi, sample/2);
    beta_vals  = linspace(0, 2*pi, sample);

    min_entropy = Inf;
    best_params = [];
    IN = eye(N);

    for t = t_vals
        for alpha = alpha_vals
            for beta = beta_vals
                psi0 = [cos(alpha); exp(1i*beta)*sin(alpha)];

                % Evolve under all Hamiltonians
                rho_FY = zeros(2*N, 2*N);
                for j = 1:N
                    H = H_list{j};
                    psi_j = expm(-1i * H * t) * psi0;
                    rho_j = psi_j * psi_j';
                    rho_FY(2*j-1:2*j, 2*j-1:2*j) = weights(j) * rho_j;
                end

                for theta = theta_vals
                    for phi = phi_vals
                        u1 = [cos(theta/2); exp(1i*phi)*sin(theta/2)];
                        u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
                        W = [u1, u2];

                        A_plus  = W' * [1,0;0,0] * W;
                        A_minus = eye(2) - A_plus;

                        rho_plus = kron(IN, A_plus) * rho_FY * ...
                            kron(IN, A_plus);
                        rho_minus = kron(IN, A_minus) * rho_FY * ...
                            kron(IN, A_minus);

                        prob_plus = trace(rho_plus);
                        prob_minus = trace(rho_minus);

                        if prob_plus > 0
                            rho_plus = rho_plus / prob_plus;
                        else
                            continue;
                        end

                        if prob_minus > 0
                            rho_minus = rho_minus / prob_minus;
                        else
                            continue;
                        end

                        ent = prob_plus * Entropy(rho_plus) + ...
                            prob_minus * Entropy(rho_minus);

                        if ent < min_entropy
                            min_entropy = ent;
                            best_params = [t, theta, phi, alpha, beta];
                        end
                    end
                end
            end
        end
    end
end

% Min_Energy_Gap - Find the minimal energy gap in a given list of Hamiltonians
%   INPUT:  list - a list of 2*2 Hermitian matrices
%   OUTPUT: delta - the minimal energy gap
function delta = Min_Energy_Gap(list)
    N = length(list);
    gap = zeros(N,1);
    for i = 1:N
        D = eig(list{i});
        gap(i) = abs(D(2)-D(1));
    end
    delta = min(gap(gap>0));
end
