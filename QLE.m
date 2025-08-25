clc;
clearvars;

sigX = [0, 1; 1, 0];
sigY = [0, -1i; 1i, 0];
sigZ = [1, 0; 0, -1];
sigX2 = 2 * sigX;
sigZ2 = 2 * sigZ; 

H1 = sigX+sigZ;
H2 = 1.2 * [1, 0; 0, 0] - 0.8 * [0, 0; 0, 1];  
H3 = 2*sigY+[1, 0; 0, 2];
H4 = 4*(1/sqrt(2)) * [1, 1; 1, -1];
H5 = sigX+H4;
H6= [1, 2; 2, -1];

% Choose one of the following two sets of Hamiltonians
%H_list = {H1, H2, H3, H4, H5, H6};
H_list = {sigX, sigX2, sigZ, sigZ2};

N = length(H_list);
theta = 0.7854;
phi=6.2832;
 u1=[cos(theta/2); exp(1i*phi)*sin(theta/2)];%W gate
 u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
 W=[u1,u2];
% Initial state parameters
alpha = 1.2566;
beta   = 0.5712;
psi0   = [cos(alpha); exp(1i*beta)*sin(alpha)];
iter_arr=zeros(N,1);
% Loop through Hamiltonians
for idx = 1:N
    H_true = H_list{idx};
    weights = ones(1, N) / N;
    num_iterations = 10000;

    iter = 1;
    
    while iter <= num_iterations+1 && max(weights) < 0.99
        % Determine evolution time
        t = PGH(H_list, weights);

        % Step 1: Prepare ensemble states
        psi_all = repmat(psi0, 1, N);

        % Build classical register density rho_tilda_F
        rho_tilda_F = diag(weights);

        % Build joint density rho_total = \sum_j |psi_j><psi_j| \otimes rho_tilda_F
        % Here psi_all(:,j) is psi_j
        rho_total = zeros(2*N);
        for j = 1:N
            proj = psi_all(:,j) * psi_all(:,j)';   % 2x2
            % embed into block-diagonal positions
            block = kron(proj, rho_tilda_F);
            rho_total = rho_total + block;
        end

        disp(['Iteration ', num2str(iter-1), ':']);
        disp('rho_tilda_F ='); disp(rho_tilda_F);
        disp('rho_total =');     disp(rho_total);

        % Step 2: Oracle evolution and measurement
        U_true = expm(-1i * H_true * t);
        psi_f  = U_true * psi0;
        p_u1   = abs(u1' * psi_f)^2;
        outcome = sample_from_distribution([p_u1, 1 - p_u1]);

        % Compute likelihoods and update weights
        likelihoods = zeros(1, N);
        psi_all = zeros(2, N);
        for j = 1:N
            U_j = expm(-1i * H_list{j} * t);
            psi_j = U_j * psi0;
            psi_all(:,j) = psi_j;
            pj = abs(u1' * psi_j)^2;
            likelihoods(j) = (outcome==0)pj + (outcome==1)(1-pj);
        end

        weights = weights .* likelihoods;
        weights = weights / sum(weights);

        iter = iter + 1;
    end
    iter_arr(idx)=iter;
end
disp('results:')
disp(iter_arr);
% Helper functions
function t = PGH(H_list, weights)
    i1 = discrete_sample(weights);
    i2 = discrete_sample(weights);
    while i2 == i1
        i2 = discrete_sample(weights);
    end
    deltaH = H_list{i1} - H_list{i2};
    nH = norm(deltaH, 'fro');
    t = (nH < 1e-6) * 0.1 + (nH >= 1e-6) * (1 /  nH);
end

function idx = discrete_sample(w)
    r = rand(); cw = cumsum(w);
    idx = find(r <= cw, 1, 'first');
end

function outcome = sample_from_distribution(p)
    if rand() < p(1), outcome = 0; else, outcome = 1; end
end
