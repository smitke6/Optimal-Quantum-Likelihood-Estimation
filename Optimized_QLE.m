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

H_list = {sigX, sigX2, sigZ, sigZ2};
sigX  = [0, 1; 1, 0];
sigZ  = [1, 0; 0, -1];
sigX2 = 2 * sigX;
sigZ2 = 2 * sigZ;

H_list = {sigX, sigX2, sigZ, sigZ2};
N = length(H_list);

convergence_iters = NaN(1, N);
success_flags = false(1, N);

for idx = 1:N
	disp('******************************************************************')
	disp('the right H is:'); disp(idx);
	H_true = H_list{idx};
	weights = ones(1, N) / N;
	num_iterations = 10;
	
	iter = 1;
	while iter <= num_iterations+1
		if max(weights) >= 0.9999
			convergence_iters(idx) = iter;
		break;
		end
		
		best_params = optimize_entropy(weights, H_list);
		t     = best_params(1);
		theta = best_params(2);
		phi   = best_params(3);
		alpha = best_params(4);
		beta  = best_params(5);
		disp("idial t")
		disp(t);
		
		psi0 = [cos(alpha); exp(1i*beta)*sin(alpha)];
		psi0 = psi0 / norm(psi0);
		
		U_true = expm(-1i * H_true * t);
		psi_f  = U_true * psi0;
		psi_f = psi_f / norm(psi_f);
		
		u1 = [cos(theta/2); exp(1i*phi)*sin(theta/2)];
		u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
		W = [u1, u2];
		
		psi_final = W * psi_f;
		psi_final = psi_final / norm(psi_final);
		p_u1 = abs(psi_final(1))^2;
		outcome = sample_from_distribution([p_u1, 1 - p_u1]);
		
		likelihoods = zeros(1, N);
		rho_Y = zeros(2);
		H_YF = 0;
		
		for j = 1:N
			U_j = expm(-1i * H_list{j} * t);
			psi_j = W * (U_j * psi0);
			psi_j = psi_j / norm(psi_j);
			pj = abs(psi_j(1))^2;
			pj = min(max(real(pj), 1e-12), 1 - 1e-12);
			likelihoods(j) = (outcome == 0)*pj + (outcome == 1)*(1 - pj);
			rho_Y = rho_Y + weights(j) * (psi_j * psi_j');
			H_YF = H_YF + weights(j) * (-pj*log2(pj) - (1 - pj)*log2(1 - pj));
		end
		
		ent_y = Entropy(diag(diag(rho_Y)));
		disp(['Iteration ', num2str(iter), ':']);
		disp('H(Y)='); disp(ent_y);
		disp('H(Y|F)='); disp(H_YF);
		disp('I(FY)='); disp(ent_y - H_YF);
		
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
	disp(convergence_iters - 1);
	disp('Did each H converge to correct hypothesis?');
	disp(success_flags);
	disp(['All correct: ', mat2str(all(success_flags))]);
	
	
	% ========================= FUNCTIONS =========================
	
	function best_params = optimize_entropy(weights, H_list)
		% Simulated Annealing optimization
		max_iters = 200;
		T_init = 1.0;
		alpha = 0.9;
		
		x = [rand*2*pi, rand*pi, rand*2*pi, rand*pi, rand*2*pi];
		cost = entropy_cost(x, weights, H_list);
		T = T_init;
		
		best_x = x;
		best_cost = cost;
		
		num_neighbors = 20; 
		
		for iter = 1:max_iters
			candidates = zeros(num_neighbors, 5);
			candidate_costs = zeros(num_neighbors, 1);
			
			for k = 1:num_neighbors
				step = [pi, pi/2, pi, pi/2, pi];
				x_new = x + (rand(1,5)*2 - 1) .* step * T;
				
				% wrap
				x_new(1) = mod(x_new(1), 2*pi);
				x_new(2) = mod(x_new(2), pi);
				x_new(3) = mod(x_new(3), 2*pi);
				x_new(4) = mod(x_new(4), pi);
				x_new(5) = mod(x_new(5), 2*pi);
				
				cost_new = entropy_cost(x_new, weights, H_list);
				
				candidates(k,:) = x_new;
				candidate_costs(k) = cost_new;
			end
			
			[min_cost, best_k] = min(candidate_costs);
			x_best = candidates(best_k, :);
			
			if min_cost < cost || rand < exp((cost - min_cost)/T)
			x = x_best;
			cost = min_cost;
			
			if cost < best_cost
				best_x = x;
				best_cost = cost;
			end
		end
		
		T = T * alpha;
	end
	
		best_params = best_x;
		fprintf("Best entropy after annealing: %.5f\n", best_cost);
	end
	
	function cost = entropy_cost(x, weights, H_list)
		t = x(1); theta = x(2); phi = x(3); alpha = x(4); beta = x(5);
		N = length(H_list);
		psi0 = [cos(alpha); exp(1i*beta)*sin(alpha)];
		rho_FY = zeros(2*N);
		
		for j = 1:N
			psi_j = expm(-1i * H_list{j} * t) * psi0;
			rho_j = psi_j * psi_j';
			rho_FY(2*j-1:2*j, 2*j-1:2*j) = weights(j) * rho_j;
		end
		
		u1 = [cos(theta/2); exp(1i*phi)*sin(theta/2)];
		u2 = [exp(-1i*phi)*sin(theta/2); -cos(theta/2)];
		W = [u1, u2];
		
		A_plus  = W' * [1, 0; 0, 0] * W;
		A_minus = eye(2) - A_plus;
		
		rho_plus  = kron(eye(N), A_plus) * rho_FY * kron(eye(N), A_plus);
		rho_minus = kron(eye(N), A_minus) * rho_FY * kron(eye(N), A_minus);
		
		prob_plus  = real(trace(rho_plus));
		prob_minus = real(trace(rho_minus));
		
		S_plus = 0; S_minus = 0;
		if prob_plus > 0
			rho_plus = rho_plus / prob_plus;
			S_plus = Entropy(rho_plus);
		end
		if prob_minus > 0
			rho_minus = rho_minus / prob_minus;
			S_minus = Entropy(rho_minus);
		end
		
		cost = prob_plus * S_plus + prob_minus * S_minus;
	end
	
	function S = Entropy(rho)
		rho = (rho + rho') / 2;
		eigvals = real(eig(rho));
		eigvals = eigvals(eigvals > 1e-12);
		S = -sum(eigvals .* log2(eigvals));
	end
	
	function outcome = sample_from_distribution(p)
		if rand() < p(1)
			outcome = 0;
		else
			outcome = 1;
		end
end
