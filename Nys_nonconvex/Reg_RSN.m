function [w, infos] = Reg_RSN(problem, in_options,reg,co,ct,mc,gamma)

    
    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    

    % set local options 
    %local_options.sub_mode = 'Nystrom_sgd';  
    local_options.mem_size = 20;    
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      
    
    
    col = options.column;
    K = options.clusters;
    P = options.partitions;
    
    %%%%%%%%%%%%%%%%
    lambda = [];
    g_nrm = [];
    dp=0;
    %%%%%%%%%%%%%%%%
    


    % set paramters
    if options.batch_size > n
        options.batch_size = n;
    end   
    
    if ~isfield(in_options, 'batch_hess_size')
        options.batch_hess_size = 20 * options.batch_size;
    end    

    if options.batch_hess_size > n
        options.batch_hess_size = n;
    end    
        
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    g = norm(problem.full_grad(w));
    num_of_bachces = floor(n / options.batch_size);     


    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);  
    
    % set start time
    start_time = tic();
    
    % display infos
    if options.verbose > 0
        fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', 'Reg_RSN', epoch, f_val, optgap);
    end     

    % main loop
    %while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
    while (epoch < options.max_epoch)
            w0 = w;
                   
                    if any(isnan(f_val)) || any(isinf(f_val)) || any(isnan(w0)) || any(isinf(w0))
                    return;
                    end
                    
          %  step = options.stepsizefun(total_iter, options);    
            rng(epoch);
            set = randperm(d,mc);

            [rsn,S] =  problem.randomized_sub(w,1:n,set,ct);

            % calculate gradient 
            ro = 1/2;
            cc = 0.3;%  1e-4;
            
            if gamma==0.5            

                rho = (co*sqrt(g)); % c_1 * |sqrt(g)|

            elseif gamma ==1

                rho = (co*(g)); % c_1 * ||g||

            end

            grad = problem.full_grad(w);
            rsn = rsn + rho*eye(size(rsn));
            grad_vect = S*grad;
            
                     NI = S'*pinv(rsn)*grad_vect;
                     g = norm(grad);
                     step = backtracking_line_search(problem, -NI, w, ro, cc);
                     %steps = max(step,1e-3);
                     v = step*NI;
                     w = w - v;
            

            
            total_iter = total_iter + 1;
        
        
        
        % measure elapsed time
        elapsed_time = toc(start_time);
        
%         lambda = [lambda 1/nfg];
%         grad = problem.grad(w,1:n);
        g_nrm = [g_nrm g];
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + epoch * options.batch_size;        
        epoch = epoch + 1;

        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);            

        % display infos
        if options.verbose > 0 %&& mod(epoch,10)==1
            fprintf('%s: Ep = %3d, cost = %.5e, optgap = %.4e, time=%0.3f, |g| = %.4e, c_one = %.e\n', 'Reg_RSN', epoch, f_val, optgap,elapsed_time,g,co);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
      
%  infos.RHO = RH;
% infos.lambda = lambda';
infos.g_nrm = g_nrm';
end
