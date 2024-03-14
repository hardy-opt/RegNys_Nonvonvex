function  nys_curve_expLM(darg,col,e) 

    clc;
    close all;
    addpath(genpath(pwd));
    NUM_RUN = 3;
    NUM_EPOCH = e;
    P = 10;  %Partition for DP sampling
    K = 0;  % No. of clusters for DP sampling
    dat = strcat('Nonconvex_result_Jan24/RSN/',darg);  % result path
    %gs = gamma = square root ; go = gamma = one
    method = {'RS-RNM-GS','RNYS-GSA','RNYS-GSB','RNYS-GSC','RNYS-GSD','RNYS-GSE','RNM-GS' };
    omethod = {'SVRG-LBFGS', 'SVRG-SQN', 'adam', 'SQN', 'OBFGS', 'SVRG', 'SGD', 'LBFGS','GD','NG','RNGS','NEWTON'};
    %omethod = {'adam','SGD','NEWTON'};
    
    COLS = col;

    for s=1:NUM_RUN
        for reg= [ 1e-5] % lambda l_2 regularizer
            for step = [1 ]
                data = loaddata(s, reg, step, dat);
                BATCHES = size(data.x_train,2);%add batches with different size ex: [128 256]
                for c_two = [2  ] % c_2 is multiple of negative eigen and it should be > 1

                    for m= [7] %method from method_vector
                        for COL =  COLS

                            if COL > size(data.x_train,1)
                                break;
                            end

                            for c_one= [2 1 ]% 0.01 0.001]
                                %rng(s); % do not remove 
                                %disp([m, method{m}])
                                options.batch_size = size(data.x_train,2);
                                BATCH_SIZE = size(data.x_train,2);
                                fprintf('K%d - B%d - %s - Reg:%f - Step:%f - c_two:%f - c_one:%f - Run:%d\n', COL, BATCH_SIZE, method{m}, reg, step, c_two, c_one, s);
                                options.max_epoch=NUM_EPOCH;    
                                problem = SVR(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                                options.w_init = data.w_init;   
                                options.step_alg = 'fix'; 
                                options.step_init = step; 
                                options.verbose = 2;
                                options.column = COL;
                                options.partitions = P;
                                options.clusters = K;
                                mc = col;
                                Name = sprintf('%s/K%d_B%d_%s_%.1e_reg_%.1e_ctwo_%.1e_cone_%.1e_run_%d.mat',dat, COL, BATCH_SIZE, method{m},options.step_init,reg, c_two, c_one,s);
                                
                                if m==1

                                        gamma = 0.5;
                                        [w_s1, info_s1] = Reg_RSN(problem, options,reg,c_one,c_two,mc,gamma);  
                                        save(Name,'info_s1');
                                    

                                elseif m==2

                                        gamma = 0.5;
                                        [w_s1, info_s1] = RNYSA(problem, options,reg,c_one,c_two,mc,gamma);  
                                        save(Name,'info_s1');
                                    

                                elseif m==3 

                                        gamma = 0.5;
                                        [w_s1, info_s1] = RNYSB(problem, options,reg,c_one,c_two,mc,gamma);  
                                        save(Name,'info_s1');

                                elseif m==4

                                        gamma = 0.5;
                                        [w_s1, info_s1] = RNYSC(problem, options,reg,c_one,c_two,mc,gamma);  
                                        save(Name,'info_s1');  
                                
                                elseif m==5

                                        gamma = 0.5;
                                        [w_s1, info_s1] = RNYSD(problem, options,reg,c_one,c_two,mc,gamma);  
                                        save(Name,'info_s1');  

                                elseif m==6

                                        gamma = 0.5;
                                        [w_s1, info_s1] = RNYSE(problem, options,reg,c_one,c_two,mc,gamma);  
                                        save(Name,'info_s1');  
                                    
                                elseif m==7
                                    
                                      gamma = 0.5;
                                      options.sub_mode = 'STANDARD';                                      
                                      options.step_alg = 'backtracking';
                                      [w_s1, info_s1] = Reg_newton(problem, options,c_one,c_two,gamma); % Reg_newton
                                      save(Name,'info_s1');

                                end
 
                            end
                            
                        end
                    end
                end
                 for BATCH_SIZE = BATCHES

                    if BATCH_SIZE > size(data.x_train,2)
                        break;
                    end

                    for m = [ ] %9
                        
                        fprintf('%s - Reg:%f - Step:%f  - Run:%d\n', omethod{m}, reg, step, s);
                        options.max_epoch=NUM_EPOCH;    
                        problem = SVR(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                        options.w_init = data.w_init;   
                        options.step_alg = 'fix';
                        options.step_init = step; 
                        options.verbose = 2;
                        options.batch_size = BATCH_SIZE;

                        Name = sprintf('%s/B%d_%s_%.1e_reg_%.1e_run_%d.mat',dat,BATCH_SIZE,omethod{m},options.step_init,reg,s);

                        if m==1    

                            options.sub_mode='SVRG-LBFGS';
                            %options.sub_mode= 'Lim-mem';
                            [w_s1, info_s1] = slbfgs(problem, options);

                        elseif m==2

                            options.sub_mode='SVRG-SQN';
                            %options.sub_mode= 'Lim-mem';
                            [w_s1, info_s1] = slbfgs(problem, options);

                        elseif m==3

                            options.step_alg = 'decay-2'; 
                            options.sub_mode='Adam';
                            [w_s1, info_s1] = adam(problem, options);

                        elseif m==4

                            options.store_grad = 0;
                            options.sub_mode = 'SQN';
                            options.step_alg = 'decay-2';
                            [w_s1, info_s1] = slbfgs(problem, options);

                        elseif m==5

                            options.sub_mode = 'Lim-mem';
                            [w_s1, info_s1] = obfgs(problem, options);

                        elseif m==6

                             %options.step_alg = 'decay-2'; 
                             [w_s1, info_s1] = svrg(problem, options);

                        elseif m==7

                            options.step_alg = 'decay-2'; 
                            [w_s1, info_s1] = sgd(problem, options);
                       
                        elseif m==8
                           
                            options.sub_mode = 'STANDARD';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = lbfgs(problem, options);

                        elseif m==9
                            options.sub_mode = 'STANDARD';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = grd(problem, options);
                         
                        elseif m==10
                            options.sub_mode = 'STANDARD';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = reg_newton(problem, options,ct);
                        elseif m==11
                            options.sub_mode = 'STANDARD';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = rngd(problem, options,col);
                        
                        elseif m==12
                            %options.max_epoch=15;    

                            %options.sub_mode = 'STANDARD';
                            options.sub_mode = 'CHOLESKY';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = newton(problem, options);

                        end                    

                        save(Name,'info_s1');

                    end
                end
            end
        end
    end
end

function [data]=loaddata(s,reg,step,dat)
    strs = strsplit(dat,'/');
    if strcmp(strs{end}, 'REALSIM')
        data = REALSIM(s,reg,step);
    elseif strcmp(strs{end}, 'CIFAR10B')
        data = CIFAR10B(s,reg,step);
    elseif strcmp(strs{end}, 'MNISTB')
        data = MNISTB(s,reg,step);
    elseif strcmp(strs{end}, 'EPSILON')
        data = EPSILON(s,reg,step);
    elseif strcmp(strs{end}, 'ADULT')
        data = ADULT(s,reg,step);
    elseif strcmp(strs{end}, 'W8A')
        data = W8A(s,reg,step);
    elseif strcmp(strs{end}, 'ALLAML')
        data = ALLAML(s,reg,step);
    elseif strcmp(strs{end}, 'SMK_CAN')
        data = SMK_CAN(s,reg,step);
    elseif strcmp(strs{end}, 'GISETTE')
        data = GISETTE(s,reg,step);
    elseif strcmp(strs{end}, 'A8AN')
        data = A8A(s,reg,step);
    elseif strcmp(strs{end}, 'MRI')
    data = MRI(s);
    elseif strcmp(strs{end}, 'IJCNN')
        data = IJCNN(s,reg,step);
    elseif strcmp(strs{end}, 'WEBSPAM')
        data = WEBSPAM(s,reg,step);
    elseif strcmp(strs{end}, 'RCV1')
        data = RCV1(s,reg,step);    
    else
        disp('Dataset tho de');
    end
end
