function data = ADULT(seed,reg,ler)
    
    M = load('adult.mat');
    %data matrix M loads four files (x_train, y_train, x_test, y_test)
    
    [n,d] = size(M.x_train);

    D = M.x_train;
    
    %Data normalization (0 mean, unit varianve)
    s = std(D);
    s(s==0)=1;
    m=mean(D);
    D = (D-m)./s;
    
    D = [D  ones(n,1)];
    
    rng(seed);
    perm = randperm(n);
    A =  D(perm,:);
    B = M.y_train(perm);
    data.x_train = A';
    data.y_train = B';
    
    fprintf('This is Adult train data with n=%d, d=%d\n',size(data.x_train'));

    %%%%

    P = M.x_test;
    [nn,~] = size(M.x_test);
    %Data normalization (0 mean, unit varianve)
    s = std(P);
    s(s==0)=1;
    m=mean(P);
    P = (P-m)./s;
    
    P = [P  ones(nn,1)];
    
    rng(seed);
    perm = randperm(nn);
    data.x_test =  P(perm,:)';
    data.y_test = M.y_test(perm)';


    fprintf('This is Adult test data with n=%d, d=%d\n',size(data.x_test'));

    %Initial point with different random seed
    rng(seed);
    w = randn(d+1,1);
    w = w./norm(w);
    data.w_init = w;


%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      tic;
%     options.max_epoch=2;
%     options.batch_size = 64;
%     % define problem definitions
%     problem = logistic_regression1(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
%     % For large data set, use sub-sample instead of data.x_train
     %data.w_init = inv(data.x_train*data.x_train'+reg*eye(size(data.x_train,1)))*data.x_train*data.y_train'; 
%     %options.w_init = data.w_init;   
%     options.step_alg = 'fix';
%     options.step_init = ler; 
%     options.verbose = 2;
%    % [w_sgd,~] = sgd(problem,options);
%     data.w_init = options.w_init;
%      toc                   
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end