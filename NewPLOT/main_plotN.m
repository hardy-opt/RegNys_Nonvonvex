function [] = main_plotN()
    close all;
    
    d =1; %dataet % 5 =Adult, 1= Realsim, 6 = W8A, 8=Gisette
    yp = 1;
    xp = 1;
    % yparams = {'cost', 'val_cost', 'acc_tr', 'acc_val','gnorm'};
    % xparams = {'time', 'epoch'};
    % plot_params.sort = yparams{1};
    % plot_params.y = yparams{1};
    % plot_params.x = xparams{1}; 

    % Markers = {'o','+','*','.','x','_','|','square','diamond','^','v','>','<','pentagram' }
    % To change X and Y axis properties GOTO LINE 44
    % To change dataset : Go to line 65
    %Relativer error LINE 219

    RUNS = 3;
    EPOCHS = 0;
    lambdas = 1e-5;% 1e-2 1e-4];

    etas = [ 1 1e-1 0.01 0.001];%
    %rhos c_twos
    c_twos = [2 1 0.1 0.01 0.001]; %
    %deltas c_ones
    c_ones = [2 1 0.1 0.01 0.001];

    if d== 5 % ADULT
        COLS = 30;%30;
        BSS = 32561;
        EPOCHS = 100;

    elseif d==8 % GISETTE
        COLS = 250;
        BSS = 6000;
        EPOCHS = 20;
    elseif d==6 % W8A
        COLS = 30;
        BSS = 49749;
        EPOCHS = 30;

    elseif d == 1 % realsim
        COLS = 500;
        BSS = 57848;
        EPOCHS = 20;
    end

    path = 'Nonconvex_result_Jan24/RSN/';

    
    datasets = {         % COL
        'REALSIM'   %1
        'CIFAR10B'  %2
        'MNISTB'    %3
        'EPSILON'   %4
        'ADULT'     %5 ---> 10
        'W8A'       %6 ---> 20
        'ALLAML'    %7
        'GISETTE'   %8 ---> 50 
        'MRI'       %9
        'IJCNN'     %10
        'A8AN'      %11
        };

    lw = 2.5;%RUNS;
    ms = 14;
    params = initN(lw, ms, lambdas, etas, c_twos, RUNS, EPOCHS, COLS(1), BSS(1),c_ones);
    
     sparams = {

% params('NEWTON')
params('RNM-GS')
params('RNM-GO')
params('RS-RNM-GS')
params('RS-RNM-GO')
%params('RNYS-GS')
%params('RNYS-GO')
params('RNYS-GSA')
params('RNYS-GSB') % complex
%params('RNYS-GSC')
%params('RNYS-GSD')
%params('RNYS-GSE')
params('GD')
%params('LBFGS')


        };
    
    yparams = {'cost', 'val_cost', 'acc_tr', 'acc_val','gnorm'};
    xparams = {'time', 'epoch'};
    plot_params.sort = yparams{1};
    plot_params.y = yparams{yp};
    plot_params.x = xparams{xp};
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %To change the data enter the number from the data list of that dataset
    
    for dsi=d %6
            disp(datasets{dsi})
    for l = 1 :length(lambdas)
%          sparams{9}.etas=etas(l);
%          sparams{1}.etas=etas(4);
         %sparams{2}.c_twos=c_twos(l);
         %subplot(1,4,l);

        figure;
        plot_method_lambda(strcat(path, datasets{dsi}, '/'), sparams, lambdas(l), plot_params, 0*1*1e-5);
%         xlim([0,200]);
%         saveas(gcf,strcat('s4qn_',lower(datasets{dsi}),'_vtc_lam',num2str(l+1),'.eps'),'epsc');
 %       legendmarkeradjust(20,20);
        fprintf('\n');
    end
    end
%    legend({'S4QN-N', 'S4QN-F', 'SVRG-LBFGS', 'OBFGS', 'SVRG-SQN', 'SQN', 'ADAM', 'NSGD-D', 'NSGD', 'NSVRG', 'NSVRG-D'})
% legend({'S4QN-N', 'S4QN-F', 'NSVRG-D', 'NSGD'});
end

function plot_method_lambda(dataset, sparams, lambda, plot_params, ref)
    hold on;
    fs=28;
    fss=32;
    set(gca, 'FontSize', fs);
    title(strcat('\lambda=',sprintf('10^{%0.0f}', log10(lambda))),'FontSize',fss);
    switch plot_params.x
        case 'time'
            xlabel('CPU time (seconds)','FontSize',fss);
        case 'epoch'
            xlabel('Epoch','FontSize',fss);
            %xlabel('Iterations','FontSize',fss);
    end
    switch plot_params.y
        case 'cost'
            ylabel('$f(w)\ $ (log scale)','Interpreter','latex','FontSize',fss);
        case 'val_cost'
            ylabel('Test Error (log scale)','FontSize',fss);
        case 'acc_tr'
            ylabel('Train Accuracy','FontSize',fss);
        case 'acc_val'
            ylabel('Test Accuracy','FontSize',fss);
        case 'gnorm'
            ylabel('$\| \nabla f(w) \|$','Interpreter','latex','FontSize',fss);
    end
    if ref>=0
        opt_cost=inf;
        for m = 1:length(sparams)
            hold on;
            [bestmu, bestsg, besteta, bestc_two, bestc_one] = find_bestN(dataset, sparams{m}.name, lambda, sparams{m}.etas, sparams{m}.c_twos, sparams{m}.EPOCHS, sparams{m}.RUNS, sparams{m}.COL, sparams{m}.BS, plot_params.sort,sparams{m}.c_ones);
            if bestmu.Count > 1
                y = bestmu(plot_params.y);
                if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost') || strcmp(plot_params.y,'gnorm')
                    opt_cost = min(opt_cost,min(y));
                end
            end
        end
    end
    maxy = -inf;
    miny = inf;
    for m = 1:length(sparams)
        hold on;
        [bestmu, bestsg, besteta, bestc_two, bestc_one] = find_bestN(dataset, sparams{m}.name, lambda, sparams{m}.etas, sparams{m}.c_twos, sparams{m}.EPOCHS, sparams{m}.RUNS, sparams{m}.COL, sparams{m}.BS, plot_params.sort,sparams{m}.c_ones);
        if bestmu.Count > 1
            if strcmp(plot_params.y, 'val_cost') || strcmp(plot_params.y,'acc_val')
                x=bestmu(plot_params.x);
                x=x(2:end);
            else
                x=bestmu(plot_params.x);
            end
            if ref>=0 && (strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost')) % || strcmp(plot_params.y,'gnorm'))
                y = (bestmu(plot_params.y)-opt_cost+ref)/(1+opt_cost);
                s = bestsg(plot_params.y)/(1+opt_cost);
            else
                y = bestmu(plot_params.y);
                s = bestsg(plot_params.y);
            end
            if bestc_two == -1
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f})',log10(besteta)));
            elseif bestc_two == 0
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f},',log10(besteta)),' c_2=\mid\midZ\mid\mid_{F})');
            else
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f},',log10(besteta)),' c_2=', sprintf('{%0.1f},', (bestc_two)),' c_1=', sprintf('{%0.1f})', (bestc_one)));
            end
            if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost') || strcmp(plot_params.y,'gnorm')
                [~, idx] = min(y);
                if bestc_two == -1
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  %-12s  %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, ' ', plot_params.y, y(idx), idx, round(x(idx)));
                else
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  c_two: %.1e c_one: %.1e %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, bestc_two, bestc_one, plot_params.y, y(idx), idx, round(x(idx)));
                end
                maxy = max(maxy,y(1));
                miny = min(miny,min(y));
            else
                [~, idx] = max(y);
                if bestc_two == -1
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  %-12s  %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, ' ', plot_params.y, y(idx), idx, round(x(idx)));
                else
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  c_two: %.1e c_one: %.1e %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, bestc_two, bestc_one, plot_params.y, y(idx), idx, round(x(idx)));
                end
                maxy = max(maxy,max(y));
                miny = min(miny,min(y));
            end
            idx=length(y);
            if length(sparams)==1
               displayname = strcat(displayname, '@\lambda=', sprintf('10^{%0.0f})', log10(lambda)));
               errorbar(x(1:5:idx), y(1:5:idx), s(1:5:idx), 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
               %plot(x(1:idx), y(1:idx), 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
            else
               errorbar(x(1:5:idx), y(1:5:idx), s(1:5:idx), 'linestyle', sparams{m}.line, 'color', sparams{m}.linecolor, 'Marker', sparams{m}.marker, 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
                %plot(x(1:1:idx), y(1:1:idx), 'linestyle', sparams{m}.line, 'color', sparams{m}.linecolor, 'Marker', sparams{m}.marker, 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname, 'MarkerIndices', 1:1:idx);
            end
        end
    end
    if maxy > miny 
        ylim([miny, maxy]);
    end
    %l = legend('FontSize',18,'Orientation','horizontal','NumColumns',3);
    %get(legend)
    %l.Orientation = 'horizontal';
    %legendmarkeradjust(20,20);

    if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost')  || strcmp(plot_params.y,'gnorm')
        set(gca, 'YScale', 'log');
    end
end
