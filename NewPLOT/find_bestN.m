function [bestmu, bestsg, besteta, bestc_two, bestc_one] = find_bestN(dataset, method, lambda, etas, c_twos, EPOCHS, RUNS, COL, BS, sort_param,c_ones)
    bestmu = containers.Map();
    bestsg = containers.Map();
    besteta = -1;
    bestc_two = -1;
    bestc_one = -1;
    if strcmp(sort_param, 'cost') || strcmp(sort_param, 'val_cost') || strcmp(sort_param, 'gnorm')
        best = inf;
    else
        best = -inf;
    end
    for eta = etas
        if isempty(c_twos)
            [mu, sg] = mean_runN(dataset, method, lambda, eta, -1, EPOCHS, RUNS, COL, BS,-1);            
            if mu.Count > 0
                av=mu(sort_param);
                at=mu('epoch');
%                 if strcmp(sort_param, 'acc_val') || strcmp(sort_param, 'val_cost')
%                 at = at(2:end);
%                 end
                if strcmp(sort_param, 'cost') || strcmp(sort_param, 'val_cost') || strcmp(sort_param,'gnorm')
                    %auc=trapz(at,av);
                    auc=min(mu(sort_param));
                    if auc < best
                        best = auc; %min(mu(sort_param));
                        bestmu = mu;
                        bestsg = sg;
                        besteta = eta;
                        bestc_two = -1;
                        bestc_one = -1;
                    end
                else
                    %auc=trapz(at,av);
                    auc=max(mu(sort_param));
                    if auc > best
                        best = auc; %max(mu(sort_param));
                        bestmu = mu;
                        bestsg = sg;
                        besteta = eta;
                        bestc_two = -1;
                        bestc_one = -1;
                    end
                end
            end
        else
            for c_two = c_twos
                for c_one = c_ones
                        [mu, sg] = mean_runN(dataset, method, lambda, eta, c_two, EPOCHS, RUNS, COL, BS,c_one);
                        if mu.Count > 0
                            av=mu(sort_param);
                            at=mu('time');
                            if strcmp(sort_param, 'acc_val') || strcmp(sort_param, 'val_cost') || strcmp(sort_param, 'gnorm')
                                at = at(2:end);
                            end
                            auc=trapz(at,av); 
                            if strcmp(sort_param, 'cost') || strcmp(sort_param, 'val_cost') || strcmp(sort_param, 'gnorm')
                                if auc < best   %%min(mu(sort_param))
                                    best = auc;
                                    bestmu = mu;
                                    bestsg = sg;
                                    besteta = eta;
                                    bestc_two = c_two;
                                    bestc_one = c_one;
                                end
                            else
                                if auc > best   %max(mu(sort_param))
                                    best = auc;
                                    bestmu = mu;
                                    bestsg = sg;
                                    besteta = eta;
                                    bestc_two = c_two;
                                    bestc_one = c_one;
                                end
                            end
                        end
                end
            end
        end
    end
end
