
%For Newton Sketch comparison with Nysttrom (also loot at the end of file)
%First load the file and run following


%For Norm difference
F1 = figure;
F2 = figure;

figure(F1);
fs = 22;
fss=32;
endd=length(Info.mvec)-9;
iter=length(Info.AvNp)-8;
semilogy(Info.mvec(1:endd,:),Info.AvNp(1:iter),'-.pr',Info.mvec(1:endd,:),Info.AvNq(1:iter),'--ok','LineWidth',2.3,'MarkerSize',12)%,'MarkerIndices', 1:2:123);
legend({'Nystrom','NS'},'Location','northoutside','FontSize',fs)
ax = gca;
ax.FontSize = fs;
xlabel('m-columns','FontSize',fss) 
ylabel('$\| H - N \|\ $ (log scale)','interpreter','latex','FontSize',fss)

%For CPU time
a = 8;
figure(F2);
fs=22;
fss=32;
%endd=length(Info.mvec)-1;
%plot(Info.mvec,Info.AvGp,'-.pk',Info.mvec,Info.AvGq,'--om',Info.mvec,Info.AvGr,'--sc','LineWidth',2,'MarkerSize',10); hold on;
semilogy(Info.mvec(1:endd),Info.AvGp(2:end-a),'-.pr',Info.mvec(1:endd),Info.AvGq(2:end-a),'--ok','LineWidth',2.3,'MarkerSize',12)%,'MarkerIndices', 1:2:123);
legend({'Nystrom','NS'},'Location','southeast','FontSize',fs)
% xticklabels({'1','1000','2000','3000','4000','5000'})
ax = gca;
ax.FontSize = fs;
xlabel('m-columns','FontSize',fss) 
ylabel('CPU time (log scale)','FontSize',fss) 

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % m = no. of columns picked
% % For Nystrom for m, Newton Sketch for m, Newton Setch with 4m
% %For Norm difference
% 
% fs = 22;
% fss=32;
% endd=length(Info.mvec)-1;
% iter=length(Info.AvNp)-0;
% semilogy(Info.mvec(1:endd,:),Info.AvNp(1:iter),'-.pk',Info.mvec(1:endd,:),Info.AvNq(1:iter),'--om',Info.mvec(1:endd,:),Info.AvNr(1:iter),'-sc','LineWidth',2,'MarkerSize',10)%,'MarkerIndices', 1:2:123);
% legend({'Nystrom','NS','NS (4m)'},'Location','northoutside','FontSize',fs)
% ax = gca;
% ax.FontSize = fs;
% xlabel('m-columns','FontSize',fss) 
% ylabel('$\| H - N \|\ $ (log scale)','interpreter','latex','FontSize',fss)
% 
% %For CPU time
% 
% % fs=22;
% fss=32;
% endd=length(Info.mvec)-1;
% %plot(Info.mvec,Info.AvGp,'-.pk',Info.mvec,Info.AvGq,'--om',Info.mvec,Info.AvGr,'--sc','LineWidth',2,'MarkerSize',10); hold on;
% semilogy(Info.mvec(1:endd),Info.AvGp(2:end-0),'-.pk',Info.mvec(1:endd),Info.AvGq(2:end-0),'--om',Info.mvec(1:endd),Info.AvGr(2:end-0),'--sc','LineWidth',2,'MarkerSize',10)%,'MarkerIndices', 1:2:123);
% legend({'Nystrom','NS','NS (4m)'},'Location','southeast','FontSize',fs)
% % xticklabels({'1','1000','2000','3000','4000','5000'})
% ax = gca;
% ax.FontSize = fs;
% xlabel('m-columns','FontSize',fss) 
% ylabel('CPU time (log scale)','FontSize',fss) 