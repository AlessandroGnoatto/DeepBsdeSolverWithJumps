%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all
close all

load 'C:\Users\mrcpa\Dropbox\Documenti\Marco\UNIVERSITA\RtdA_Verona\Ricerca\Deep Learning\20221005_Deep-Solver-master PureJump_ez_MP\Plot\CGMYsimRes.mat' 

step = 0:100:8000; step = step';
% step = 0:100:20000; step = step';

figure(1)
subplot(1,2,1)
% plot(step,Exp40_lam03(:,1),'LineWidth',3);
plot(step,CGMYnnPrice1(:,2),'LineWidth',3);
title('Loss Function');
xlabel('Iteration number');
ylabel('Loss');
axis tight;
set(gca,'FontSize',18)

subplot(1,2,2)
% plot(step,Exp40_lam03(:,2),'LineWidth',3);
plot(step,CGMYnnPrice1(:,3),'LineWidth',3);
hold on
yline(CGMYmcPrice1,'LineWidth',3,'Color','r') 
title('Initial Value');
xlabel('Iteration number');
ylabel('Y');
axis tight;
set(gca,'FontSize',18)