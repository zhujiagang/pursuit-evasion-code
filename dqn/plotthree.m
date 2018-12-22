clc;
clear all;
close all;
qmax=importdata('qmax.t7');
length(qmax)

txtsize=17;
linewid=2.2;

subplot(1,3,1)
% plot(ex,'color','c')
% hold on
plot(qmax(7:4:length(qmax)),'color','k','linewidth',linewid)
set(gca,'linewidth',2.5,'FontSize',txtsize);
set(gca,'yticklabel',get(gca,'ytick'));
box off;
ylabel('Average action value (Q)','FontSize',txtsize)
xlabel('Training epochs','FontSize',txtsize)


subplot(1,3,2)
rewardhistory=importdata('reward_history.t7');
% plot(ex,'color','c')
% hold on
plot(rewardhistory(7:4:length(rewardhistory)),'color','k','linewidth',linewid)
axis([0 100 -1000 10000]) 
set(gca,'linewidth',2.5,'FontSize',txtsize);
set(gca,'yticklabel',get(gca,'ytick'));
box off;
ylabel('Average total reward per episode','FontSize',txtsize)
xlabel('Training epochs','FontSize',txtsize)

subplot(1,3,3)
stephistory=importdata('step_history.t7');
% plot(ex,'color','c')
% hold on

plot(stephistory(7:4:length(stephistory)),'color','k','linewidth',linewid)
axis([0 100 0 10000]) 
set(gca,'linewidth',2.5,'FontSize',txtsize);
set(gca,'yticklabel',get(gca,'ytick'));

box off;
ylabel('Average step per episode','FontSize',txtsize)
xlabel('Training epochs','FontSize',txtsize)

%ht=subtitle(3,'G4',txtsize)