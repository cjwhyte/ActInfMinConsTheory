%% Active Inference Model of rewed Troxler Fading

% Single level POMDP model of binocular rivalry originally reported in Parr, Corcoran, Friston, and Hohwy (2019).
% Adpated to account for Levelt's Laws and their potential violation under reward 

% Christopher Whyte
% 04/11/2025

clear 
close all
rng('shuffle')

%% Parameters 

% set reward to 1 to simulate a pragmatic value vs epistemic value tradeoff
% set reward to 0 for purely epistemit behaviour (i.e., typical rivalry
% conditions)

reward = 0;
if reward == 1
    rew = .5;
elseif reward == 0
    rew = 0;
end 

% beliefs about volatility for each stimulus
volatility_1 = .8;
volatility_2 = .8;

% beliefs about precision
pL = .3;
pR = .3;

% background_precision when fovea is not present
background_precision = 0;
% number of time steps
T = 20;

%% Level 1
%==========================================================================

% prior beliefs about initial states
% --------------------------------------------------------------------------

D{1} = [1 0]';   % Stim 1 {grating1, blank}
D{2} = [1 0]';   % Stim 2 {grating2, blank}
D{3} = [1 1]';   % Attention {Stim1, Stim2}

d = D;
d{1} = [1 1]';
d{2} = [1 1]';
d{3} = [1 1]';

% probabilistic mapping from hidden states to outcomes: A
% --------------------------------------------------------------------------

% A{1} = Stim1, A{2} = Stim2

% --- Generative process
% -- A{1}
for j = 1:length(D{2})
    for k = 1:length(D{3})
        A{1}(:,:,j,k) = [1 0; % grating 
                         0 1];% blank
    end 
end 

%- - A{2}
for j = 1
    for k = 1:length(D{3})
        A{2}(:,:,j,k) = [1 1; % grating 
                         0 0];% blank
    end 
end 
for j = 2
    for k = 1:length(D{3})
        A{2}(:,:,j,k) = [0 0; % grating 
                         1 1];% blank
    end 
end 

% --- Generative model

a = A;
% reduce background_precision unless attention is present
a{1} = spm_softmax(background_precision*a{1});
a{2} = spm_softmax(background_precision*a{2});

% -- a{1}
for j = 1:length(D{2})
    for k = 1
        a{1}(:,:,j,k) = [1-pL pL;
                         pL   1-pL];
    end 
end 

% -- a{2}
for j = 1
    for k = 2
        a{2}(:,:,j,k) = [1-pR 1-pR;
                         pR   pR];
    end 
end 
for j = 2
    for k = 2
        a{2}(:,:,j,k) = [pR   pR;
                         1-pR 1-pR];
    end 
end 

% turn off learning/novelty value
a{1} = a{1}*100;
a{2} = a{2}*100;


% Transitions between states: B
% --------------------------------------------------------------------------

% -- generative process
B{1} = eye(2,2);
B{2} = eye(2,2); 
B{3}(:,:,1) = [1 1; 
               0 0];
B{3}(:,:,2) = [0 0; 
               1 1];

% -- generative model
b = B;

% introduce volatility into transitions
b{1} = spm_softmax(volatility_1*b{1});
b{2} = spm_softmax(volatility_2*b{2});

% prevent b learning 
b{1} = b{1}*100;
b{2} = b{2}*100;
b{3} = b{3}*100;
           
% Policies
% --------------------------------------------------------------------------
U(1,1,:)  = [1 1 1]';  % Stim 1
U(1,2,:)  = [1 1 2]';  % Stim 2

% C matrices (outcome modality by timesteps)
% --------------------------------------------------------------------------
C{1} = zeros(2,T);
C{1}(1,:) = rew;
C{2} = zeros(2,T);

% MDP Structure
% --------------------------------------------------------------------------
mdp.T = T;                      % number of updates
mdp.A = A;                      % observation model (generative process)
mdp.B = B;                      % transition probabilities (generative process)
mdp.D = D;                      % prior over initial states
mdp.C = C;                      % preferences
mdp.U = U;                      % policies
mdp.b = b;                      % transition probabilities (generative model)
mdp.a = a;                      % observation model (generative model)
mdp.erp = 1;
mdp.alpha = 10;

mdp.Aname = {'Stim1', 'Stim2'};
mdp.Bname = {'Stim1', 'Stim2','Attention location'};

clear A B D C U b

MDP = spm_MDP_check(mdp);
MDP = spm_MDP_VB_X_ActInfConsTheory(MDP);

%% Basic sim plots

% trial plots
spm_figure('GetWin','Stim 1, Stim 2, Attention location'); clf
spm_MDP_VB_trial(MDP);

% stimulus 1
figure(3)
imagesc(MDP.X{1})
cmap = gray(256);
colormap(flipud(cmap))
title('stim 1 beliefs')
set(gca,'xtick',[1:T])
set(gca,'ytick',[])
set(gca,'fontsize',14)
xlabel('Time')
% stimulus 2
figure(4)
imagesc(MDP.X{2})
cmap = gray(256);
colormap(flipud(cmap))
title('stim 2 beliefs')
set(gca,'xtick',[1:T])
set(gca,'ytick',[])
set(gca,'fontsize',14)
xlabel('Time')
% attention 
figure(5)
imagesc(MDP.X{3})
cmap = gray(256);
colormap(flipud(cmap))
title('attention beliefs')
set(gca,'xtick',[1:T])
set(gca,'ytick',[])
set(gca,'fontsize',14)
xlabel('Time')

%% Simulation params

% num sims
num_sims = 50;
sim_length = 30;

%% Simulation Levelt's 4th Law

pb_L4 = linspace(.2,.45,10);

sim_counter = 0;
for prec = 1:size(pb_L4,2)
    rng('default')
    for ns = 1:num_sims
        
        sim_counter = sim_counter + 1;
        
        disp(['sim = ',num2str(sim_counter)]);
        
        mdp_L4 = mdp;
        
        pL = pb_L4(prec); pR = pb_L4(prec);
        
        a_L4 = a;
        
        %-- a{1}
        for j = 1:2
            for k = 1
                a_L4{1}(:,:,j,k) = [1-pL pL;
                                    pL   1-pL];
            end 
        end 
        %-- a{2}
        for j = 1
            for k = 2
                a_L4{2}(:,:,j,k) = [1-pR 1-pR;
                                    pR   pR];
            end 
        end 
        for j = 2
            for k = 2
                a_L4{2}(:,:,j,k) = [pR   pR;
                                    1-pR 1-pR];
            end 
        end 
        
        mdp_L4.a{1} = a_L4{1};
        mdp_L4.a{2} = a_L4{2};
        mdp_L4.T = sim_length;
        mdp_L4.C{1} = zeros(2,sim_length);
        mdp_L4.C{2} = zeros(2,sim_length);
        
        % run sim
        MDP_L4 = spm_MDP_VB_X_ActInfConsTheory(mdp_L4);
        
        % calculate domunance durations
        perceiveL = MDP_L4.X{3}(1,:) > (MDP_L4.X{3}(2,:));
        perceiveR = MDP_L4.X{3}(2,:) > (MDP_L4.X{3}(1,:));

        vals = find(perceiveL>0);
        diff_tp = diff(vals);       % find differences between time points
        nz = find([diff_tp inf]>1); % grab non-zero values
        durationL = diff([0 nz]);   % find differences between successive non-zero values

        vals = find(perceiveR>0);
        diff_tp = diff(vals);
        nz = find([diff_tp inf]>1);
        durationR = diff([0 nz]);

        duration_L{prec,ns} = durationL;
        duration_R{prec,ns} = durationR;
        durations{prec,ns} = [durationL,durationR];
        alternations(prec,ns) = numel(durationL) + numel(durationR);
        
    end 
end 

alternations_mean = mean(alternations,2);
alternations_err = std(alternations',1)./sqrt(num_sims);

for prec = 1:size(pb_L4,2)
    total_durations_LV4 = [];
    for  ns = 1:num_sims
        total_durations_LV4 = [total_durations_LV4, durations{prec,ns}];
    end 
    Levelt_durations_mean(prec) = mean(total_durations_LV4);
    Levelt_durations_err(prec) = std(total_durations_LV4)/sqrt(num_sims);
end 


%% Levelt's second law

pb_L2 = linspace(.1,.11,10);

sim_counter = 0;
for prec = 1:size(pb_L2,2)
    rng('default')
    for ns = 1:num_sims
        
        sim_counter = sim_counter + 1;
        
        disp(['sim = ',num2str(sim_counter)]);
        
        mdp_L3 = mdp;
        
        pL = .1; pR = pb_L2(prec);
        
        a_L3 = a;
        
        %-- a{1}
        for j = 1:2
            for k = 1
                a_L3{1}(:,:,j,k) = [1-pL pL;
                                    pL   1-pL];
            end 
        end 
        %-- a{2}
        for j = 1
            for k = 2
                a_L3{2}(:,:,j,k) = [1-pR 1-pR;
                                    pR   pR];
            end 
        end 
        for j = 2
            for k = 2
                a_L3{2}(:,:,j,k) = [pR   pR;
                                    1-pR 1-pR];
            end 
        end 
        
        mdp_L3.a{1} = a_L3{1};
        mdp_L3.a{2} = a_L3{2};
        mdp_L3.T = sim_length;
        mdp_L3.C{1} = zeros(2,sim_length);
        mdp_L3.C{2} = zeros(2,sim_length);
        mdp_L3.C{2}(1,:) = rew;
        
        % run sim
        MDP_L3 = spm_MDP_VB_X_ActInfConsTheory(mdp_L3);
        
        % calculate domunance durations
        perceiveL = MDP_L3.X{3}(1,:) > (MDP_L3.X{3}(2,:));
        perceiveR = MDP_L3.X{3}(2,:) > (MDP_L3.X{3}(1,:));

        vals = find(perceiveL>0);
        diff_tp = diff(vals);       % find differences between time points
        nz = find([diff_tp inf]>1); % grab non-zero values
        durationL = diff([0 nz]);   % find differences between successive non-zero values

        vals = find(perceiveR>0);
        diff_tp = diff(vals);
        nz = find([diff_tp inf]>1);
        durationR = diff([0 nz]);

        duration_L2{prec,ns} = durationL;
        duration_R2{prec,ns} = durationR;
        
    end 
end 

for prec = 1:size(pb_L2,2)
    total_durations_LV2 = [];
    durations_L = []; durations_R = []; 
    for  ns = 1:num_sims
        durations_L = [durations_L, duration_L2{prec,ns}];
        durations_R = [durations_R, duration_R2{prec,ns}];
    end 
    total_durations_L_mean(prec) = mean(durations_L);
    total_durations_R_mean(prec) = mean(durations_R);
    total_durations_L_err(prec) = std(durations_L)./sqrt(num_sims);
    total_durations_R_err(prec) = std(durations_R)./sqrt(num_sims);
end 

%% Levelt's law plots
set(0,'defaulttextinterpreter','latex')

figure(8)
errorbar(1-pb_L4,alternations_mean/alternations_mean(1),alternations_err/alternations_mean(1),'k','linewidth', 4)
ylim([.8 1.05]);
set(gca, 'XDir','reverse')
ylabel('Reversal Frequency')
xlabel('Precision (a.u.)')
ax = gca;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
ax.FontSize = 25; 
box off
axis padded

figure(9)
hold on
errorbar((1-pL)-(1-pb_L2),total_durations_L_mean, total_durations_L_err,'k','linewidth', 4)
errorbar((1-pL)-(1-pb_L2),total_durations_R_mean, total_durations_R_err,'k--','linewidth', 4)
set(gca, 'XDir','reverse')
ylabel('Dominance Duration (a.u)')
xlabel('Precision (a.u.)')
ylim([1,2.5])
ax = gca;
ax.LineWidth = 2;
set(gca, 'FontName', 'Times')
ax.FontSize = 25; 
axis padded
box off
