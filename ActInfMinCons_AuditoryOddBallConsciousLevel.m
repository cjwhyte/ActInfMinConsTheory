%% Auditory oddball conscious level simulations 

% Christopher Whyte 
% 04/11/2025

clear
close all

%% Level 1: Perception of individual stimuli
%==========================================================================

% prior beliefs about initial states
% --------------------------------------------------------------------------

D{1} = [1 1]'; % stimulus tone {high, low}

d = D;

% Here we seperate the generative process (the capital D)
% from the generative model (the lower case d) allowing learning to occur
% (i.e. to acccumulate concentration paramaters) in the generative model, 
% independent of the generative process.

% probabilistic (likelihood) mapping from hidden states to outcomes: A
% --------------------------------------------------------------------------

% outcome modality 1: stimulus tone
A{1}= [1 0;  %high tone
       0 1]; %low tone

% seperate generative model from generative process
a = A;

% reduce precision
pr1 = 2; % precision (inverse termperature) parameter (lower = less precise)
a{1} = spm_softmax(pr1*log(A{1}+exp(-4)));

a = a{1}*100;


% Transitions between states: B
% --------------------------------------------------------------------------

B{1}= [1 0; %high tone
       0 1];%low tone

% MDP Structure
% --------------------------------------------------------------------------
mdp_1.T = 1;                      % number of updates
mdp_1.A = A;                      % likelihood mapping
mdp_1.B = B;                      % transition probabilities
mdp_1.D = D;                      % prior over initial states
mdp_1.d = d;
mdp_1.a = a;
mdp_1.erp = 1;

mdp_1.Aname = {'Stimulus'};
mdp_1.Bname = {'Stimulus'};

clear a d A B D

MDP_1 = spm_MDP_check(mdp_1);

clear mdp_1

%% Level 2: Slower-timescale representations of perceived stimulus sequences
%==========================================================================

% prior beliefs about initial states in generative process (D) and
% generative model (d) in terms of counts (i.e., concentration parameters)
% --------------------------------------------------------------------------
D2{1} = [1 1 1 1]'; % Sequence type: {high, low, high-low, low-high}
D2{2} = [1 0 0 0 0 0]'; % time-in-trial
D2{3} = [1 0 0]'; % Report: {null, same, different} 

d2 = D2;
d2{2} = d2{2}*100;
d2{3} = d2{3}*100;

% --------------------------------------------------------------------------

% --- Stimulus
for i = 1:6
    for j = 1:3
        A2{1}(:,:,i,j) = [1 0 1 0; %high
                          0 1 0 1];%low
    end 
end

% oddball at fourth timestep for high-low and low-high hidden states. 
for i = 4
    for j = 1:3
        A2{1}(:,:,i,j) = [1 0 0 1; %high
                          0 1 1 0];%low
    end
end

% --- Report
for i = 1:6
    for j = 1:3
        A2{2}(:,:,i,j) = [1 1 1 1; %null
                          0 0 0 0; %incorrect
                          0 0 0 0];%correct                    
    end
end

% report "same"
for i = 6
    for j = 2
        A2{2}(:,:,i,j) = [0 0 0 0; %null
                          0 0 1 1; %incorrect
                          1 1 0 0];%correct
    end
end

% report "different"
for i = 6
    for j = 3
        A2{2}(:,:,i,j) = [0 0 0 0; %null
                          1 1 0 0; %incorrect
                          0 0 1 1];%correct
    end
end

a2 = A2; % likelihood (concentration parameters) for generative model

% reduce precision
pr2 = 2; % precision (inverse termperature) parameter (lower = less precise)
a2{1} = spm_softmax(pr2*log(A2{1}+exp(-4)));

a2{1} = a2{1}*100;
a2{2} = a2{2}*100;

% Transition probabilities: B
% --------------------------------------------------------------------------

% Precision of sequence mapping
B2{1} = eye(4,4); % maximally precise identity matrix (i.e., the true 
                  % sequence is stable within a trial)

B2{2} = [0 0 0 0 0 0;
         1 0 0 0 0 0;
         0 1 0 0 0 0;
         0 0 1 0 0 0;
         0 0 0 1 0 0;
         0 0 0 0 1 1]; % Deterministically transition through trial sequence
 
     
% Report
B2{3}(:,:,1) = [1 1 1;
                0 0 0;
                0 0 0]; % Pre-report    
B2{3}(:,:,2) = [0 0 0;
                1 1 1;
                0 0 0]; % Report "same"   
B2{3}(:,:,3) = [0 0 0;
                0 0 0;
                1 1 1]; % Report "different"

% transition precision
b2 = B2;
b2{1} = b2{1}*100;
b2{2} = b2{2}*100;
b2{2} = b2{2}*100;
           
% Policies
% --------------------------------------------------------------------------

 T = 6;  % number of timesteps
 Nf = 3; % number of factors
 Pi = 2; % number of policies
 V2 = ones(T-1,Pi,Nf);

% Report: "same" (left column) or "different" (right column)
 V2(:,:,3) = [1 1; 
              1 1;
              1 1;
              1 1;
              2 3];

% C matrices (outcome modality by timestep)
% --------------------------------------------------------------------------
C2{1} = zeros(2,T);

% report
C2{2} = [0 0 0 0 0 0;  % no feedback yet
         0 0 0 0 0 -1; % preference not to be incorrect at last timestep
         0 0 0 0 0 1]; % preference for being correct at last timestep

% MDP Structure
% --------------------------------------------------------------------------

mdp.MDP = MDP_1; % lower level model

mdp.link = [1 0]; % identifies lower level state factors (rows) with higher  
                  % level observation modalities (columns). Here this means the
                  % first observation at the higher level corresponds to
                  % the first state factor at the lower level.

mdp.T = T;                      % number of time points
mdp.A = A2;                     % likelihood mapping for generative process
mdp.a2 = a2;                    % likelihood mapping for generative model
mdp.B = B2;                     % transition probabilities for generative process
mdp.b = b2;                     % transition probabilities for generative model
mdp.C = C2;                     % preferred outcomes
mdp.D = D2;                     % priors over initial states for generative process
mdp.d = d2;                     % priors over initial states for generative model
mdp.V = V2;                     % policies
mdp.erp = 1;                    % reset/decay paramater

mdp.Aname = {'Stimulus', 'Report Feedback'};
mdp.Bname = {'Sequence', 'Time in trial', 'Report'};


% level one labels 
label.factor{1}   = 'Stimulus';   label.name{1}    = {'High','Low'};
label.modality{1} = 'Stimulus';   label.outcome{1} = {'High','Low'};
mdp.MDP.label = label;

label.factor{1}   = 'Sequence type';   label.name{1}    = {'High','Low','High-low','Low-high'};
label.factor{2}   = 'Time in trial';    label.name{2}    = {'T1', 'T2', 'T3', 'T4', 'T5', 'T6'};
label.factor{3}   = 'Report';    label.name{3}    = {'Null', 'Same', 'Different'};
label.modality{1} = 'Tone';    label.outcome{1} = {'High', 'Low'};
label.modality{2} = 'Feedback';  label.outcome{2} = {'Null','Incorrect','Correct'};
label.action{3} = {'Null','Same','Different'};
mdp.label = label;

mdp = spm_MDP_check(mdp);
MDP = spm_MDP_VB_X_ActInfConsTheory(mdp);

%% Simulate all conditions

% Here we specify the number of trials N and use a deal function (which copies 
% the input to N outputs) to create 10 identical mdp structures. We can
% then pass this to the spm_MDP_VB_X() script, which sequentially updates
% the concentration paramaters aquired on each trial and passes them to the
% mdp structure for the next trial (allowing learning to occur).

N = 10; %number of trials

% Local deviation - global standard
mdp.s = 3; % first nine trials are high-low 
MDP_condition1(1:N) = deal(mdp);
MDP_condition1(10).s = 3; % tenth trial is also high-low 
MDP_LDGS = spm_MDP_VB_X_ActInfConsTheory(MDP_condition1);

% Local standard - global deviation
mdp.s = 3; % first nine trials are high-low
MDP_condition2(1:N) = deal(mdp);
MDP_condition2(10).s = 1; % tenth trial is a high trial 
MDP_LSGD = spm_MDP_VB_X_ActInfConsTheory(MDP_condition2);

% Local deviation - global standard: sleep
mdp.s = 3; % first nine trials are high-low 
b2 = B2;
pr2 = .8;
b2{1} = spm_softmax(pr2*log(B2{1}+exp(-4)));
b2{1} = b2{1}*100;
b2{2} = b2{2}*100;
b2{2} = b2{2}*100;
mdp.b = b2;     
MDP_condition1(1:N) = deal(mdp);
MDP_condition1(10).s = 3; % tenth trial is also high-low 
MDP_LDGS_sleep = spm_MDP_VB_X_ActInfConsTheory(MDP_condition1);

% Local standard - global deviation: sleep
mdp.s = 3; % first nine trials are high-low
b2 = B2;
pr2 = .8;
b2{1} = spm_softmax(pr2*log(B2{1}+exp(-4)));
b2{1} = b2{1}*100;
b2{2} = b2{2}*100;
b2{2} = b2{2}*100;
mdp.b = b2;     
MDP_condition2(1:N) = deal(mdp);
MDP_condition2(10).s = 1; % tenth trial is a high trial 
MDP_LSGD_sleep = spm_MDP_VB_X_ActInfConsTheory(MDP_condition2);

%% custom ERP plots

% The ERP plotting routines give three outputs: 
% [level 2 ERPs, level 1 ERPs, indices]
% There are 32 time indices per time step/epoch of gradient decent. Here   
% there are 6 timesteps so there are 32x6 = 192 individual time indexes.
% The level 1 and 2 ERPs are the first derivitives at each time index.

[u1_1,v1_1,ind] = spm_MDP_VB_ERP(MDP_LDGS(1),1);  
[u1_10,v1_10] = spm_MDP_VB_ERP(MDP_LDGS(10),1); 

[u2_1,v2_1] = spm_MDP_VB_ERP(MDP_LSGD(1),1);  
[u2_10,v2_10] = spm_MDP_VB_ERP(MDP_LSGD(10),1); 

[u1_1_sleep,v1_1_sleep] = spm_MDP_VB_ERP(MDP_LDGS_sleep(1),1);  
[u1_10_sleep,v1_10_sleep] = spm_MDP_VB_ERP(MDP_LDGS_sleep(10),1); 

[u2_1_sleep,v2_1_sleep] = spm_MDP_VB_ERP(MDP_LSGD_sleep(1),1);  
[u2_10_sleep,v2_10_sleep] = spm_MDP_VB_ERP(MDP_LSGD_sleep(10),1); 

% The indexes below are arbitarily chosen to best represent the ERPs at the
% 4th time step, which starts at 96ms and ends at 128ms. To do this for 
% yourself we recommend just plotting the ERPs and selecting the appropiate
% time window. For example, the 1st level ERPs start at the begining of 
% the epoch whereas the 2nd ERPs appear towards the end of the epoch. So to
% include baseline periods in the plot you will likley have to select 
% slightly different time windows for each level as we have done here.

% index into 2nd level
index = (96:140); 
u1_1  = u1_1(index,:); % level 2
u1_10 = u1_10(index,:);

u2_1  = u2_1(index,:);% level 2
u2_10  = u2_10(index,:);

% index into ist level
index = (70:120); 
v1_1  = v1_1(index,:);% level 1
v1_10  = v1_10(index,:);
v2_1  = v2_1(index,:);% level 1
v2_10  = v2_10(index,:);

% index into 2nd level sleep
index = (96:140); 
u1_1_sleep  = u1_1_sleep(index,:); % level 2
u1_10_sleep = u1_10_sleep(index,:);

u2_1_sleep  = u2_1_sleep(index,:);% level 2
u2_10_sleep  = u2_10_sleep(index,:);

% index into ist level sleep
index = (70:120); 
v1_1_sleep  = v1_1_sleep(index,:);% level 1
v1_10_sleep  = v1_10_sleep(index,:);
v2_1_sleep  = v2_1_sleep(index,:);% level 1
v2_10_sleep  = v2_10_sleep(index,:);

time_low = (1:length(v1_1)); 
time_high = (1:length(u1_1)); 

%--- Lets make the plots! 

% low level plot
limits = [20 45 -.5 1.2];

figure(2)
hold on
plot(time_low,sum(v2_10,2),'b','LineWidth',4) % local standard
plot(time_low,sum(v1_10,2),'r','LineWidth',4) % local deviation
axis(limits)
set(gca,'FontSize',10)
title('Mismatch negativity: wake')
legend('Local standard', 'Local deviation')


% high level plot
limits = [1 45 -.5 .5];

figure(3)
hold on
plot(time_high,sum(u1_10,2),'b','LineWidth',4) % Global standard
plot(time_high,sum(u2_10,2),'r','LineWidth',4) % Global deviation
axis(limits)
set(gca,'FontSize',10)
title('P300: wake')
legend('Global standard', 'Global deviation')

% MMN (standard - mismatch)
limits = [1 45 -1.2 .5];

figure(4)
hold on
plot(time_low,sum(v2_10-v1_10,2),'r','LineWidth',4) 
plot(time_low,sum(v2_10_sleep-v1_10_sleep,2),'b','LineWidth',4) 
axis(limits)
set(gca,'FontSize',10)
legend('Wake','Sleep')
title('Mismatch negativity: local standard - local deviation')


% P300 (standard - mismatch)
limits = [1 45 -.5 .5];

figure(5)
hold on
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 1.5;
set(gca,'TickDir','out'); 
plot(time_high,sum(u1_10-u2_10,2),'LineWidth',4, 'Color', 'r') 
plot(time_high,sum(u1_10_sleep-u2_10_sleep,2),'LineWidth',4, 'Color', 'b') 
axis(limits)
legend('Wake','Sleep')
title('P300: Global standard - Global deviation')


