clear; clc; close all;
bidsdir         =  '..\dataset\DTU\';
addpath (genpath([bidsdir 'snhl-master\snhl-master\src\examples\fieldtrip-lite-20220104\']));
addpath (genpath([bidsdir 'E:snhl-master\mTRF-Toolbox-master\']));

% only for qzl
data_types = {'1D','2D'};
paralen = 44*128;
sbnum = 21;
trnum = 32;

dataset = 'DTU';

data1D_name = [dataset '_1D.mat'];

EEG = zeros(sbnum,trnum,paralen,64);
ENV = zeros(sbnum,trnum,paralen,1);

fs = 128; % sampling rate
Wn = [1 50]/(fs/2);
order = 8;
[b,a] = butter(order,Wn,'bandpass');
% First, you'll need to point to where the mTRF toolbox is stored and 
% the BIDs directory for the source dataset (bidsdir):


if 1
% We import information about the participants
participants    = readtable(fullfile(bidsdir,'participants.tsv'),'FileType','text','Delimiter','\t','TreatAsEmpty',{'N/A','n/a'});

dataout         = cell(44,1);
sb = 0;
for subid = 1 : 44
    if (subid<21)||(subid == 41||subid==42||subid ==24)
        continue
    end
    sb = sb + 1;

    % The EEG data from each subject is stored in the following folder:
    eeg_dir     = fullfile(bidsdir,sprintf('sub-%0.3i',subid),'eeg');
    
    
    % The EEG data from sub-024 is split into two runs due to a break in the
    % experimental session. For this reason, we ensure that we loop over
    % these two runs for sub-024. For every other subject we do nothing.
    fname_bdf_file  = {};
    fname_events    = {};
    
    fname_bdf_file{1} = fullfile(eeg_dir,sprintf('sub-%0.3i_task-selectiveattention_eeg.bdf',subid));
    fname_events{1} = fullfile(eeg_dir,sprintf('sub-%0.3i_task-selectiveattention_events.tsv',subid));
    

    % Prepare cell arrays that will contain EEG and audio features
    eegdat      = {};
    audiodat    = {};
    

    run = 1;
        
        
    % Import the events that are stored in the .bdf EEG file. The
    % bdf_events table also contains information about which of the 
    % audio files that were presented during the EEG experiment. 
    bdf_events = readtable(fname_events{run},'FileType','text','Delimiter','\t','TreatAsEmpty',{'N/A','n/a'});
    
    % Select the rows in the table that points to onset triggers
    % (either onsets of target speech or onset of masker speech)
%         bdf_target_masker_events = bdf_events(ismember(bdf_events.trigger_type,{'targetonset','maskeronset'}),:);
    bdf_target_events = bdf_events(strcmp(bdf_events.single_talker_two_talker,'twotalker'),:);
    
    fprintf('\n Importing data from sub-%0.3i',subid)
    fprintf('\n Preprocessing EEG data')
    
    
    % Preprocess the EEG data according to the proposed preprocessing
    % pipeline. Please inspect <preprocess_eeg> for more details. This
    % function can be found in the bottom of this script
    eegdat = preprocess_eeg(fname_bdf_file{run},bdf_events);
    
   
    fprintf('\n Preprocessing audio data')
   
    
    index = 1;
    
    audiofeat = {};
    for tr = 1 : size(eegdat.trial,2)
        
        eegtrain = double(eegdat.trial{1,tr});
        if strcmp(bdf_target_events.attend_left_right{tr},'attendleft') == 1
            env = ones(paralen,1);
        else
            env = zeros(paralen,1);
        end
        
        eegtrain_new = zeros(size(eegtrain));
    
        % We use 8-order IIR filter this time, and all the later result is
        % same
        for ch = 1:64
            x = eegtrain(ch,:);
            y = filter(b,a,x);
            eegtrain_new(ch,:) = y;
        end
        fs = 128;
        
        EEG(sb,tr,:,:) = eegtrain_new';
        ENV(sb,tr,:,:) = env';
        
    end
    

end

save(['../preprocess_data/' data1D_name],'EEG','ENV');

end



function [dat,info] = preprocess_eeg(fname,bdf_events)

% Import the .bdf files
cfg=[];
cfg.channel = 'all';
cfg.dataset = fname;
dat = ft_preprocessing(cfg);


% Define trials and segment EEG data using the events stored in the tsv
% files. Note that we here only focus on the target trials
% http://www.fieldtriptoolbox.org/example/making_your_own_trialfun_for_conditional_trial_definition/
        
bdf_target_events = bdf_events(strcmp(bdf_events.single_talker_two_talker,'twotalker'),:);
info = bdf_target_events;


cfg             = [];
cfg.trl         = [ bdf_target_events.sample-5*dat.fsample, ...                     % start of segment (in samples re 0)
                    bdf_target_events.sample+50*dat.fsample, ...                    % end of segment
                    repmat(-5*dat.fsample,size(bdf_target_events.sample,1),1), ...  % how many samples prestimulus
                    bdf_target_events.value];                                       % store the trigger values in dat.trialinfo
dat             = ft_redefinetrial(cfg,dat);


if sum(sum(isnan(cat(1,dat.trial{:}))))
    error('Warning: For some reason there are nans produced. Please make sure that the trials are not defined to be too long')
end

cfg             = [];
cfg.resamplefs  = 128;
cfg.detrend     = 'no';
cfg.method      = 'resample';
dat             = ft_resampledata(cfg, dat);


% Select a subset of electrodes
cfg = [];
cfg.channel     = 1:64;
dat             = ft_preprocessing(cfg,dat);


% We only focus on data from 6-s post stimulus onset to 43-s post
% stimulus onset
cfg             = [];
cfg.latency     = [6+1/128 50];
dat             = ft_selectdata(cfg, dat);

end



