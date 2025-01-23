%% arrange EEG data for NN model

if 1
clear
clc

DataSaveDir = ['./PKU_process/'];
if exist(DataSaveDir,'dir')==0, mkdir(DataSaveDir); end

DataDir = ['../dataset/PKU/'];
sblist = dir(DataDir);
sblist(1:2) = [];

fs = 128; % sampling rate
Wn = [1 50]/(fs/2);
order = 8;
[b,a] = butter(order,Wn,'bandpass');



for i_sub = 1:length(sblist)
    disp(['---------Subject NO. ---------' num2str(i_sub)])
    
    EEGDataDir = [DataDir sblist(i_sub).name '/'];
    dirOutput = dir(fullfile(EEGDataDir,'*'));
    EEGDatafileNames = {dirOutput.name}';
    fileNumber = length(EEGDatafileNames);
    trnum = 0;
    trlist = EEGDatafileNames;
    tmp = zeros(1,length(trlist));
    for i = 1:length(trlist)
        if trlist{i}(1) == '.'
            tmp(i) = 1;
        end
    end
    trlist(tmp==1) = [];


    for EEGDataFileNo = 1:length(trlist)
        trnum = trnum + 1;
        EEGDataFileName = trlist{EEGDataFileNo};
        [degree, dur, order, audio_n] = ResolveFileName(EEGDataFileName);
        
        EEG = pop_loadcnt([EEGDataDir EEGDataFileName], 'dataformat', 'auto', 'memmapfile', '');
        EEG = eeg_checkset( EEG );
        % load electrodes information
        EEG = pop_chanedit(EEG, 'lookup',['standard-10-5-cap385.elp']);
        EEG = eeg_checkset( EEG );
        % delete unused electrodes
        EEG = pop_select( EEG,'nochannel',{'M1' 'M2'});
        EEG = eeg_checkset( EEG );
        EEG_raw = EEG;
    %     Fp1 = 1, FPZ = 2, C4 = 30, T8 = 32, T7 = 24, TP8 = 42, 17, 34
%         EEG = pop_interp(EEG, [17 34], 'spherical');
%         EEG = eeg_checkset( EEG );
        % re-reference, except for VEoG/HEoG/EMG/EKG
%         EEG = pop_reref( EEG, [],'exclude',[63 64 65 66] );
%         EEG = eeg_checkset( EEG );
        % downsample from 500 to 128 Hz
        EEG = pop_resample( EEG, fs);
        EEG = eeg_checkset( EEG );
        % epoch from -1 to 60s relative to trigger
        EEG = pop_epoch( EEG, {  '255'  }, [-1  60], 'newname', 'CNT file resampled epochs', 'epochinfo', 'yes','eventindices',1);
        EEG = eeg_checkset( EEG );


        if isempty(strfind(EEGDataDir,'58')) 
            
        else
            eegdata = adjust_electrodes(EEG.data')';
            EEG.data = eegdata;
        end
        EEGdata = EEG.data; % channel by time
        fs1 = EEG.srate;
        EEGdata(:,1:fs1) = [];
        EEGdata(65:66,:) = [];
        eegtrain_new = zeros(64,fs*60);
        eegtrain = double(EEGdata);
        for ch = 1:64
            x = eegtrain(ch,:);
            y = filter(b,a,x);
            eegtrain_new(ch,:) = y;
        end
        EEGdata = eegtrain_new;

        % organize attended_ear tensor
        trial_dur = 60;
        att_ears = zeros(1,trial_dur*fs1);
        n_switch = trial_dur/dur;
        for i_switch = 1:n_switch
            if mod(i_switch,2) == 1
                att_ears(((i_switch-1)*dur*fs1+1):i_switch*dur*fs1) = 1;% left=1, right=0
            end
        end
        if degree==30
            att_ears = not(att_ears);
        end
        if strcmp(order,'MF')
            att_ears = not(att_ears);
        end
        
        % 将att_ear一样的归类，先存储
        savedir = [DataSaveDir 'Sub' num2str(i_sub) '/'];
        if exist(savedir,'dir')==0, mkdir(savedir); end
        save([savedir num2str(degree) '_' num2str(trnum) '.mat'],'EEGdata','att_ears');
    end
end
end

%% arrange audio temporal envelope data for NN model,
% segmenting original 60 s into FrameLen-s windows with 0.5*FrameLen-s
% overlap, and lag-ms shift between env and eeg
if 1
clear
clc

sbnum = 12;
trnum = 36*2;
paralen = 30*128;
dataset = 'PKU';

data1D_name = [dataset '_1D.mat'];

EEG = zeros(sbnum,trnum,paralen,64);
ENV = zeros(sbnum,trnum,paralen,1);

DataDir = './PKU_process/';

sblist = dir(DataDir);
sblist(1:2) = [];


for sb = 1:length(sblist)
    sbname = sblist(sb).name;
    sbdir = [DataDir sbname];
    trlist = dir(sbdir);
    trlist(1:2) = [];

    for tr = 1:36
        disp(['  sb:' num2str(sb) '  tr:' num2str(tr)]);
        trname = trlist(tr).name;
        trdir = [sbdir filesep trname];
        load(trdir);
        eegdata = EEGdata;
        eeg_right = double(eegdata(:,att_ears==0));
        eeg_left = double(eegdata(:,att_ears==1));
        

        EEG(sb,tr*2-1,:,:) = eeg_left';
        ENV(sb,tr*2-1,:,:) = ones(3840,1);
        EEG(sb,tr*2,:,:) = eeg_right';
        ENV(sb,tr*2,:,:) = zeros(3840,1);       
    end
end

save(['../preprocess_data/' data1D_name],'EEG','ENV');
end



function [degree, dur, order, audio_n] = ResolveFileName(fileName)
    fileName = strrep(fileName,'.cnt','');
    fileName = strrep(fileName,'.mat','');
    fileName = strrep(fileName,'.txt','');
    
    splits = strsplit(fileName,'_');
    degree = str2double(splits{1});
    dur = str2double(splits{2});
    order = splits{3}(1:2);
    tmp = strrep(splits{3},'FM','');
    tmp = strrep(tmp,'MF','');
    [lia,locb] = ismember('(',tmp);
    if lia
        audio_n = str2double(tmp(1:locb-1));
    else
        audio_n = str2double(tmp);
    end
end