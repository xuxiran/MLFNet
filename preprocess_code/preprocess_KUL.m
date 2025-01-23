c3;
format long
% add your eeglab address,or you can add the path to dir
% addpath(genpath('D:\eeglab_current\eeglab2022.0'));
% produce 2*2*2=8 data
data_types = {'1D','2D'};
paralen = 60*128;
sbnum = 16;
trnum = 8;

dataset = 'KUL';

data1D_name = [dataset '_1D.mat'];

EEG = zeros(sbnum,trnum,6*paralen,64);
ENV = zeros(sbnum,trnum,6*paralen,1);

rawdir=['../dataset/' dataset];

fs = 128; % sampling rate
Wn = [1 50]/(fs/2);
order = 8;
[b,a] = butter(order,Wn,'bandpass');



for sb = 1:sbnum
    load([rawdir filesep 'S' num2str(sb) '.mat']);

    for tr = 1:trnum
        disp(['preprocess_data      subject:' num2str(sb) '   trial:' num2str(tr)]);
        trial = trials{tr};%read the trialnum's trial

        tmp = double(trial.RawData.EegData);

        eegtrain = tmp(1:6*paralen,:)';
        eegtrain_new = zeros(size(eegtrain));
        
        % We use 8-order IIR filter this time, and all the later result is
        % same
        for ch = 1:64
            x = eegtrain(ch,:);
            y = filter(b,a,x);
            eegtrain_new(ch,:) = y;
        end
        fs = 128;
        EEG_trial = pop_importdata('dataformat','array','nbchan',0,'data','eegtrain_new','srate',fs,'pnts',0,'xmin',0);


        % give label
        if trial.attended_ear=='L'
            labeltrain = ones(6*paralen,1);
        else
            labeltrain = zeros(6*paralen,1);
        end

        EEG(sb,tr,:,:) = eegtrain';
        ENV(sb,tr,:,:) = labeltrain';
    end

end

save(['../preprocess_data/' data1D_name],'EEG','ENV');









