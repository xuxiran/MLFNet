import numpy as np
import h5py
import torch
import config as cfg
import argparse
from AADdataset import AADdataset
from torch.utils.data import DataLoader
import os
import get_model
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()

# BASE and CNN3D is same but for domain generalization and direct classification
parser.add_argument('--dataset', type=str, default="pkutmp",choices=["KUL","PKU","DTU","pkutmp"])

parser.add_argument('--model', type=str, default="RSC",choices=
["CNN","STANet","XANet","DCNN","DenseNet","DARNet","DBPNet","ConcatNet",
 "BASE","RSC","Nocat","Nofre","Notem","Dft"])

parser.add_argument('--sbfold', type=int, default=3,choices=[0,1,2,3])
# 1: only evaluate the model, 0: train and evaluate the model
parser.add_argument('--only_evaluate', type=int, default=0,choices=[0,1])

parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--decision_window', type=int, default=64)


args, unknown = parser.parse_known_args()

def from_mat_to_tensor(raw_data):
    #transpose, the dimention of mat and numpy is contrary
    Transpose = np.transpose(raw_data)
    Nparray = np.array(Transpose)
    return Nparray

if __name__ == '__main__':



    #random seed
    seed_num = 2025
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed_num)

    np.random.seed(seed_num)

    decision_window = args.decision_window
    dataset_name = args.dataset
    model_name = args.model
    sbfold = args.sbfold
    batch_size = args.batch_size
    epoch_num = args.epoch


    eegname = cfg.process_data_dir + '/' + dataset_name + '_1D.mat'


    sample_rate = 128


    eegdata = h5py.File(eegname, 'r')
    data = from_mat_to_tensor(eegdata['EEG'])  # eeg data
    label = from_mat_to_tensor(eegdata['ENV'])  # 0 or 1, representing the attended direction

    sbnum = data.shape[0]


    data_new = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
    label_new = np.zeros((label.shape[0],label.shape[1],label.shape[2]))

    # this to help split the data into train, valid and test
    for sb in range(data.shape[0]):
        cnt0 = 0
        cnt1 = 0
        for tr in range(data.shape[1]):
            if label[sb,tr,0] == 0:
                label_new[sb,cnt0*2] = label[sb,tr]
                data_new[sb,cnt0*2] = data[sb,tr]
                cnt0 += 1
            else:
                label_new[sb,cnt1*2+1] = label[sb,tr]
                data_new[sb,cnt1*2+1] = data[sb,tr]
                cnt1 += 1
    trnum = data.shape[1]

    data = data_new
    label = label_new
    # get the data of all subject

    res = torch.zeros((4,1))


    #for fold, (train_vaild_ids,  test_ids) in enumerate(kfold.split(eegdata)):
    # train a model for each subject


    device_ids = sbfold
    device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")


    fold = sbfold
    # fold*trnum//4:(fold+1)*trnum//4
    # from (fold*(trnum//4) to (fold+1)*(trnum//4))
    test_ids = np.arange(fold*(trnum//4),(fold+1)*(trnum//4))
    valid_ids = (test_ids + (trnum//4))%trnum
    train_ids = np.setdiff1d(np.arange(trnum),np.concatenate((test_ids,valid_ids)))
    # print(train_ids,valid_ids,test_ids)
    traindata = data[:,train_ids]
    trainlabel = label[:,train_ids]
    validdata = data[:,valid_ids]
    validlabel = label[:,valid_ids]
    testdata = data[:,test_ids]
    testlabel = label[:,test_ids]

    traindata = traindata.reshape(traindata.shape[0]*traindata.shape[1],traindata.shape[2],traindata.shape[3])
    trainlabel = trainlabel.reshape(trainlabel.shape[0]*trainlabel.shape[1],trainlabel.shape[2])
    validdata = validdata.reshape(validdata.shape[0]*validdata.shape[1],validdata.shape[2],validdata.shape[3])
    validlabel = validlabel.reshape(validlabel.shape[0]*validlabel.shape[1],validlabel.shape[2])
    testdata = testdata.reshape(testdata.shape[0]*testdata.shape[1],testdata.shape[2],testdata.shape[3])
    testlabel = testlabel.reshape(testlabel.shape[0]*testlabel.shape[1],testlabel.shape[2])
    csp_pipe = None
    if model_name == 'DARNet' or model_name == 'DBPNet':#or model_name == 'FusionNet2' or model_name == 'FusionNet3':
        csp_params = {
            'n_components': 64,
            'transform_into': 'csp_space',
            'cov_est': 'concat',
            'norm_trace':True,
            'log':None,
            'reg':None
        }

        csp = CSP(**csp_params)
        csp_pipe = Pipeline([('CSP', csp)])


        train_valid_data = np.concatenate((traindata, validdata), axis=0)
        train_valid_label = np.concatenate((trainlabel, validlabel), axis=0)

        train_valid_data = train_valid_data.transpose(0,2,1)

        train_valid_label = train_valid_label[:,0]
        csp_pipe.fit(train_valid_data, train_valid_label)

        print("finish csp fit")


    train_trnum = traindata.shape[0]

    model,model_type = get_model.get_model(model_name, decision_window, train_trnum,device)
    # if model_type == 'BASEmodels':
    train_dataset = AADdataset(traindata, trainlabel,sample_rate,decision_window,dataset_name,csp_pipe)
    valid_dataset = AADdataset(validdata, validlabel,sample_rate,decision_window,dataset_name,csp_pipe)
    test_dataset = AADdataset(testdata, testlabel,sample_rate,decision_window,dataset_name,csp_pipe)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # else:
    #     train_dataset = AADdataset_DGtrain(traindata, trainlabel,sample_rate,decision_window,dataset_name,csp_pipe)
    #     valid_dataset = AADdataset_DGtest(validdata, validlabel,sample_rate,decision_window,dataset_name,csp_pipe)
    #     test_dataset = AADdataset_DGtest(testdata, testlabel,sample_rate,decision_window,dataset_name,csp_pipe)
    #     train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #     valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    #     device = 0
    # if model_name != 'Mixup':
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # else:
    #     train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    #     valid_loader = DataLoader(valid_dataset, batch_size=2*trnum*sbnum, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=2*trnum*sbnum, shuffle=True)



    if args.only_evaluate == 0:
        decoding_acc_final = 0
        for epoch in range(epoch_num):
            model.train(train_loader, device,epoch,epoch_num)
            result, gt = model.test(valid_loader, device)
            decoding_acc = np.sum(result == gt) / len(result)
            print(f"{model_name}_{dataset_name} valid: fold {fold} epoch {epoch} decoding accuracy: {decoding_acc}")
            savedir = './model/' + dataset_name + '/' + model_name + '/'
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            if decoding_acc > decoding_acc_final:
                decoding_acc_final = decoding_acc
                saveckpt = savedir + '/fold' + str(fold) + '.ckpt'
                torch.save(model.state_dict(), saveckpt)

    # test the model
    savedir = './model/' + dataset_name + '/' + model_name + '/'
    saveckpt = savedir + '/fold' + str(fold) + '.ckpt'

    model.load_state_dict(torch.load(saveckpt))

    result, gt = model.test(test_loader, device)
    decoding_acc = np.sum(result == gt) / len(result)
    print(f"test: fold {fold} decoding accuracy: {decoding_acc}")
    res[fold] = decoding_acc


    # save the result with the name of dataset and model_name
    savedir = './subresults/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.savetxt(savedir + dataset_name+'_'+model_name+'_'+str(sbfold)+'.csv', res.numpy(), delimiter=',')


