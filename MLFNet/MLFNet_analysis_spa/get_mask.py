import torch
# OURmodels in this work
from OURmodels.Base import Base
from OURmodels.RSC import RSC
from OURmodels.Nofre import Nofre
from OURmodels.Notem import Notem
from OURmodels.Nocat import Nocat
from OURmodels.Dft import Dft


# BASEmodels in this work

from BASEmodels.STANet import STANet
from BASEmodels.CNN import CNN
from BASEmodels.XANet import XANet
from BASEmodels.DCNN import DCNN
from BASEmodels.DenseNet import DenseNet
from BASEmodels.DARNet import DARNet
from BASEmodels.DBPNet import DBPNet


def get_model(model_name, decision_window, sbnum,device):
    model_type = ""
    device0 = device
    # OURmodels
    if model_name == "BASE":
        model = Base(device, decision_window).to(device0)
        model_type = "OURmodels"
    if model_name == "RSC":
        model = RSC(device, decision_window).to(device0)
        model_type = "OURmodels"
    if model_name == "Notem":
        model = Notem(device, decision_window).to(device0)
        model_type = "OURmodels"
    if model_name == "Nofre":
        model = Nofre(device, decision_window).to(device0)
        model_type = "OURmodels"
    if model_name == "Nocat":
        model = Nocat(device, decision_window).to(device0)
        model_type = "OURmodels"
    if model_name == "Dft":
        model = Dft(device, decision_window).to(device0)
        model_type = "OURmodels"

    # BASEmodels
        model_type = "BASEmodels"
    if model_name == "STANet":
        model = STANet(device, decision_window).to(device)
        model_type = "BASEmodels"
    if model_name == "CNN":
        model = CNN(device, decision_window).to(device)
        model_type = "BASEmodels"
    if model_name == "XANet":
        model = XANet(device, decision_window).to(device)
        model_type = "BASEmodels"
    if model_name == "DCNN":
        model = DCNN(device, decision_window).to(device)
        model_type = "BASEmodels"
    if model_name == "DenseNet":
        model = DenseNet(device, decision_window).to(device)
        model_type = "BASEmodels"
    if model_name == "DARNet":
        model = DARNet(device, decision_window).to(device)
        model_type = "BASEmodels"
    if model_name == "DBPNet":
        model = DBPNet(device, decision_window).to(device)
        model_type = "BASEmodels"



    return model, model_type