import torch
import argparse
from eval import test_model
from load_data import *
from models.multmodel import MULTModel
from models.resnet50 import ResNet50
from models.ensemble import AvgEnsemble, Ensemble
from train import train_model
import torch.nn as  nn
import random
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--type', type=str, default='avg_decision',
                    help='name of fusion strategy (Decision, Feature, Attention, None)')
parser.add_argument('--batch_size', type=int, default=25, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--section', type=str, default="lego",
                    help= 'part of the dataset')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--wd', type=float, default=0,
                    help='weight decay (default: 0)')
parser.add_argument('--num_epochs', type=int, default=12,
                    help='number of epochs (default: 40)')
parser.add_argument('--optimizer', type=str, default="Adam",
                    help= 'part of the dataset')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

torch.manual_seed(1111)
random.seed(1)
np.random.seed(0)


args = parser.parse_args()

valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

modality = args.type
BATCH_SIZE = args.batch_size
section = args.section

hyp_params = args
hyp_params.lr = args.lr
hyp_params.num_epochs = args.num_epochs
hyp_params.criterion = nn.MSELoss()
hyp_params.mae = nn.L1Loss()
hyp_params.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print("Running on" + section)

if modality in ["avg_decision", "decision", "feature","attention"]:
    trainloader, testloader, dims = fused_data(modality, BATCH_SIZE, section)
    model = ResNet50(5, channels=1).to(device)
    if modality in ["decision", "avg_decision"]:
        gaze_model = ResNet50(5, channels=1)
        pose_model = ResNet50(5, channels=1)
        au_model = ResNet50(5, channels=1)
        model = Ensemble(gaze_model, pose_model, au_model).to(device)
        if modality == "avg_decision":
            model = AvgEnsemble(gaze_model, pose_model, au_model).to(device)  
    elif modality == "attention":
        hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = [x for (x,y) in dims]
        hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = [y for (x,y) in dims]
        hyp_params.layers = args.nlevels
        model = MULTModel(hyp_params).to(device)
 
else:
    print(section)
    trainloader, testloader = modality_data(modality, BATCH_SIZE, section)
    model = ResNet50(5, channels=1).to(device)

train_model(1, model, hyp_params, trainloader, testloader)
# model.load_state_dict(torch.load("saved_models/animals_au_resnet50", map_location=torch.device('cpu')))
# test_model(1, model, hyp_params, trainloader, testloader)

PATH = "/content/drive/MyDrive/saved_models/"+ section + "_" + modality +"_resnet50"
#torch.save(model.state_dict(), PATH)
