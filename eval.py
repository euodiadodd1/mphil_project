from tqdm import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
import wandb

tb = SummaryWriter()

def test_model(interval, model, hyp_params, tr_loader, te_loader):

    criterion = hyp_params.criterion
    mae = hyp_params.mae
    device= hyp_params.device
    if hyp_params.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyp_params.lr, weight_decay=hyp_params.wd)
    type = hyp_params.type
    

    for epoch in range(1):
        test_loss = 0
        
        model.train()
        test_predictions = []
        test_labels = []

        
        model.eval()
        with torch.no_grad(): 
          for i, (images, labels) in enumerate(te_loader):

              
              labels = labels.to(device).float()

              if type == "decision" or type == "attention":
                images = [i.to(device).float() for i in images]
                outputs = model(*images)
                if type == "attention":
                    outputs, hidden = model(*images)
            
              else:
                images = images.to(device).float()
                outputs = model(images)

              test_predictions.append(outputs)
              test_labels.append(labels)
              loss = criterion(outputs, labels)
        


              if i % interval == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} running_loss: {:.6f}'.format(
                    epoch, i * len(images), len(te_loader.dataset),
                    100. * i / len(te_loader), loss.item(), 0))
          
          test_predictions = torch.cat(test_predictions)
          test_labels = torch.cat(test_labels)
          test_loss = criterion(test_predictions, test_labels)
          test_mae_loss = mae(test_predictions, test_labels)
          test_pcc = stats.pearsonr(test_predictions.detach().cpu().flatten(), test_labels.detach().cpu().flatten())[0]

        print('Epoch: %d | Test Loss: %.4f | Test MAE Loss: %.2f | Test PCC: %.2f' \
            %(epoch, test_loss, test_mae_loss, test_pcc))