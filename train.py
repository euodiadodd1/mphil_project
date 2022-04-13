from tqdm import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
import wandb
import matplotlib.pyplot as plt
import numpy as np

tb = SummaryWriter()



#trainloader, testloader = feature_level_fusion()

test_loss = 0.0
eval_loss = 0.0
test_mae_loss = 0.0
eval_mae_loss = 0.0
eval_pcc = 0.0
test_pcc = 0.0


def train_model(interval, model, hyp_params, tr_loader, te_loader):
    wandb.init(project="my-test-project", entity="euodia")
    wandb.config = {
    "learning_rate": hyp_params.lr,
    "epochs": hyp_params.num_epochs,
    "batch_size": hyp_params.batch_size
    }
    criterion = hyp_params.criterion
    mae = hyp_params.mae
    device= hyp_params.device
    if hyp_params.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyp_params.lr, weight_decay=hyp_params.wd)
    type = hyp_params.type

    
    for epoch in range(1, hyp_params.num_epochs + 1):
        train_mae_loss = 0
        test_loss = 0
        
        model.train()
        train_predictions = []
        train_labels = []
        test_predictions = []
        test_labels = []
        eval_predictions = []
        eval_labels = []


        for i,(images, labels) in enumerate(tr_loader):

          labels = labels.to(device).float()
          
          optimizer.zero_grad()

          if type in ["avg_decision", "decision", "attention"]:
            images = [i.to(device).float() for i in images]
            
            if type == "attention":
                yhat, hidden = model(*images)
            else:
                yhat = model(*images).to(device)
          else:
            images = images.to(device).float()
            yhat = model(images)
            
          train_predictions.append(yhat)
          train_labels.append(labels)
          #print("shapes",labels, min_max_scaler.inverse_transform(yhat.view(labels.shape[0],-1).detach().cpu().numpy()))
          loss = criterion(yhat, labels)
          loss.backward()
          optimizer.step()

          # train_running_loss += loss.item()
          # train_mae_loss += mae(yhat.view(labels.shape[0],-1), labels).item()
          
          
          if i % interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} running_loss: {:.6f}'.format(
                    epoch, i * len(images), len(tr_loader.dataset),
                    100. * i / len(tr_loader), loss.item(),0))
        # model.eval()
        train_predictions = torch.cat(train_predictions)
        train_labels = torch.cat(train_labels)
        train_loss = criterion(train_predictions, train_labels)
        train_mae_loss = mae(train_predictions, train_labels)
        train_pcc = stats.pearsonr(train_predictions.detach().cpu().flatten(), train_labels.detach().cpu().flatten())[0]
      
        print('Epoch: %d | Loss: %.4f | MAE: %.2f | PCC: %.2f'\
            %(epoch, train_loss, train_mae_loss, train_pcc))

        
        model.eval()
        with torch.no_grad(): 
          for i, (images, labels) in enumerate(te_loader):

              
              labels = labels.to(device).float()

              if type in ["avg_decision", "decision", "attention"]:
                images = [i.to(device).float() for i in images]
                #outputs = model(*images).to(device)
                if type == "attention":
                    outputs, hidden = model(*images)
                else:
                    outputs = model(*images).to(device)
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
          print(test_predictions, test_labels)
          print('Epoch: %d | Test Loss: %.4f | Test MAE Loss: %.2f | Test PCC: %.2f' \
              %(epoch, test_loss, test_mae_loss, test_pcc))
        
          for i, (images, labels) in enumerate(tr_loader):
             
              labels = labels.to(device).float()

              if type in ["avg_decision", "decision", "attention"]:
                images = [i.to(device).float() for i in images]
                #outputs = model(*images).to(device)
                if type == "attention":
                    outputs, hidden = model(*images)
                else:
                    outputs = model(*images).to(device)
              else: 
                images = images.to(device).float()
                outputs = model(images)

              eval_predictions.append(outputs)
              eval_labels.append(labels)
              loss = criterion(outputs, labels)

              
              if i % interval == 0:
                print('Eval Epoch: {} [{}/{} ({:.0f}%)]\evLoss: {:.6f} running_loss: {:.6f}'.format(
                    epoch, i * len(images), len(tr_loader.dataset),
                    100. * i / len(tr_loader), loss.item(),0))
                
          eval_predictions = torch.cat(eval_predictions)
          eval_labels = torch.cat(eval_labels)
          eval_loss = criterion(eval_predictions, eval_labels)
          eval_mae_loss = mae(eval_predictions, eval_labels)
          eval_pcc = stats.pearsonr(eval_predictions.detach().cpu().flatten(), eval_labels.detach().cpu().flatten())[0]
          print('Epoch: %d | Eval Loss: %.4f | Eval MAE Loss: %.2f | Eval PCC: %.2f' \
            %(epoch, eval_loss, eval_mae_loss, eval_pcc))
          

          
        tb.add_scalars("loss", {"Train_loss": eval_loss/len(tr_loader),"Test_loss": test_loss/len(te_loader), "Eval_mae": eval_mae_loss/len(tr_loader), "Test_mae": test_mae_loss/len(te_loader), 
                                "Train_pcc": eval_pcc/len(tr_loader), "Test_pcc": test_pcc/len(te_loader)}, epoch)
        wandb.log({"Train_loss": eval_loss,"Val_loss": test_loss, "Train_mae": eval_mae_loss, "Val_mae": test_mae_loss, 
                                "Train_pcc": eval_pcc, "Val_pcc": test_pcc})
        wandb.watch(model)
    
    print(test_predictions.detach().cpu().flatten(), test_labels.detach().cpu().flatten())
    plot_metrics("train",train_predictions.detach().cpu(), train_labels.detach().cpu())
    plot_metrics("test",test_predictions.detach().cpu(), test_labels.detach().cpu())
    plot_all("all_test",test_predictions.detach().cpu(), test_labels.detach().cpu())
    plot_all("all_train",train_predictions.detach().cpu(), train_labels.detach().cpu())


def plot_metrics(split, preds, labels):
   for i in range(preds.shape[1]):
      plt.scatter(labels[:,i], preds[:,i], c='crimson')
      plt.plot(labels[:,i], labels[:,i])
      # p1 = max(max(preds[:,i]), max(labels[:,i]))
      # p2 = min(min(preds[:,i]), min(labels[:,i]))
      # plt.plot([p1, p2], [p1, p2], 'b-')
      plt.xlabel('True Values', fontsize=15)
      plt.ylabel('Predictions', fontsize=15)
      
      plt.savefig(split + "_" + str(i) + ".png")
      plt.show()

def plot_all(split, preds, labels):

    plt.scatter(labels, preds, c='crimson')
    plt.plot(labels, labels)
    # p1 = max(max(preds[:,i]), max(labels[:,i]))
    # p2 = min(min(preds[:,i]), min(labels[:,i]))
    # plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    
    plt.savefig(split + ".png")
    plt.show()
