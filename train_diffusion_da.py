import pickle
import os
import torch
import pandas as pd
from tqdm import tqdm
from src.eval import evaluate
#from src.utils import *
from src.parser import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from transformers import get_linear_schedule_with_warmup
from src.my_plotting import plotter
import matplotlib.pyplot as plt
import numpy as np
import wandb

import torch
import argparse
from xmlrpc.client import Boolean



device = 'cuda'

def convert_to_windows(data, n_window):
    windows = list(torch.split(data, n_window))
    for i in range (n_window-windows[-1].shape[0]):
        windows[-1] = torch.cat((windows[-1], windows[-1][-1].unsqueeze(0)))
    return torch.stack(windows)

def load_dataset(dataset, part=None):
    loader = [] 
    #folder = 'DiffusionAE/DiffusionAE/processed/' + dataset
    folder = 'DiffusionAE/processed/' + dataset

    for file in ['train', 'test', 'validation', 'labels', 'labels_validation']:
        if part is None:
            file_path=os.path.join(folder, f'{file}.npy')
            abs_file_path = os.path.abspath(file_path)
            loader.append(np.load(abs_file_path))
        else:
            loader.append(np.load(os.path.join(folder, f'{part}_{file}.npy')))
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    validation_loader = DataLoader(loader[2], batch_size=loader[2].shape[0])
    return train_loader, test_loader, validation_loader, loader[3], loader[4]

def load_model(training_mode, lr, window_size, p1, p2, dims, batch_size, noise_steps, denoise_steps,timestep_list,lambdas):
    from models2 import Autoencoder_Diffusion, TransformerBasicBottleneckScaling, TransformerBasicv2Scaling, ConditionalDiffusionTrainingNetworkwm
    scheduler=None	
    model = None
    diffusion_training_net = ConditionalDiffusionTrainingNetworkwm(dims, int(window_size), batch_size, noise_steps, denoise_steps,timestep_list, lambdas,train=True).float()
    diffusion_prediction_net = ConditionalDiffusionTrainingNetworkwm(dims, int(window_size), batch_size, noise_steps, denoise_steps, timestep_list, lambdas, train=False).float()
    optimizer = torch.optim.Adam(diffusion_training_net.parameters(), lr=float(lr))
    return model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler

CHECKPOINT_FOLDER = './anomaly-mts/a-mts/checkpoints'
def save_model(model, experiment, diffusion_training_net, optimizer, scheduler, anomaly_score, epoch, diff_loss, ae_loss):
    folder = f'{CHECKPOINT_FOLDER}/{experiment}/'
    os.makedirs(folder, exist_ok=True)
    if model:
        file_path_model = f'{folder}/model.ckpt'
        torch.save({
            'epoch': epoch,
            'ae_loss': ae_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),}, file_path_model)
    file_path_diffusion = f'{folder}/diffusion.ckpt'
    torch.save({
        'epoch': epoch,
        'diffusion_loss': diff_loss,
        'model_state_dict': diffusion_training_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),}, file_path_diffusion)
    print('saved model at ' + folder)

def load_from_checkpoint(training_mode, experiment, model, diffusion_training_net):
    folder = f'{CHECKPOINT_FOLDER}/{experiment}'
    file_path_model = f'{folder}/model.ckpt'
    file_path_diffusion = f'{folder}/diffusion.ckpt'
    model = None
    # load diffusion
    checkpoint_diffusion = torch.load(file_path_diffusion)
    diffusion_training_net.load_state_dict(checkpoint_diffusion['model_state_dict'])
    return model, diffusion_training_net

def get_diffusion_sample(diffusion_prediction_net, conditioner, k):
    if k <= 1:
        return diffusion_prediction_net(conditioner)
    else:  
        diff_samples = []
        for _ in range(k):
            diff_samples.append(diffusion_prediction_net(conditioner))
        return torch.mean(torch.stack(diff_samples), axis = 0)

def backprop(epoch, model, diffusion_training_net, diffusion_prediction_net, data, diff_lambda, optimizer, scheduler, training_mode, anomaly_score, k, training = True):
    l = nn.MSELoss(reduction = 'none')
    data_x = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_x, data_x)
    bs = diffusion_training_net.batch_size if not model else model.batch
    dataloader = DataLoader(dataset, batch_size = bs)
    w_size = diffusion_training_net.window_size
    l1s, diff_losses, ae_losses = [], [], []
    samples = []

    # weight
    weights=[0.25,0.25,0.25,0.25]

    if training:
        diffusion_training_net.train()
        for d, _ in dataloader:
            if args.model == 'Autoencoder_Diffusion':
                local_bs = d.shape[0]
                window = d.view(local_bs, -1)
            else:
                window = d
            window = window.to(device)

            window = window.reshape(-1, w_size, feats)
            loss, _ = diffusion_training_net(window)

            l1s.append(loss.item())
            optimizer.zero_grad()
            loss.backward()                                                                                                     
            optimizer.step()
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        tqdm.write(f'Epoch {epoch},\tAE = {np.mean(ae_losses)}')
        tqdm.write(f'Epoch {epoch},\tDiff = {np.mean(diff_losses)}')
        return np.mean(l1s), np.mean(ae_losses), np.mean(diff_losses)
    else:
        with torch.no_grad():
            diffusion_prediction_net.load_state_dict(diffusion_training_net.state_dict())
            diffusion_prediction_net.eval()
            diffusion_training_net.eval()
            l1s = [] # scores
            sum_losses = []
            ae_losses = []
            diff_losses = []
            recons = []
            for d, _ in dataloader:
                if args.model == 'Autoencoder_Diffusion':
                    local_bs = d.shape[0]
                    window = d.view(local_bs, -1)
                else:
                    window = d
                window = window.to(device)
                window_reshaped = window.reshape(-1, w_size, feats)
                
                
                _, reconstruced_xs = diffusion_prediction_net(window_reshaped)
                x_recon = [torch.squeeze(x,1) for x in reconstruced_xs]
                samples.extend(x_recon)
                loss = sum([l(x, window_reshaped) * weights[i] for i,x in enumerate(x_recon)])

                l1s.append(loss)
        
        return torch.cat(l1s).detach().cpu().numpy(), np.mean(sum_losses), np.mean(ae_losses), np.mean(diff_losses), torch.cat(samples).detach().cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
    parser.add_argument('--dataset', 
                        metavar='-d', 
                        type=str, 
                        required=False,
                        default='point_global',
                        help="dataset"),
    parser.add_argument('--file', 
                        metavar='-f', 
                        type=str, 
                        required=False,
                        default=None,
                        help="None")
    parser.add_argument('--model', 
                        metavar='-m', 
                        type=str, 
                        required=False,
                        default='Autoencoder_Diffusion',
                        help="model name"),
    parser.add_argument('--training', 
                        metavar='-t', 
                        type=str, 
                        required=False,
                        default='diffusion',
                        help="model to train"),
    parser.add_argument('--anomaly_score', 
                        metavar='-t', 
                        type=str, 
                        required=False,
                        default=None,
                        help="anomaly score"),
    parser.add_argument('--lr', 
                        metavar='-lr', 
                        type=str, 
                        required=False,
                        default='1e-4',
                        help="lerning rate"),
    parser.add_argument('--window_size', 
                        metavar='-ws', 
                        type=str, 
                        required=False,
                        default='10',
                        help="window size"),
    parser.add_argument('--p1', 
                        metavar='-p1', 
                        type=float, 
                        required=False,
                        default='1',
                        help="p1"),
    parser.add_argument('--p2', 
                        metavar='-p2', 
                        type=float, 
                        required=False,
                        default='1',
                        help="p2"),
    parser.add_argument('--k', 
                        metavar='-k', 
                        type=int, 
                        required=False,
                        default='1',
                        help="number of diff samples"),					
    parser.add_argument('--v', 
                        metavar='-v', 
                        type=bool, 
                        required=False,
                        default=False,
                        help="verbose"),
    parser.add_argument('--batch_size', 
                        metavar='-t', 
                        type=int, 
                        required=False,
                        default=128,
                        help="batch_size"),
    parser.add_argument('--diff_lambda', 
                        metavar='-t', 
                        type=float, 
                        required=False,
                        default=0.1,
                        help="diff_lambda"),
    parser.add_argument('--noise_steps', 
                        metavar='-t', 
                        type=int, 
                        required=False,
                        default=100,
                        help="noise_steps"),
    parser.add_argument('--denoise_steps', 
                        metavar='-t', 
                        type=int, 
                        required=False,
                        default=10,
                        help="denoise_steps"),
    parser.add_argument('--group', 
                        metavar='-t', 
                        type=str, 
                        required=False,
                        default='search_smd',
                        help="wandb group"),
    parser.add_argument('--test_only', 
                        metavar='-t', 
                        type=bool, 
                        required=False,
                        default=False,
                        help="train new model or not"),
    parser.add_argument('--id', 
                        metavar='-t', 
                        type=int, 
                        required=False,
                        default=0,
                        help="experiment id for multiple runs"),
    parser.add_argument('--timestep_list', 
                        metavar='-ts', 
                        type=int, 
                        nargs='+', 
                        required=False,
                        default=[50, 100, 150, 200],
                        help="list of timestep_list for multi-level reconstruction"),
    parser.add_argument('--lambdas', 
                        metavar='-l', 
                        type=float, 
                        nargs='+', 
                        required=False,
                        default=[0.1, 0.2, 0.3, 0.4],
                        help="list of lambdas for multi-level reconstruction"),
    args = parser.parse_args()

    config = {
    "dataset": args.dataset,
    "file": args.file,
    "training_mode": args.training, 
    "model": args.model,
    "learning_rate": float(args.lr),
    "window_size": int(args.window_size),
    "lambda": args.diff_lambda,
    "noise_steps":args.noise_steps,
    "batch_size": args.batch_size,
    "timestep_list": args.timestep_list,
    "lambdas": args.lambdas
    }

    print(config)


    anomaly_scores = [args.anomaly_score]

    if args.training == 'diffusion':
        experiment = 'diffv4'
    elif args.model == 'Autoencoder_Diffusion':
        experiment = 'autoencoder_both'
    elif args.model == 'TransformerBasicBottleneckScaling':
        experiment = 'tr_bn_diffv4'
    else:
        experiment = 'tr_basic_diffv4'
    
    experiment += f'_{args.dataset}_{args.noise_steps}-{args.denoise_steps}_{args.diff_lambda}_{args.lr}_{args.batch_size}_{args.window_size}'


    wandb.init(project="anomaly-mts", config=config, group=args.group)
    wandb.run.name = experiment
    
    dataset_name = args.dataset
    part = None if not args.file else args.file
    training_mode = 'both' if not args.training else args.training
    anomaly_score = None if not args.anomaly_score else args.anomaly_score
    window_size = int(args.window_size)
    synthetic_datasets = ['point_global', 'point_contextual', 'pattern_shapelet', 'pattern_seasonal', 'pattern_trend', 'all_types', 'pattern_trendv2']
    
    train_loader, test_loader, validation_loader, labels, validation_labels = load_dataset(dataset_name, part)
    model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler = \
                        load_model(training_mode ,args.lr, args.window_size, args.p1, args.p2, labels.shape[1], args.batch_size, args.noise_steps, args.denoise_steps, args.timestep_list, args.lambdas)
    if model:
        model = model.to(device)
            
    diffusion_training_net = diffusion_training_net.to(device)
    diffusion_prediction_net = diffusion_prediction_net.to(device)

    trainD, testD, validationD = next(iter(train_loader)), next(iter(test_loader)), next(iter(validation_loader))
    trainO, testO, validationO = trainD, testD, validationD
    if args.v:
        print(f'\ntrainD.shape: {trainD.shape}')
        print(f'testD.shape: {testD.shape}')
        print(f'validationD.shape: {validationD.shape}')
        print(f'labels.shape: {labels.shape}')
    
    feats=labels.shape[1]    

    trainD, testD, validationD = convert_to_windows(trainD, window_size), convert_to_windows(testD, window_size), convert_to_windows(validationD, window_size)

    num_epochs = 100   


    epoch = -1

    e = epoch + 1; start = time()
    max_roc_scores = [[0, 0, 0]] * 6
    max_f1_scores = [[0, 0, 0]] * 6
    roc_scores = []
    f1_scores = []
    f1_max = 0
    roc_max = 0
    validation_thresh = 0

    if not args.test_only:
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            train_loss, ae_loss, diff_loss = backprop(e, model, diffusion_training_net, diffusion_prediction_net, trainD, args.diff_lambda, optimizer, scheduler, training_mode, anomaly_score, args.k)
            wandb.log({
                'sum_loss_train': train_loss,
                'ae_loss_train': ae_loss,
                'diff_loss_train': diff_loss,
                'epoch': e
            }, step=e)
            if train_loss < 0.15:
                loss0, _, _, val_loss, samples = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
                wandb.log({'val_loss': loss0.mean(), 'epoch': e}, step=e)
                loss0 = loss0.reshape(-1,feats)
                lossFinal = np.mean(np.array(loss0), axis=1)
                labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0
                result, fprs, tprs = evaluate(lossFinal, labelsFinal)
                result_roc = result["ROC/AUC"]
                result_f1 = result["f1"]
                wandb.log({'roc': result_roc, 'f1': result_f1}, step=e)
                if result_f1 > f1_max:
                    save_model(None, experiment, diffusion_prediction_net, optimizer, None, -1, e, train_loss, None)
                    f1_max = result_f1
                    validation_thresh = result['threshold']
                    wandb.run.summary["best_f1"] = f1_max
                    wandb.run.summary["roc_for_best_f1"] = result_roc
                    wandb.run.summary["best_f1_epoch"] = e
                    wandb.run.summary["validation_thresh"] = validation_thresh
                if result_roc > roc_max:
                    roc_max = result_roc 
                    wandb.run.summary["f1_for_best_roc"] = result_f1
                    wandb.run.summary["best_roc"] = roc_max
                    wandb.run.summary["best_roc_epoch"] = e
                wandb.log({'roc': result_roc, 'f1': result_f1}, step=e)
                if e % 100 == 0:
                    for dim in range(0, feats):
                        plotter(f'{experiment}_VAL', args.dataset, validationD.reshape(-1, feats), lossFinal, labelsFinal, result, None, samples.reshape(-1, feats), None, dim=dim, plot_test=True, epoch=e)
                if args.v:
                    print(f"testing loss #{e}: {loss0.mean()}")
                    print(f"final ROC #{e}: {result_roc}")
                    print(f"F1 #{e}: {result_f1}")

    # TEST ON TEST SET
    #load model from checkpoint
    model, diffusion_training_net, diffusion_prediction_net, optimizer, scheduler = \
                        load_model(training_mode ,args.lr, args.window_size, args.p1, args.p2, labels.shape[1], args.batch_size, args.noise_steps, args.denoise_steps,args.timestep_list,args.lambdas)
    model, diffusion_training_net = load_from_checkpoint(training_mode, experiment, model, diffusion_training_net)
    
    if model:
        model = model.to(device)
            
    diffusion_training_net = diffusion_training_net.to(device)
    diffusion_prediction_net = diffusion_prediction_net.to(device)
    # pass test set through the model
    if model:
        if args.test_only:
        #test again on val for double check + get best thresh on validation set to use for test
            loss0, val_loss, ae_loss_val, diff_loss_val, samples, recons = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
            loss0 = loss0.reshape(-1,feats)

            lossFinal = np.mean(np.array(loss0), axis=1)
            # lossFinal = np.max(np.array(loss0), axis=1)
            labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0

            result, fprs, tprs = evaluate(lossFinal, labelsFinal)
            validation_thresh = result['threshold']
            result_roc = result["ROC/AUC"]
            result_f1 = result["f1"]
            wandb.run.summary["f1_val"] = result_f1
            wandb.run.summary["roc_val"] = result_roc
            wandb.run.summary["f1_pa_val"] = result['f1_max'] 
            wandb.run.summary["roc_pa_val"] = result['roc_max']
            wandb.run.summary["val_loss"] = val_loss
            wandb.run.summary["ae_loss_val"] = ae_loss_val
            wandb.run.summary["diff_loss_val"] = diff_loss_val

        loss0, test_loss, ae_loss_test, diff_loss_test, samples, recons = backprop(e, model, diffusion_training_net, diffusion_prediction_net, testD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
        loss0 = loss0.reshape(-1,feats)

        lossFinal = np.mean(np.array(loss0), axis=1)
        np.save(f'../../{args.dataset}_{args.anomaly_score}_score_scores_swat.npy', lossFinal)
        np.save(f'../../{args.dataset}_{args.anomaly_score}_score_recons_swat.npy', samples)

        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0

        result = evaluate(lossFinal, labelsFinal, validation_thresh=validation_thresh)
        result_roc = result["ROC/AUC"]
        result_f1 = result["f1"]
        wandb.run.summary["f1_test"] = result_f1
        wandb.run.summary["roc_test"] = result_roc
        wandb.run.summary["f1_pa_test"] = result['f1_max'] 
        #wandb.run.summary["roc_pa_test"] = result['roc_max']
        wandb.run.summary["test_loss"] = test_loss
        wandb.run.summary["ae_loss_test"] = ae_loss_test
        wandb.run.summary["diff_loss_test"] = diff_loss_test
        wandb.run.summary["validation_thresh"] = validation_thresh

        
    else: 
        if args.test_only:
            loss0, _, _, val_loss, samples = backprop(e, model, diffusion_training_net, diffusion_prediction_net, validationD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
            loss0 = loss0.reshape(-1,feats)

            lossFinal = np.mean(np.array(loss0), axis=1)
            labelsFinal = (np.sum(validation_labels, axis=1) >= 1) + 0

            result, fprs, tprs = evaluate(lossFinal, labelsFinal)
            result_roc = result["ROC/AUC"]
            result_f1 = result["f1"]
            validation_thresh = result['threshold']
            wandb.run.summary["f1_val"] = result_f1
            wandb.run.summary["roc_val"] = result_roc
            wandb.run.summary["f1_pa_val"] = result['f1_max'] 
            #wandb.run.summary["roc_pa_val"] = result['roc_max']
            wandb.run.summary["val_loss"] = val_loss
            wandb.run.summary["validation_thresh"] = validation_thresh
        loss0, _, _, test_loss, samples = backprop(e, model, diffusion_training_net, diffusion_prediction_net, testD, args.diff_lambda, optimizer, scheduler, training_mode, args.anomaly_score, args.k, training=False)
        loss0 = loss0.reshape(-1,feats)

        lossFinal = np.mean(np.array(loss0), axis=1)
        np.save(f'./dataset/{args.dataset}_mrdiff_scores.npy', lossFinal)
        np.save(f'./dataset/{args.dataset}_mrdiff_recons.npy', samples)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0

        # training
        #result, _, _ = evaluate(lossFinal, labelsFinal, validation_thresh=validation_thresh)

        # testing
        result = evaluate(lossFinal, labelsFinal, validation_thresh=validation_thresh)

        result_roc = result["ROC/AUC"]
        result_f1 = result["f1"]
        wandb.run.summary["f1_test"] = result_f1
        wandb.run.summary["roc_test" ] = result_roc
        wandb.run.summary["f1_pa_test"] = result['f1_max'] 
        wandb.run.summary["roc_pa_test"] = result['roc_max']
        wandb.run.summary["test_loss"] = test_loss
    
    wandb.finish()  
