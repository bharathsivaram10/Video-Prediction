import os 
import json
import time
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from ConvLSTM import Seq2Seq
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(args):
    
    gpu = args.gpu_id if torch.cuda.is_available() else None
    print(f'Device - GPU:{args.gpu_id}')
    
    if args.data_source == 'moving_mnist':
        from moving_mnist import MovingMNIST
        train_dataset = MovingMNIST(root=args.data_path,
                                    is_train=True, 
                                    seq_len=args.seq_len, 
                                    horizon=args.horizon)
        
    elif args.data_source == 'sst':
        from sst import SST
        train_dataset = SST(root=args.data_path,
                            is_train=True, 
                            seq_len=args.seq_len, 
                            horizon=args.horizon)
                    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True)
    
    # [TODO: define your model, optimizer, loss function]
    h_channels = []
    for i in range(args.num_layers):
        h_channels.append(64)
    device = torch.device('cuda')

    writer = SummaryWriter()  # initialize tensorboard writer
    
    if args.use_teacher_forcing:
        teacher_forcing_rate = 1.0
    else:
        teacher_forcing_rate = None

    model = Seq2Seq(args)
    model.to(device)

    step = 0.01
    #print(teacher_forcing_rate)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    itera = 0

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        losses = 0
        for i, data in enumerate(train_loader):
            
            # [TODO: train the model with a batch]
            img,target = data

            # print(img.shape)
            # print(target.shape)

            if device == torch.device('cuda'):
                img,target = img.cuda(), target.cuda()

            optimizer.zero_grad()

            output = model(img,target,teacher_forcing_rate)

            # print(output.shape,target.shape)

            loss = criterion(output,target)
            losses += loss.item()
            loss.backward()
            optimizer.step()   
            
            # [TODO: if using teacher forcing, update teacher forcing rate]
            writer.add_scalar('SST Loss Iter',loss.item(), itera)
        
            itera += 1

        if teacher_forcing_rate is not None and (teacher_forcing_rate-step) > 0:
            teacher_forcing_rate -= step

        if teacher_forcing_rate is not None:
            print(teacher_forcing_rate)

        print(losses/len(train_loader))
        # [TODO: use tensorboard to visualize training loss]
        writer.add_scalar('SST Loss Epochs',losses/len(train_loader), epoch)
        torch.save(model.state_dict(), os.path.join('outputs/', 'ep_' + str(epoch) +'.pth'))
    
    
def main():
    
    parser = argparse.ArgumentParser(description='video_prediction')
    
    # load data from file
    parser.add_argument('--data_path', type=str, default='../data', help='path to the datasets')
    parser.add_argument('--data_source', type=str, required=True, help='moving_mnist | sst')
    #parser.add_argument('--model_name', type=str, required=True, help='name of the saved model')
    parser.add_argument('--seq_len', type=int, required=True, help='input frame length')
    parser.add_argument('--horizon', type=int, required=True, help='output frame length')
    parser.add_argument('--use_teacher_forcing', action='store_true', help='if using teacher forcing, default is False')
    parser.add_argument('--result_path', type=str, default='../results', help='path to the results, containing model and logs')

    # training parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=5)    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--h_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=1)
    args = parser.parse_args()
    
    train(args)
    
    
if __name__ == "__main__":
    main()
