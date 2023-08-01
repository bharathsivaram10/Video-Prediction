import os 
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from ConvLSTM import Seq2Seq
from tqdm import tqdm
import matplotlib.pyplot as plt


def test(args):
    batch_size = args.batch_size
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
  
    if args.model_name.split('_')[-1] == 'tf':
        teacher_forcing_rate = 0
    else:
        teacher_forcing_rate = None
        
    # [TODO: load testset and model]
    if args.data_source == 'moving_mnist':
        from moving_mnist import MovingMNIST
        test_dataset = MovingMNIST(root=args.data_path,
                                    is_train=False, 
                                    seq_len=args.seq_len, 
                                    horizon=args.horizon)
        
    elif args.data_source == 'sst':
        from sst import SST
        test_dataset = SST(root=args.data_path,
                            is_train=False, 
                            seq_len=args.seq_len, 
                            horizon=args.horizon)
                    
    data_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False)

    num_samples = 0
    avg_mse = 0
    avg_ssim = 0
    frame_mse = [0] * args.horizon
    frame_ssim = [0] * args.horizon
 
    ground_truth, prediction = [], []  # these are for visualization

    model = Seq2Seq(args)
    
    model.load_state_dict(torch.load((args.model_name + '.pth')))
    model.to(device)
    

    model.eval()    
    with torch.no_grad():
        for it, data in enumerate(data_loader):
            # [TODO: generate predicted frames]
            img,target = data

            # print(img.shape,target.shape)

            img,target = img.cuda(),target.cuda()

            # num_samples += [batch size]
            # frames = [ground truth frames]
            # out = [predicted frames]
            
            num_samples += batch_size
            frames = target
            out = model(img,target,teacher_forcing_rate)

            frames = frames.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            
            frames = frames[:, -args.horizon:, ...]
            out = out[:, -args.horizon:, ...]            
             
            if num_samples < 100:
                ground_truth.append(frames)
                prediction.append(out)

            for i in range(args.horizon):
                
                frame_i = frames[:, i, ...]
                out_i = out[:, i, ...]
                mse = np.square(frame_i - out_i).sum()

                ssim = 0
                for b in range(batch_size):
                    ssim += compare_ssim(frame_i[b, 0], out_i[b, 0])
                
                frame_mse[i] += mse
                frame_ssim[i] += ssim
                avg_mse += mse
                avg_ssim += ssim
                                
    ground_truth = np.concatenate(ground_truth)
    prediction = np.concatenate(prediction)

    for i in range(6):
        truth = ground_truth[6,i,0,:,:]
        pred = prediction[6,i,0,:,:]

        fig = plt.figure(figsize=(4,4))

        fig.add_subplot(1,2,1)
        plt.imshow(truth)
        fig.add_subplot(1,2,2)
        plt.imshow(pred)
        plt.show()


    avg_mse = avg_mse / (num_samples * args.horizon)
    avg_ssim = avg_ssim / (num_samples * args.horizon)
    
    print('mse: {:.4f}, ssim: {:.4f}'.format(avg_mse, avg_ssim))
    for i in range(args.horizon):
        print('frame {} - mse: {:.4f}, ssim: {:.4f}'.format(i + args.seq_len + 1, 
                                                            frame_mse[i] / num_samples, 
                                                            frame_ssim[i] / num_samples))
        
    np.savez_compressed(
        os.path.join('{}/{}_{}_prediction.npz'.format(args.result_path, args.model_name, args.data_source)),
        ground_truth=ground_truth,
        prediction=prediction)


def main():
    
    parser = argparse.ArgumentParser(description='video_prediction')
    
    parser.add_argument('--model_name', type=str, required=True, help='the name of model')
    parser.add_argument('--data_path', type=str, default='../data', help='path to the datasets')
    parser.add_argument('--data_source', type=str, required=True, help='moving_mnist | sst')
    parser.add_argument('--result_path', type=str, default='results/', help='path to the results, containing model and logs')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--seq_len', type=int, required=True, help='input frame length')
    parser.add_argument('--horizon', type=int, required=True, help='output frame length')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--h_channels', type=int, default=32)
    parser.add_argument('--in_channels', type=int, default=1)
    
    args = parser.parse_args()
    test(args)
    
    
if __name__ == "__main__":
    main()
