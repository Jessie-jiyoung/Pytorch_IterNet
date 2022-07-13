import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import *
from utils.data_loader import get_loader
from models.iternet import IterNet
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import random


def get_w_from_pixel_distribution(gt, lamb=1):
    N = (gt == 0).sum()
    P = (gt == 1).sum()
    w1 = 2.*N / (lamb*P + N)
    w2 = 2.*P / (P + lamb*N)
    return w1, w2

def weighted_bce_loss(output, target, w1, w2):
    y = output * target
    loss1 = (w1 - w2) * F.binary_cross_entropy(y, target)
    loss2 = w2 * F.binary_cross_entropy(output, target)
    """
    1) target = 0
    loss1 = (w1-w2) * (0 + log(1)) = 0
    loss2 = w2 * (0 + log(1-a))
    => loss = w2 * (log(1-a))
    
    2) target = 1
    loss1 = (w1-w2) * (log(a) + 0)
    loss2 = w2 * (log(a) + 0)
    => loss = w1 * log(a)
    
    => loss = w1*y*log(a) + w2*(1-y)*log(1-a)
    """
    return loss1 + loss2

if __name__ == "__main__":

    random_seed = 2022
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    netname = 'iternet'
    num_epochs = 1000
    eps = 1e-6    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = IterNet().to(device)
    # model.load_state_dict(torch.load('./models/weights/iternet_model_final.pth'))
    # from torchsummary import summary
    # summary(model, input_size=(3, 512, 512))

    loader, val_loader= get_loader(image_dir='./data/', batch_size=2, mode='train')
    total_iter = len(loader)

    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    stime = time.time()
    total_epoch_iter = total_iter * num_epochs
    iter_count = 0

    bce = nn.BCELoss()

    losses = []
    val_losses = []
    summary = SummaryWriter()
    for epoch in range(num_epochs):
        for i, (ximg, yimg) in enumerate(loader):

            model.train()
            ximg = ximg.to(device)
            yimg = yimg.to(device)

            y1, y2, y3, y4 = model(ximg)

            w1, w2 = get_w_from_pixel_distribution(yimg)

            loss1 = weighted_bce_loss(y1, yimg, w1, w2)
            loss2 = weighted_bce_loss(y2, yimg, w1, w2)
            loss3 = weighted_bce_loss(y3, yimg, w1, w2)
            loss4 = weighted_bce_loss(y4, yimg, w1, w2)

            lambda1 = 1e-1
            lambda2 = 2e-1
            lambda3 = 3e-1
            lambda4 = 4e-1

            loss  = lambda1*loss1 + lambda2*loss2 + lambda3*loss3 + lambda4*loss4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            etime = time.time() - stime
            iter_count += 1
            rtime = etime * (total_epoch_iter - iter_count) / (iter_count + eps)
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d}, Iter: {i+1:03d}/{total_iter:03d}, ', end='')
            print(f'Loss: {losses[-1]:.5f}, ', end='')
            print(f'Elapsed: {sec2time(etime)}, Remaining: {sec2time(rtime)}')

            summary.add_scalar(f'loss/loss', loss.item(), iter_count)

            if (i+1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for j, (ximg, yimg) in enumerate(val_loader):
                        ximg = ximg.to(device)
                        yimg = yimg.to(device)

                        y1, y2, y3, y4 = model(ximg)

                        w1, w2 = get_w_from_pixel_distribution(yimg)

                        loss1 = weighted_bce_loss(y1, yimg, w1, w2)
                        loss2 = weighted_bce_loss(y2, yimg, w1, w2)
                        loss3 = weighted_bce_loss(y3, yimg, w1, w2)
                        loss4 = weighted_bce_loss(y4, yimg, w1, w2)

                        lambda1 = 1e-1
                        lambda2 = 2e-1
                        lambda3 = 3e-1
                        lambda4 = 4e-1

                        val_loss  = lambda1*loss1 + lambda2*loss2 + lambda3*loss3 + lambda4*loss4
                        val_losses.append(val_loss.item())
                        print(f'Iter: {j+1:03d}/{len(val_loader):03d}, ', end='')
                        print(f'Loss: {val_losses[-1]:.5f}, ', end='')
                        print()
            
        summary.add_image(f'image/input', ximg[0], epoch)
        summary.add_image(f'image/output4', y4[0], epoch)
        summary.add_image(f'image/output3', y3[0], epoch)
        summary.add_image(f'image/output2', y2[0], epoch)
        summary.add_image(f'image/output1', y1[0], epoch)
        summary.add_image(f'image/GT', yimg[0], epoch)
        summary.add_image(f'image/GT-output4', torch.abs(y4[0]-yimg[0]), epoch)
        
        if (epoch+1) % 200 == 0:
            lr *= 0.9
            update_lr(optimizer, lr)
            torch.save(model.state_dict(), f'./models/weights/{netname}_model_epoch_{epoch+1}.pth')

