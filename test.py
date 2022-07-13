import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
import torch.nn.functional as F
from models.iternet import IterNet
from utils.data_loader import get_loader
import time
import random

from sklearn.metrics import confusion_matrix, f1_score, jaccard_score

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=1):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        print(pred_mask)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
                
        return np.nanmean(iou_per_class)

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def f1(pred_mask, mask):
    with torch.no_grad():
        pred_mask = pred_mask.view(-1)
        threshold_confusion = 0.1
        print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))
        y_pred = np.empty((pred_mask.shape[0]))
        for i in range(pred_mask.shape[0]):
            if pred_mask[i] >= threshold_confusion:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        
        #mask=np.reshape(mask.cpu(), (2,512,512))
        mask = mask.view(-1)
        mask = np.array(mask.cpu())
        # F1 score
        confusion_m = confusion_matrix(y_pred, mask)
        accuracy = 0
        if float(np.sum(confusion_m)) != 0:
            accuracy = float(confusion_m[0, 0] + confusion_m[1, 1]) / float(np.sum(confusion_m))
        print("Global Accuracy: " + str(accuracy))
        specificity = 0
        if float(confusion_m[0, 0] + confusion_m[0, 1]) != 0:
            specificity = float(confusion_m[0, 0]) / float(confusion_m[0, 0] + confusion_m[0, 1])
        print("Specificity: " + str(specificity))
        sensitivity = 0
        if float(confusion_m[1, 1] + confusion_m[1, 0]) != 0:
            sensitivity = float(confusion_m[1, 1]) / float(confusion_m[1, 1] + confusion_m[1, 0])
        print("Sensitivity: " + str(sensitivity))
        precision = 0
        if float(confusion_m[1, 1] + confusion_m[0, 1]) != 0:
            precision = float(confusion_m[1, 1]) / float(confusion_m[1, 1] + confusion_m[0, 1])
        print("Precision: " + str(precision))

        # Jaccard similarity index
        jaccard_index = jaccard_score(mask, y_pred)
        print("\nJaccard similarity score: " + str(jaccard_index))

        F1_score = f1_score(y_pred, mask, labels=None, average='binary', sample_weight=None)
        print(F1_score)
        print(confusion_m)
    return F1_score

def im_save(input, mask, pred_mask, idx):
    tensor2img = T.ToPILImage()
    ximg = np.array(tensor2img(input[0]))
    gimg = np.array(tensor2img(mask[0]))
    yimg = np.array(tensor2img(pred_mask[0]))
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.imshow(ximg)
    plt.subplot(1,3,2)
    plt.imshow(gimg, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(yimg, cmap='gray')
    plt.savefig('./results/{}.png'.format(idx))
    return

def test():
    netname = 'iternet'
    iou_score = 0
    accuracy = 0
    f_score = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IterNet().to(device)
    model.load_state_dict(torch.load('./models/weights/iternet_model_epoch_1000.pth'))
    model.eval()

    test_loader = get_loader(image_dir='./data/', batch_size=2, mode='test')

    since = time.time()
    for i, (x_img, y_img) in enumerate(test_loader):
        x_img = x_img.to(device)
        y_img = y_img.to(device)

        y1, y2, y3, y4 = model(x_img)
        
        f_score += f1(y4, y_img)
        iou_score += mIoU(y4, y_img)
        accuracy += pixel_accuracy(y4, y_img)

        im_save(x_img, y_img, y4, i)

    print("Val mIoU: {:.3f}..".format(iou_score/len(test_loader)),
            "Val Acc:{:.3f}..".format(accuracy/len(test_loader)),
            "Val f1 score:{:.3f}..".format(f_score/len(test_loader)),
            "Time: {:.2f}m".format((time.time()-since)/60))



if __name__ == "__main__":
    random_seed = 2022
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    test()
