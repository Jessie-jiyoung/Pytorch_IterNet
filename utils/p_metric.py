import os
import torch
import numpy as np
from torchvision import transforms as T
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, f1_score, jaccard_score, classification_report

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=1):
    with torch.no_grad():
        threshold = 0.1
        pred_mask[pred_mask >= threshold] = 1
        pred_mask[pred_mask < threshold] = 0
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            #print(clas)
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

def confusion(pred_mask, mask):
    with torch.no_grad():
        #print(pred_mask)
        #print(mask)
        #pred_mask = pred_mask.view(-1)
        pred_mask = pred_mask.flatten().cpu()
        mask = mask.flatten().cpu()
        threshold_confusion = 0.1
        #print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))
        pred_mask[pred_mask >= threshold_confusion] = 1
        pred_mask[pred_mask < threshold_confusion] = 0
        
        pred_mask=pred_mask.type(torch.int)
        mask = mask.type(torch.int)
        
        #mask=np.reshape(mask.cpu(), (2,512,512))
        #y_pred = pred_mask.view(-1)
        #y_pred = np.array(y_pred.cpu())
        #mask = mask.view(-1)
        #mask = np.array(mask.cpu())
        
        # F1 score
        confusion_m = confusion_matrix(pred_mask, mask)
        accuracy = 0
        if float(np.sum(confusion_m)) != 0:
            accuracy = float(confusion_m[0, 0] + confusion_m[1, 1]) / float(np.sum(confusion_m))
        #print("Global Accuracy: " + str(accuracy))
        specificity = 0
        if float(confusion_m[0, 0] + confusion_m[0, 1]) != 0:
            specificity = float(confusion_m[0, 0]) / float(confusion_m[0, 0] + confusion_m[0, 1])
        #print("Specificity: " + str(specificity))
        sensitivity = 0
        if float(confusion_m[1, 1] + confusion_m[1, 0]) != 0:
            sensitivity = float(confusion_m[1, 1]) / float(confusion_m[1, 1] + confusion_m[1, 0])
        #print("Sensitivity: " + str(sensitivity))
        precision = 0
        if float(confusion_m[1, 1] + confusion_m[0, 1]) != 0:
            precision = float(confusion_m[1, 1]) / float(confusion_m[1, 1] + confusion_m[0, 1])
        #print("Precision: " + str(precision))

        # Jaccard similarity index
        jaccard_index = jaccard_score(mask, pred_mask)
        #print("\nJaccard similarity score: " + str(jaccard_index))

        F1_score = f1_score(pred_mask, mask, labels=None, average='binary', sample_weight=None)
        #print(F1_score)
        #print(confusion_m)
    return accuracy, specificity, sensitivity, precision, F1_score