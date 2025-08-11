# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
from collections import defaultdict
import torch
import torch.nn as nn


def get_accuracy(output, label):
    _, prediction = torch.max(output, 1)    # argmax
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)
    return accuracy


def get_prediction(output, label):
    prob = nn.functional.softmax(output, dim=1)[:, 1]
    prob = prob.view(prob.size(0), 1)
    label = label.view(label.size(0), 1)
    datas = torch.cat((prob, label.float()), dim=1)
    return datas


def calculate_metrics_for_train(label, output):
    """计算训练时的指标"""
    # Ensure inputs are 1D
    label = label.squeeze()
    output = output.squeeze()
    
    # Handle different input formats
    if output.dim() > 1 and output.size(1) == 2:
        # If output is 2D (batch_size, num_classes), extract probability for class 1
        prob = torch.softmax(output, dim=1)[:, 1]
        # For accuracy calculation, we need the original output
        output_for_acc = output
    else:
        # If output is already 1D probability
        prob = output
        # For accuracy calculation, we need to convert probability to predictions
        output_for_acc = (prob > 0.5).float()
    
    # Accuracy
    if output_for_acc.dim() > 1:
        _, prediction = torch.max(output_for_acc, 1)
    else:
        prediction = (output_for_acc > 0.5).long()
    
    correct = (prediction == label).sum().item()
    accuracy = correct / label.size(0)
    
    # Average Precision
    y_true = label.cpu().detach().numpy()
    y_pred = prob.cpu().detach().numpy()
    
    # Ensure y_true and y_pred are 1D arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    ap = metrics.average_precision_score(y_true, y_pred)
    
    # AUC and EER
    try:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    except:
        # for the case when we only have one sample
        return None, None, accuracy, ap
    
    if np.isnan(fpr[0]) or np.isnan(tpr[0]):
        # for the case when all the samples within a batch is fake/real
        auc, eer = None, None
    else:
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return auc, eer, accuracy, ap


def compute_class_losses(pred, label, loss_func):
    """计算分类别的损失"""
    # Overall loss
    loss = loss_func(pred, label)
    
    # Create masks for real and fake classes
    mask_real = label == 0  # Boolean tensor
    mask_fake = label == 1  # Boolean tensor
    
    # Compute loss for real class
    if mask_real.sum() > 0:
        pred_real = pred[mask_real]
        label_real = label[mask_real]
        loss_real = loss_func(pred_real, label_real)
    else:
        # No real samples in batch
        loss_real = torch.tensor(0.0, device=pred.device)
    
    # Compute loss for fake class
    if mask_fake.sum() > 0:
        pred_fake = pred[mask_fake]
        label_fake = label[mask_fake]
        loss_fake = loss_func(pred_fake, label_fake)
    else:
        # No fake samples in batch
        loss_fake = torch.tensor(0.0, device=pred.device)
    
    # Return a dictionary with all losses
    loss_dict = {
        'overall': loss,
        'real_loss': loss_real,
        'fake_loss': loss_fake,
    }
    
    return loss_dict


# ------------ compute average metrics of batches ---------------------
class Metrics_batch():
    def __init__(self):
        self.tprs = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.aucs = []
        self.eers = []
        self.aps = []
        self.correct = 0
        self.total = 0
        self.losses = []

    def update(self, label, output):
        acc = self._update_acc(label, output)
        if output.size(1) == 2:
            prob = torch.softmax(output, dim=1)[:, 1]
        else:
            prob = output
        
        auc, eer = self._update_auc(label, prob)
        ap = self._update_ap(label, prob)
        return acc, auc, eer, ap

    def _update_auc(self, lab, prob):
        fpr, tpr, thresholds = metrics.roc_curve(lab.squeeze().cpu().numpy(),
                                                prob.squeeze().cpu().numpy(),
                                                pos_label=1)
        if np.isnan(fpr[0]) or np.isnan(tpr[0]):
            return -1, -1
        
        auc = metrics.auc(fpr, tpr)
        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(auc)
        
        # EER
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        self.eers.append(eer)
        return auc, eer

    def _update_acc(self, lab, output):
        _, prediction = torch.max(output, 1)    # argmax
        correct = (prediction == lab).sum().item()
        accuracy = correct / prediction.size(0)
        self.correct = self.correct + correct
        self.total = self.total + lab.size(0)
        return accuracy

    def _update_ap(self, label, prob):
        y_true = label.cpu().detach().numpy()
        y_pred = prob.cpu().detach().numpy()
        ap = metrics.average_precision_score(y_true, y_pred)
        self.aps.append(ap)
        return np.mean(ap)

    def get_mean_metrics(self):
        mean_acc, std_acc = self.correct/self.total, 0
        mean_auc, std_auc = self._mean_auc()
        mean_err, std_err = np.mean(self.eers), np.std(self.eers)
        mean_ap, std_ap = np.mean(self.aps), np.std(self.aps)
        return {'acc': mean_acc, 'auc': mean_auc, 'eer': mean_err, 'ap': mean_ap}

    def _mean_auc(self):
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        return mean_auc, std_auc

    def clear(self):
        self.tprs.clear()
        self.aucs.clear()
        self.correct = 0
        self.total = 0
        self.eers.clear()
        self.aps.clear()
        self.losses.clear()


# ------------ compute average metrics of all data ---------------------
class Metrics_all():
    def __init__(self):
        self.probs = []
        self.labels = []
        self.correct = 0
        self.total = 0

    def store(self, label, output):
        prob = torch.softmax(output, dim=1)[:, 1]
        _, prediction = torch.max(output, 1)    # argmax
        correct = (prediction == label).sum().item()
        self.correct += correct
        self.total += label.size(0)
        self.labels.append(label.squeeze().cpu().numpy())
        self.probs.append(prob.squeeze().cpu().numpy())

    def get_metrics(self):
        y_pred = np.concatenate(self.probs)
        y_true = np.concatenate(self.labels)
        
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        
        # acc
        acc = self.correct / self.total
        
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def clear(self):
        self.probs.clear()
        self.labels.clear()
        self.correct = 0
        self.total = 0


# only used to record a series of scalar value
class Recorder:
    def __init__(self):
        self.sum = 0
        self.num = 0

    def update(self, item, num=1):
        if item is not None:
            self.sum += item * num
            self.num += num

    def average(self):
        if self.num == 0:
            return None
        return self.sum/self.num

    def clear(self):
        self.sum = 0
        self.num = 0