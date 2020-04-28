import time
from os.path import join

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import PIL.Image as pimg
from tqdm import tqdm

import libs.cylib as cylib

def _multiforward(model, images, target_size, out, flip=False):
    for img in images:
        img = img.cuda(non_blocking=True)
        if img.dim() == 3:
            img.unsqueeze_(0)
        logits, _ = model(img, target_size)
        logits = F.softmax(logits, dim=1)
        if flip:
            idx = [i for i in range(logits.size(3)-1, -1, -1)]
            idx = torch.LongTensor(idx).cuda(non_blocking=True)
            logits = logits.index_select(3, idx)
        out.append(logits.data)


def multiscale_forward(batch, model, target_size):
    with torch.no_grad():
        out_all = []
        _multiforward(model, batch['multiscale_imgs'], target_size, out_all)
        _multiforward(model, batch['multiscale_imgs_flip'], target_size, out_all, flip=True)

        prob_avg = out_all[0]
        for i in range(1, len(out_all)):
            prob_avg.add_(out_all[i])
        prob_avg.div_(len(out_all))

        _, pred = prob_avg.max(1)
        pred = pred.byte().cpu()
        return pred


def forward_loss(batch, model, args, return_aux=False, target_size=None):
    with torch.no_grad():
        model.send_to_gpu(batch)
        loss, output = model.forward_loss(batch, return_outputs=True)
        logits, aux_logits = output

        _, pred = logits.max(1)
        if return_aux:
            aux_preds = []
            for x in aux_logits:
                _, aux = x.data.max(1)
                aux_preds.append(aux.byte().cpu())
            return loss, pred.byte().cpu(), aux_preds
        else:
            return loss, pred.byte().cpu()


def forward(batch, model, args, return_aux=False, target_size=None):
    with torch.no_grad():
        img = batch['image'].cuda(non_blocking=True)
        logits, aux_logits = model(img, target_size)
        _, pred = logits.max(1)
        if return_aux:
            aux_preds = []
            for x in aux_logits:
                x = F.interpolate(x, target_size, mode='bilinear', align_corners=False)
                _, aux = x.data.max(1)
                aux_preds.append(aux.byte().cpu())
            return pred.byte().cpu(), aux_preds
        else:
            return pred.byte().cpu()


def evaluate_classification(epoch, model, data_loader, dataset, args, plot_data=None):
    with torch.no_grad():
        TP = 0
        N = 0
        loss_avg = 0
        for step, batch in enumerate(tqdm(data_loader)):
            loss_avg += model.forward_loss(batch)
            labels = batch['label_2d']
            _, pred = model.output.max(1)
            pred = pred.byte().cpu().long()
            TP += torch.sum(pred == labels)
            N += labels.numel()
        print(TP)
        acc = TP.item() / N * 100
        loss_avg /= len(data_loader)
        print(f'Accuracy = {acc:.2f}\%')
        print(f'Loss = {loss_avg.item():.2f}')
    return acc, loss_avg


def evaluate_semseg(epoch, model, data_loader, dataset, args, plot_data=None):
    print('\nEvaluating model...')
    conf_mat = np.zeros((model.num_classes, model.num_classes), dtype=np.uint64)
    avg_loss = 0
    num_logs = 20
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader)):
            labels = batch['labels']
            if dataset.multiscale_test:
                pred = multiscale_forward(batch, model)
            else:
                loss, pred = forward_loss(batch, model, args)
                avg_loss += loss
            pred = pred.numpy().astype(np.int32)
            true = labels.numpy().astype(np.int32)
            cylib.collect_confusion_matrix(pred.reshape(-1), true.reshape(-1), conf_mat)
    print('')
    class_names = dataset.class_names
    pixel_acc, miou, ciou, recall, precision, _ = get_eval_metrics(
            conf_mat, dataset.subsets, class_names, verbose=True)
    if dataset.category_map is not None:
        get_category_iou(conf_mat, dataset.category_map)

    avg_loss /= len(data_loader)
    avg_loss = avg_loss.item()
    miou = float(miou)
    print('Average loss = %.4f' % avg_loss)
    if plot_data is not None:
        plot_data['val_iou'].append(miou)
        plot_data['val_loss'].append(avg_loss)
        plot_data['val_conf_mat'].append(conf_mat)
        for i in range(model.num_classes):
            class_name = class_names[i]
            key = 'val_iou_' + class_name
            if key in plot_data.keys():
                plot_data[key].append(float(ciou[i]))
            else:
                plot_data[key] = [float(ciou[i])]

    return miou, avg_loss


def get_eval_metrics(conf_mat, name, class_names, verbose=True):
    num_correct = conf_mat.trace()
    num_classes = conf_mat.shape[0]
    total_size = conf_mat.sum()
    avg_pixel_acc = num_correct / total_size * 100.0
    TPFP = conf_mat.sum(0)
    TPFN = conf_mat.sum(1)
    FN = TPFN - conf_mat.diagonal()
    FP = TPFP - conf_mat.diagonal()
    class_iou = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    if verbose:
        print(name, 'IoU accuracy:')
    for i in range(num_classes):
        TP = conf_mat[i,i]
        class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
        if TPFN[i] > 0:
            class_recall[i] = (TP / TPFN[i]) * 100.0
        else:
            class_recall[i] = 0
        if TPFP[i] > 0:
            class_precision[i] = (TP / TPFP[i]) * 100.0
        else:
            class_precision[i] = 0
        if verbose:
            print('\t%s = %.2f %%' % (class_names[i], class_iou[i]))
    avg_class_iou = class_iou.mean()
    avg_class_recall = class_recall.mean()
    avg_class_precision = class_precision.mean()
    if verbose:
        print('IoU mean class accuracy -> TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
        print('mean class recall -> TP / (TP+FN) = %.2f %%' % avg_class_recall)
        print('mean class precision -> TP / (TP+FP) = %.2f %%' % avg_class_precision)
        print('pixel accuracy = %.2f %%' % avg_pixel_acc)
    return avg_pixel_acc, avg_class_iou, class_iou, avg_class_recall, \
        avg_class_precision, total_size


def get_category_iou(conf_matrix, category_map):
    print('\nCategory IoU results:')
    num_categories = len(category_map)
    category_iou = np.zeros(num_categories, dtype=np.float64)
    for i, category in enumerate(category_map):
        labels = category_map[category]

        # the number of true positive pixels for this category
        # this is the sum of all entries in the confusion matrix
        # where row and column belong to a label ID of this category
        tp = conf_matrix[labels,:][:,labels].sum()

        # the number of false negative pixels for this category
        # that is the sum of all rows of labels within this category
        # minus the number of true positive pixels
        fn = conf_matrix[labels,:].sum() - tp

        # the number of false positive pixels for this category
        # we count the column sum of all labels within this category
        # while skipping the rows of ignored labels and of labels within this category
        # notIgnoredAndNotInCategory = [l for l in args.evalLabels if not id2label[l].ignoreInEval and id2label[l].category != category]
        not_this_category = [l for l in range(len(conf_matrix)) if l not in labels]
        fp = conf_matrix[not_this_category,:][:,labels].sum()

        category_iou[i] = tp / (tp + fp + fn)
        print(category, '= %0.2f' % (category_iou[i] * 100))
    miou = category_iou.mean() * 100
    print('Category mIoU = %0.2f%%\n' % miou)
    return miou


def colorize_labels(y, class_colors):
    width = y.shape[1]
    height = y.shape[0]
    y_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for cid in range(len(class_colors)):
        cpos = np.repeat((y == cid).reshape((height, width, 1)), 3, axis=2)
        cnum = cpos.sum() // 3
        y_rgb[cpos] = np.array(class_colors[cid] * cnum, dtype=np.uint8)
    return y_rgb