import torch
import torch.nn.functional as F


def segmentation_loss(logits, aux_logits, batch, aux_loss_weight, ignore_index=-1,
                      equal_level_weights=False):
    if len(aux_logits) > 0:
        main_wgt = 1 - aux_loss_weight
    else:
        main_wgt = 1

    num_classes = logits.shape[1]
    labels = batch['labels']
    log_softmax = F.log_softmax(logits, dim=1)
    loss_val = F.nll_loss(log_softmax, labels, ignore_index=ignore_index)
    loss = main_wgt * loss_val
    separated_losses = [loss_val.detach()]
    # if self.args.class_balancing:
    #   loss = main_wgt * F.nll_loss(log_softmax, labels, weight=self.dataset.class_weights,
    #                     ignore_index=self.dataset.ignore_id)
    # else:
    #   loss = main_wgt * F.nll_loss(log_softmax, labels, ignore_index=self.dataset.ignore_id)

    if len(aux_logits) > 0:
        aux_targets = batch['aux_targets']
        aux_valid_masks = batch['aux_valid_mask']
        if equal_level_weights:
            aux_wgt = aux_loss_weight / len(aux_logits)
        else:
            aux_loss = []
        for i in range(len(aux_logits)):
            target_dist = aux_targets[i].reshape(-1, num_classes).cuda(non_blocking=True)
            valid_mask = aux_valid_masks[i].reshape(-1, 1).cuda(non_blocking=True)
            logits_1d = aux_logits[i].permute(0,2,3,1).contiguous().reshape(-1, num_classes)
            if equal_level_weights:
                loss_val = softmax_cross_entropy_with_ignore(logits_1d, target_dist, valid_mask)
                loss += aux_wgt * loss_val
                separated_losses.append(loss_val.detach())
            else:
                level_loss = softmax_cross_entropy_with_ignore(
                    logits_1d, target_dist, valid_mask, average=False)
                aux_loss.append(level_loss)
        if not equal_level_weights:
            loss += aux_loss_weight * torch.mean(torch.cat(aux_loss, dim=0))
    return loss, separated_losses