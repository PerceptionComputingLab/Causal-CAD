import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

import functions as funcs


class adi_net_optimization_objective(nn.Module):
    def __init__(self,num_queries, max_branch, eta, num_classes=1, eos_coef=0.2, matcher=funcs.HungarianMatcher(), device=None):
        super().__init__()
        self.num_queries = num_queries
        self.max_branch = max_branch
        self.num_classes = num_classes
        self.eta = eta
        self.matcher = matcher
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.cls_loss = nn.CrossEntropyLoss()
        self.device = device

    def loss_labels(self, outputs, targets, indices):

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        empty_batch = False

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        if target_classes_o.numel() != 0:
            target_classes_o = target_classes_o.to(dtype=torch.long)
            target_classes[idx] = target_classes_o
        else:
            empty_batch = True

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight.to(src_logits.device))
        return loss_ce, empty_batch

    def loss_boxes(self, outputs, targets, indices, num_boxes):

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(funcs.generalized_box_iou(funcs.box_cxcywh_to_xyxy(src_boxes),
                                                             funcs.box_cxcywh_to_xyxy(target_boxes)))
        return loss_bbox.sum() / num_boxes + loss_giou.sum() / num_boxes

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        batch_idx = batch_idx.to(dtype=torch.long)
        src_idx = src_idx.to(dtype=torch.long)
        return batch_idx, src_idx

    def classification_label_mapping(self, cls_list, pair_idx_list):

        device_tmp = self.device
        batch_size = len(cls_list)

        out_tensor = torch.zeros((batch_size, self.num_queries),
                                 dtype=cls_list[0].dtype,
                                 device=device_tmp)

        for i in range(batch_size):
            idx2, idx1 = pair_idx_list[i]
            if idx2.numel() > 0:
                out_tensor[i, idx2] = cls_list[i][idx1].to(device_tmp)

        return out_tensor

    def reporting_and_data_system_dix(self, x):
        batch_size, length = x.shape
        result = []

        for i in range(batch_size):
            row = x[i]
            if torch.all(row == 0):
                result.append(torch.arange(length, device=x.device))
            else:
                max_val = torch.max(row)
                result.append(torch.nonzero(row == max_val, as_tuple=True)[0])

        return result

    def overall_amount_of_plaque_idx(self, x):
        batch_size, seq_len = x.shape
        all_indices = torch.arange(seq_len, device=x.device)
        result = []

        for i in range(batch_size):
            row = x[i]
            nonzero_idx = torch.nonzero(row, as_tuple=True)[0]
            if nonzero_idx.numel() == 0:
                result.append(all_indices.clone())
            else:
                result.append(nonzero_idx)

        return result

    def patient_level_index(self, sten_targets, plq_targets):

        max_idx_sten = torch.max(sten_targets, dim=1).values
        batch_size = max_idx_sten.size(0) // self.max_branch
        sten_out = max_idx_sten.view(batch_size, self.max_branch)

        max_idx_plq = torch.max(plq_targets, dim=1).values
        batch_size = max_idx_plq.size(0) // self.max_branch
        plq_out = max_idx_plq.view(batch_size, self.max_branch)

        rds_idx = self.reporting_and_data_system_dix(sten_out)
        oap_idx = self.overall_amount_of_plaque_idx(plq_out)

        return rds_idx, oap_idx

    def patient_level_prediction_loss(self, pred, target, idx_list):

        batch_size = len(idx_list)
        branch_num = pred.size(0) // batch_size
        seq_len = pred.size(1)
        num_class = pred.size(2)

        target_r = target.view(batch_size, branch_num, seq_len)
        pred_r = pred.view(batch_size, branch_num, seq_len, num_class)

        losses = []

        for b in range(batch_size):
            idx = idx_list[b]
            if idx.numel() == 0:
                continue

            target_sel = target_r[b, idx, :]
            pred_sel = pred_r[b, idx, :, :]

            target_flat = target_sel.reshape(-1)
            pred_flat = pred_sel.reshape(-1, num_class)
            loss = self.cls_loss(pred_flat, target_flat)
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=pred.device)

        final_loss = torch.stack(losses).mean()
        return final_loss

    def exclusive_classification_labels(self, targets):
        sten_list = []
        plq_list = []

        for tgt in targets:
            boxes = tgt["boxes"]

            if boxes.shape[0] == 0:
                sten_list.append(0)
                plq_list.append(0)
            else:
                sten_list.append(int(tgt["sten"][0]))
                plq_list.append(int(tgt["plq"][0]))

        sten_gt = torch.tensor(sten_list, dtype=torch.long)
        plq_gt = torch.tensor(plq_list, dtype=torch.long)

        return sten_gt, plq_gt

    def targets_to_device(self, sten_cls_targets, plq_cls_targets, targets):

        device_tmp = self.device

        if device_tmp is None:
            device_tmp = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sten_cls_targets_device = [t.to(device_tmp) for t in sten_cls_targets]
        plq_cls_targets_device = [t.to(device_tmp) for t in plq_cls_targets]

        targets_device = []
        for t in targets:
            t_gpu = {}
            for k, v in t.items():
                t_gpu[k] = v.to(device_tmp)
            targets_device.append(t_gpu)

        return sten_cls_targets_device, plq_cls_targets_device, targets_device

    def forward(self, group_outputs, group_targets, pattern='main_training'):

        if pattern == 'pre_training_1':

            sten_outputs, plq_outputs = group_outputs['sten_cls'], group_outputs['plq_cls']
            sten_targets, plq_targets = self.exclusive_classification_labels(group_targets)
            plq_targets += 6

            sten_targets = sten_targets.to(self.device)
            plq_targets = plq_targets.to(self.device)

            return (self.cls_loss(sten_outputs, sten_targets) + self.cls_loss(plq_outputs, plq_targets)) / 2


        group_targets = funcs.process_target_dict(group_targets)

        sten_cls_targets, plq_cls_targets, targets = group_targets['sten_cls_target'], \
                                                     group_targets['plq_cls_target'], \
                                                     group_targets['detection_target']

        sten_outputs, plq_outputs, outputs = group_outputs['out_sten'], \
                                             group_outputs['out_plq'], \
                                             group_outputs['out_roi']

        outputs["pred_boxes"] = outputs["pred_boxes"].sigmoid()

        sten_cls_targets, plq_cls_targets, targets = self.targets_to_device(sten_cls_targets, plq_cls_targets, targets)

        indices = self.matcher(outputs, targets)

        sten_targets = self.classification_label_mapping(sten_cls_targets, indices)
        plq_targets = self.classification_label_mapping(plq_cls_targets, indices)

        if pattern == 'main_training':
            rds_idx, oap_idx = self.patient_level_index(sten_targets, plq_targets)
            loss_patient_level = (self.patient_level_prediction_loss(sten_outputs, sten_targets, rds_idx) +
                                  self.patient_level_prediction_loss(plq_outputs, plq_targets, oap_idx)) / 2

        sten_outputs, sten_targets = sten_outputs.view(-1, sten_outputs.size(-1)), sten_targets.view(-1)
        plq_outputs, plq_targets = plq_outputs.view(-1, plq_outputs.size(-1)), plq_targets.view(-1)

        loss_artery_level = (self.cls_loss(sten_outputs, sten_targets) + self.cls_loss(plq_outputs, plq_targets)) / 2

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / funcs.get_world_size(), min=1).item()

        loss_labels, empty_batch = self.loss_labels(outputs, targets, indices)

        if empty_batch == True:
            return loss_labels

        loss_boxes = self.loss_boxes(outputs, targets, indices, num_boxes)

        if pattern == 'main_training':
            return (loss_boxes + loss_labels + loss_artery_level) * self.eta + loss_patient_level * (1 - self.eta)
        return loss_boxes + loss_labels + loss_artery_level



