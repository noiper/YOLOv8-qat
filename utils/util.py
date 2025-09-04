import copy
import math
import random
from time import time

import numpy
import torch
import torchvision


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def make_anchors(x, strides, offset=0.5):
    anchors, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=x[i].device, dtype=x[i].dtype) + offset  # shift x
        sy = torch.arange(end=h, device=x[i].device, dtype=x[i].dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchors.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchors), torch.cat(stride_tensor)


def compute_metric(output, target, iou_v):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2, 2)
    (b1, b2) = output[:, :4].unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 5]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)


def non_max_suppression(outputs, conf_threshold, iou_threshold, nc):
    max_wh = 7680
    max_det = 300
    max_nms = 30000

    shape = outputs[0].shape[0]
    stride = torch.tensor([8.0, 16.0, 32.0], device=outputs[0].device)

    anchors, strides = (x.transpose(0, 1) for x in make_anchors(outputs, stride, 0.5))

    box, cls = torch.cat([i.view(shape, nc + 4, -1) for i in outputs], dim=2).split((4, nc), 1)
    a, b = box.chunk(2, 1)
    a = anchors.unsqueeze(0) - a
    b = anchors.unsqueeze(0) + b
    box = torch.cat(((a + b) / 2, b - a), dim=1)
    outputs = torch.cat((box * strides, cls.sigmoid()), dim=1)

    bs = outputs.shape[0]
    nc = outputs.shape[1] - 4
    xc = outputs[:, 4:4 + nc].amax(1) > conf_threshold

    start_time = time()
    time_limit = 0.5 + 0.05 * bs
    nms_outputs = [torch.zeros((0, 6), device=outputs.device)] * bs
    for index, output in enumerate(outputs):
        output = output.transpose(0, -1)[xc[index]]

        if not output.shape[0]:
            continue

        box, cls = output.split((4, nc), 1)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            output = torch.cat((box[i], output[i, 4 + j, None], j[:, None].float()), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            output = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        n = output.shape[0]
        if not n:
            continue
        output = output[output[:, 4].argsort(descending=True)[:max_nms]]

        c = output[:, 5:6] * max_wh
        boxes, scores = output[:, :4] + c, output[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_threshold)
        i = i[:max_det]

        nms_outputs[index] = output[i]
        if (time() - start_time) > time_limit:
            break

    return nms_outputs


def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = numpy.ones(nf // 2)
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')


def compute_ap(tp, conf, pred_cls, target_cls, eps=1E-16):
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px = numpy.linspace(0, 1, 1000)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]
        no = i.sum()
        if no == 0 or nl == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (nl + eps)
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc)
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))
            x = numpy.linspace(0, 1, 101)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)

    f1 = 2 * p * r / (p + r + eps)
    i = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    ap50, ap = ap[:, 0], ap.mean(1)
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap


def compute_iou(box1, box2, eps=1E-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def weight_decay(model, decay):
    p1, p2 = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            p1.append(param)
        else:
            p2.append(param)
    return [{'params': p1, 'weight_decay': 0.0}, {'params': p2, 'weight_decay': decay}]


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num += n
            self.sum += v * n
            self.avg = self.sum / self.num

class Assigner:
    def __init__(self, top_k=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        self.top_k = top_k
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(self, pred_scores, pred_bboxes, anchors, gt_labels, gt_bboxes, mask_gt):
        num_gt = gt_labels.size(1)
        if num_gt == 0:
            device = gt_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.bg_idx),
                    torch.zeros_like(pred_bboxes),
                    torch.zeros_like(pred_scores),
                    torch.zeros_like(pred_scores[..., 0]).bool())

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, mask_gt)
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, num_gt)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def get_pos_mask(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, mask_gt):
        align_metric, overlaps = self.get_box_metrics(pred_scores, pred_bboxes, gt_labels, gt_bboxes)
        mask_in_gts = self.select_candidates_in_gts(anchors, gt_bboxes)
        mask_top_k = self.select_top_k_candidates(align_metric * mask_in_gts, top_k_mask=mask_gt.repeat([1, 1, self.top_k]).bool())
        mask_pos = mask_top_k * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes):
        """Compute alignment metric given predicted and ground truth boxes."""
        na = pred_bboxes.size(1)
        bs, n_max_boxes, _ = gt_labels.shape
        
        # CORRECTED: Use gather for robust indexing
        pred_scores_expanded = pred_scores.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)
        gt_labels_expanded = gt_labels.long().expand(-1, -1, na).unsqueeze(-1)
        bbox_scores = torch.gather(pred_scores_expanded, 3, gt_labels_expanded).squeeze(-1)
        
        overlaps = compute_iou(gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1), pred_bboxes.unsqueeze(1).expand(-1, n_max_boxes, -1, -1))
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_candidates_in_gts(self, anchors, gt_bboxes):
        n_anchors = anchors.shape[0]
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2).view(gt_bboxes.shape[0], gt_bboxes.shape[1], n_anchors, -1)
        return bbox_deltas.amin(3).gt_(1e-9)

    def select_top_k_candidates(self, metrics, largest=True, top_k_mask=None):
        top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=largest)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        top_k_indices.masked_fill_(~top_k_mask, 0)
        one_hot_pk = torch.zeros(metrics.shape, dtype=metrics.dtype, device=metrics.device)
        one_hot_pk.scatter_(-1, top_k_indices, 1)
        return one_hot_pk

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(gt_labels.size(0), dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * gt_labels.size(1)
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
        target_labels.clamp_(0)
        
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.float32, device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores

    def select_highest_overlaps(self, mask_pos, overlaps, num_gt):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, num_gt, 1])
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos


class BoxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_bboxes, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = compute_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        return ((1.0 - iou) * weight).sum() / target_scores_sum


class ComputeLoss:
    def __init__(self, model, params):
        if hasattr(model, 'module'):
            model = model.module
        device = next(model.parameters()).device
        self.no, self.nc, self.stride = model.no, model.nc, model.stride
        self.params, self.device = params, device
        self.box_loss = BoxLoss().to(device)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = Assigner(top_k=10, num_classes=self.nc, alpha=0.5, beta=6.0)

    def __call__(self, outputs, targets):
        shape = outputs[0].shape
        x_cat = torch.cat([i.view(shape[0], self.no, -1) for i in outputs], 2)
        pred_distri, pred_scores = torch.split(x_cat, (4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        size = torch.tensor(shape[2:], device=self.device, dtype=pred_scores.dtype) * self.stride[0]
        anchors, strides = make_anchors(outputs, self.stride, 0.5)

        indices = targets['idx'].view(-1, 1)
        batch_size = pred_scores.shape[0]
        box_targets = torch.cat((indices, targets['cls'].view(-1, 1), targets['box']), 1).to(self.device)
        
        true = torch.zeros(batch_size, 0, 5, device=self.device)
        if box_targets.shape[0] > 0:
            i = box_targets[:, 0]
            _, counts = i.unique(return_counts=True)
            true = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    true[j, :n] = box_targets[matches, 1:]
            x = true[..., 1:5].mul_(size[[1, 0, 1, 0]])
            y = x.clone()
            y[..., 0], y[..., 1] = x[..., 0] - x[..., 2] / 2, x[..., 1] - x[..., 3] / 2
            y[..., 2], y[..., 3] = x[..., 0] + x[..., 2] / 2, x[..., 1] + x[..., 3] / 2
            true[..., 1:5] = y
            
        gt_labels, gt_bboxes = true.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        pred_bboxes = self.box_decode(anchors, pred_distri)
        
        scores = pred_scores.detach().sigmoid()
        bboxes = (pred_bboxes.detach() * strides).type(gt_bboxes.dtype)
        
        _, target_bboxes, target_scores, fg_mask = self.assigner(scores, bboxes, anchors * strides, gt_labels, gt_bboxes, mask_gt)
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        loss_cls = self.cls_loss(pred_scores, target_scores).sum() / target_scores_sum
        
        loss_box = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            target_bboxes /= strides
            loss_box = self.box_loss(pred_bboxes, target_bboxes, target_scores, target_scores_sum, fg_mask)
            
        loss_box *= self.params['box']
        loss_cls *= self.params['cls']
        return loss_box, loss_cls

    @staticmethod
    def box_decode(anchor_points, pred_dist):
        a, b = pred_dist.chunk(2, -1)
        a, b = anchor_points - a, anchor_points + b
        return torch.cat((a, b), -1)
