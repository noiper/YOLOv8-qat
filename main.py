import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import numpy
import torch
import tqdm
import yaml
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def train(args, params):
    util.setup_seed()
    util.setup_multi_processes()

# Model
    model = nn.yolo_v8_s(len(params['names']))
    
    # Load the official yolov8s.pt weights
    state = torch.load('./weights/v8_s.pth', map_location='cpu')['model']
    
    # Create a new state dictionary with matching keys
    new_state_dict = {}
    for k, v in state.state_dict().items():
        # The key names in this project are slightly different from the official ones.
        # This loop renames the keys to match the current model structure.
        name = k.replace('model.', 'net.') # replace "model" with "net"
        # The official model uses integer indexing, but this project uses p1, p2, etc.
        # This part of the code is a bit of a hack to remap the keys, but it works for this model.
        if '.2.' in name: name = name.replace('.2.', '.p2.1.')
        if '.3.' in name: name = name.replace('.3.', '.p3.0.')
        if '.4.' in name: name = name.replace('.4.', '.p3.1.')
        if '.5.' in name: name = name.replace('.5.', '.p4.0.')
        if '.6.' in name: name = name.replace('.6.', '.p4.1.')
        if '.7.' in name: name = name.replace('.7.', '.p5.0.')
        if '.8.' in name: name = name.replace('.8.', '.p5.1.')
        if '.9.' in name: name = name.replace('.9.', '.p5.2.')
        
        # FPN remapping
        if '12.' in name: name = name.replace('12.', 'fpn.h1.')
        if '15.' in name: name = name.replace('15.', 'fpn.h2.')
        if '16.' in name: name = name.replace('16.', 'fpn.h3.')
        if '18.' in name: name = name.replace('18.', 'fpn.h4.')
        if '19.' in name: name = name.replace('19.', 'fpn.h5.')
        if '21.' in name: name = name.replace('21.', 'fpn.h6.')
        
        # Head remapping
        if 'cv2.0.' in name: name = name.replace('cv2.0.', 'box.0.')
        if 'cv2.1.' in name: name = name.replace('cv2.1.', 'box.1.')
        if 'cv2.2.' in name: name = name.replace('cv2.2.', 'box.2.')
        if 'cv3.0.' in name: name = name.replace('cv3.0.', 'cls.0.')
        if 'cv3.1.' in name: name = name.replace('cv3.1.', 'cls.1.')
        if 'cv3.2.' in name: name = name.replace('cv3.2.', 'cls.2.')

        # BN layer remapping
        if 'bn.' in name: name = name.replace('bn.', 'norm.')
        
        new_state_dict[name] = v

    # Load the new state dictionary into the model
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()

    for m in model.modules():
        if type(m) is nn.Conv and hasattr(m, 'norm'):
            torch.ao.quantization.fuse_modules(m, [["conv", "norm"]], True)
    model.train()

    model = nn.QAT(model)
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.quantization.prepare_qat(model, inplace=True)
    model.cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(util.weight_decay(model, params['weight_decay']),
                                params['lr0'], params['momentum'], nesterov=True)

    # Scheduler
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    filenames = []
    with open('../datasets/coco/train2017.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../datasets/coco/images/train2017/' + filename)

    # --- REMOVE THIS ---
    # filenames = filenames[:1000]  # Use only the first 1000 images
    
    dataset = Dataset(filenames, args.input_size, params, True)
    loader = data.DataLoader(dataset, args.batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)
    
    best = 0
    num_steps = len(loader)
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_steps), 100)
    with open('weights/step.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch',
                                                'box', 'cls',
                                                'Recall', 'Precision', 'mAP@50', 'mAP'])
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            print(('\n' + '%10s' * 4) % ('epoch', 'memory', 'box', 'cls'))
            p_bar = tqdm.tqdm(p_bar, total=num_steps)  # progress bar

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:
                samples = samples.cuda()
                samples = samples.float()
                samples = samples / 255.0

                x = i + num_steps * epoch

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0:
                            fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                        else:
                            fp = [0.0, y['initial_lr'] * lr(epoch)]
                        y['lr'] = numpy.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x, xp, fp)

                # Forward
                outputs = model(samples)
                loss_box, loss_cls = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))

                loss_box *= args.batch_size  # loss scaled by batch_size
                loss_cls *= args.batch_size  # loss scaled by batch_size
                loss_box *= args.world_size  # gradient averaged between devices in DDP mode
                loss_cls *= args.world_size  # gradient averaged between devices in DDP mode

                # Backward
                (loss_box + loss_cls).backward()

                # Optimize
                if x % accumulate == 0:
                    util.clip_gradients(model)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()

                # Log
                memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                s = ('%10s' * 2 + '%10.3g' * 2) % (f'{epoch + 1}/{args.epochs}', memory,
                                                    avg_box_loss.avg, avg_cls_loss.avg)
                p_bar.set_description(s)

            # Scheduler
            scheduler.step()

            # Convert model
            save = copy.deepcopy(model)
            save.eval()
            save.to(torch.device('cpu'))
            torch.ao.quantization.convert(save, inplace=True)
            # mAP
            last = test(args, params, save)

            writer.writerow({'epoch': str(epoch + 1).zfill(3),
                            'box': str(f'{avg_box_loss.avg:.3f}'),
                            'cls': str(f'{avg_cls_loss.avg:.3f}'),
                            'mAP': str(f'{last[0]:.3f}'),
                            'mAP@50': str(f'{last[1]:.3f}'),
                            'Recall': str(f'{last[2]:.3f}'),
                            'Precision': str(f'{last[2]:.3f}')})
            f.flush()

            # Update best mAP
            if last[0] > best:
                best = last[0]

            # Save last, best and delete
            save = torch.jit.script(save.cpu())
            torch.jit.save(save, './weights/last.ts')
            if best == last[0]:
                torch.jit.save(save, './weights/best.ts')
            del save

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    with open('../datasets/coco/val2017.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../datasets/coco/images/val2017/' + filename)

    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, args.batch_size // 2, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)
    if model is None:
        model = torch.jit.load(f='./weights/best.ts')

    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 4) % ('precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.to(device)
        samples = samples.float()  # uint8 to fp16/32
        samples = samples / 255.0  # 0 - 255 to 0.0 - 1.0
        _, _, h, w = samples.shape  # batch size, channels, height, width
        scale = torch.tensor((w, h, w, h), device=device)
        # Inference
        outputs = model(samples)
        # NMS
        outputs = util.non_max_suppression(outputs, 0.001, 0.7, model.nc)
        # Metrics
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.to(device)
            box = box.to(device)

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=device)

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0), device=device), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                target = torch.cat((cls, util.wh2xy(box) * scale), 1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # Compute metrics
    metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)
    # Print results
    print('%10.3g' * 4 % (m_pre, m_rec, map50, mean_ap))
    # Return results
    model.float()  # for training
    return mean_ap, map50, m_rec, m_pre


def profile(args, params):
    from thop import profile, clever_format
    model = nn.yolo_v8_s(len(params['names']))
    shape = (1, 3, args.input_size, args.input_size)

    model.eval()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    macs, params = profile(model, inputs=(torch.zeros(shape),), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")

    print(f'MACs: {macs}')
    print(f'Parameters: {params}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.world_size = 1 # Set world size to 1 for single GPU

    if not os.path.exists('weights'):
        os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)
    profile(args, params)
    if args.train:
        train(args, params)
    if args.test:
        test(args, params)


if __name__ == "__main__":
    main()