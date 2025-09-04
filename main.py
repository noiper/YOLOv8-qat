import copy
import csv
import os
import random
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


def create_voc_split(dataset_path='../datasets/VOC2012'):
    """
    Creates train.txt and val.txt files for the VOC2012 dataset.
    Assumes an 80/20 split.
    """
    images_path = os.path.join(dataset_path, 'images')
    if not os.path.exists(images_path):
        print(f"VOC2012 images not found in {images_path}")
        return

    all_images = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.jpg')]
    random.shuffle(all_images)

    split_index = int(len(all_images) * 0.8)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    with open('train.txt', 'w') as f:
        for img_path in train_images:
            f.write(f"{img_path}\n")

    with open('val.txt', 'w') as f:
        for img_path in val_images:
            f.write(f"{img_path}\n")

    print("Created train.txt and val.txt for VOC2012 dataset.")


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def train(args, params):
    util.setup_seed()
    util.setup_multi_processes()

    # --- Model Loading ---
    # Create a model with the original 80 classes to load the pretrained weights
    pretrained_model = nn.yolo_v8_n(80) 
    state = torch.load('./weights/v8_n.pth')['model']
    pretrained_model.load_state_dict(state.float().state_dict())
    
    # Now, create our model with the correct number of classes for the new dataset
    model = nn.yolo_v8_n(len(params['names']))

    # Copy weights from the pretrained model, skipping the final classification layers
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # --- Mode Specific Setup (QAT vs Float) ---
    if not args.float:
        # QAT Path
        print("Running in Quantization-Aware Training (QAT) mode.")
        model.eval()
        for m in model.modules():
            if type(m) is nn.Conv and hasattr(m, 'norm'):
                torch.ao.quantization.fuse_modules(m, [["conv", "norm"]], True)
        model.train()
        model = nn.QAT(model)
        model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        torch.quantization.prepare_qat(model, inplace=True)
    else:
        # Floating Point Path
        print("Running in standard Floating-Point fine-tuning mode.")
        model.train()

    model.cuda()

    # --- Optimizer and Scheduler ---
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64
    optimizer = torch.optim.SGD(util.weight_decay(model, params['weight_decay']),
                                params['lr0'], params['momentum'], nesterov=True)
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # --- DataLoader ---
    with open('train.txt') as reader:
        filenames = [line.rstrip() for line in reader.readlines()]
    dataset = Dataset(filenames, args.input_size, params, True)
    loader = data.DataLoader(dataset, args.batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)

    # --- Training Loop ---
    best = 0
    num_steps = len(loader)
    criterion = util.ComputeLoss(model, params)
    num_warmup = max(round(params['warmup_epochs'] * num_steps), 100)
    
    csv_filename = 'weights/step_qat.csv' if not args.float else 'weights/step_float.csv'
    with open(csv_filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'box', 'cls', 'Recall', 'Precision', 'mAP@50', 'mAP'])
        writer.writeheader()
        
        for epoch in range(args.epochs):
            model.train()
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)
            print(('\n' + '%10s' * 4) % ('epoch', 'memory', 'box', 'cls'))
            p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            
            for i, (samples, targets) in p_bar:
                samples = samples.cuda().float() / 255.0
                x = i + num_steps * epoch

                # Warmup
                if x <= num_warmup:
                    xp = [0, num_warmup]
                    fp = [1, 64 / (args.batch_size * args.world_size)]
                    accumulate = max(1, numpy.interp(x, xp, fp).round())
                    for j, y in enumerate(optimizer.param_groups):
                        if j == 0: fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                        else: fp = [0.0, y['initial_lr'] * lr(epoch)]
                        y['lr'] = numpy.interp(x, xp, fp)
                        if 'momentum' in y:
                            fp = [params['warmup_momentum'], params['momentum']]
                            y['momentum'] = numpy.interp(x, xp, fp)

                # Forward
                outputs = model(samples)
                loss_box, loss_cls = criterion(outputs, targets)
                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                loss_total = loss_box + loss_cls
                
                # Backward
                loss_total.backward()

                # Optimize
                if x % accumulate == 0:
                    util.clip_gradients(model)
                    optimizer.step()
                    optimizer.zero_grad()

                # Log
                memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'
                s = ('%10s' * 2 + '%10.3g' * 2) % (f'{epoch + 1}/{args.epochs}', memory, avg_box_loss.avg, avg_cls_loss.avg)
                p_bar.set_description(s)

            scheduler.step()

            # --- Validation and Saving ---
            save = copy.deepcopy(model).eval()
            
            if not args.float:
                # QAT conversion before testing
                save.to('cpu')
                torch.ao.quantization.convert(save, inplace=True)
            
            last = test(args, params, save)

            writer.writerow({'epoch': str(epoch + 1).zfill(3),
                             'box': f'{avg_box_loss.avg:.3f}', 'cls': f'{avg_cls_loss.avg:.3f}',
                             'mAP': f'{last[0]:.3f}', 'mAP@50': f'{last[1]:.3f}',
                             'Recall': f'{last[2]:.3f}', 'Precision': f'{last[3]:.3f}'})
            f.flush()

            if last[0] > best:
                best = last[0]

            # Save model based on mode
            if not args.float:
                # Save QAT model
                save_scripted = torch.jit.script(save.cpu())
                torch.jit.save(save_scripted, './weights/last.ts')
                if best == last[0]:
                    torch.jit.save(save_scripted, './weights/best.ts')
            else:
                # Save Float model
                torch.save(save.state_dict(), './weights/last_float.pth')
                if best == last[0]:
                    torch.save(save.state_dict(), './weights/best_float.pth')
            del save

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, params, model=None):
    with open('val.txt') as reader:
        filenames = [line.rstrip() for line in reader.readlines()]

    dataset = Dataset(filenames, args.input_size, params, False)
    loader = data.DataLoader(dataset, args.batch_size // 2, False, num_workers=8,
                             pin_memory=True, collate_fn=Dataset.collate_fn)
    
    if model is None:
        if not args.float:
            print("Loading QAT model: ./weights/best.ts")
            model = torch.jit.load(f='./weights/best.ts')
        else:
            print("Loading Float model: ./weights/best_float.pth")
            model = nn.yolo_v8_n(len(params['names']))
            model.load_state_dict(torch.load('./weights/best_float.pth'))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    iou_v = torch.linspace(0.5, 0.95, 10, device=device)
    n_iou = iou_v.numel()

    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 4) % ('precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.to(device).float() / 255.0
        _, _, h, w = samples.shape
        scale = torch.tensor((w, h, w, h), device=device)
        
        outputs = model(samples)
        outputs = util.non_max_suppression(outputs, 0.001, 0.7, model.nc if hasattr(model, 'nc') else len(params['names']))
        
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls, box = targets['cls'][idx].to(device), targets['box'][idx].to(device)
            
            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool, device=device)
            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0), device=device), cls.squeeze(-1)))
                continue
            
            if cls.shape[0]:
                target = torch.cat((cls, util.wh2xy(box) * scale), 1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    m_pre, m_rec, map50, mean_ap = 0., 0., 0., 0.
    if metrics:
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
        if len(metrics) and metrics[0].any():
            tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)
    
    print('%10.3g' * 4 % (m_pre, m_rec, map50, mean_ap))
    
    if not args.float:
        model.float() # for QAT training
        
    return mean_ap, map50, m_rec, m_pre


def profile(args, params):
    from thop import profile, clever_format
    model = nn.yolo_v8_n(len(params['names']))
    shape = (1, 3, args.input_size, args.input_size)
    model.eval()
    macs, params_count = profile(model, inputs=(torch.zeros(shape),), verbose=False)
    macs, params_count = clever_format([macs, params_count], "%.3f")
    print(f'MACs: {macs}, Parameters: {params_count}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--train', action='store_true', help="Run training.")
    parser.add_argument('--test', action='store_true', help="Run testing on the best model.")
    parser.add_argument('--prepare-voc', action='store_true', help='Prepare VOC2012 dataset split files.')
    parser.add_argument('--float', action='store_true', help='Run in floating-point mode (disables QAT).')
    
    args = parser.parse_args()
    args.world_size = 1
    
    if not os.path.exists('weights'):
        os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.prepare_voc:
        create_voc_split()
        return

    profile(args, params)
    if args.train:
        train(args, params)
    if args.test:
        test(args, params)

if __name__ == "__main__":
    main()
