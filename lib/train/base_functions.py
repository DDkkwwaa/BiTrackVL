import torch
from torch.utils.data.distributed import DistributedSampler

from lib.train.data.loader import slt_collate
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, VastTrack
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val",
                        "COCO17", "VID", "TRACKINGNET", "VastTrack"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "VastTrack":
            datasets.append(VastTrack(settings.env.vasttrack_dir, image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings,
                                                       num_prev=0)

    data_processing_val = processing.STARKProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings,
                                                     num_prev=0)

    # Train sampler and loader
    #import pdb
    #pdb.set_trace()
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    processing_mode = getattr(cfg.DATA, "PROCESSING_MODE", "preprocess_all_frame")
    print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, processing_mode=processing_mode, train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True
    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode,processing_mode=processing_mode, train_cls=train_cls)

    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None

    if processing_mode == "preprocess_all_frame":
        # collate mode == "stack"
        loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                               num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                               epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)
    else:
        # collate mode == "list"
        loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                                 num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, collate_fn=slt_collate, sampler=train_sampler)

        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                               num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, collate_fn=slt_collate, sampler=val_sampler,
                               epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    if train_cls:
        print("Only training classification head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "cls" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                print(n)
    else:
        # Separate text encoder parameters for independent learning rate
        text_params = []
        other_params = []
        
        for n, p in net.named_parameters():
            if not p.requires_grad:
                continue
            
            if "text_encoder" in n:
                text_params.append(p)
            elif "backbone" not in n:
                other_params.append(p)

        param_dicts = [
            {"params": other_params},
            {
                "params": text_params, 
                "lr": cfg.TRAIN.LR * 0.1  # Independent LR for text module (10% of base LR)
            },
        ]
        
        if is_main_process():
            print("Learnable parameters are shown below.")
            print(f"Text Module Params (LR={cfg.TRAIN.LR * 0.1}): {[n for n, p in net.named_parameters() if 'text_encoder' in n and p.requires_grad]}")
            for n, p in net.named_parameters():
                if p.requires_grad and "text_encoder" not in n:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")

    # Custom Scheduler with Warmup
    warmup_epochs = getattr(cfg.TRAIN.SCHEDULER, "WARMUP_EPOCH", 0)
    warmup_factor = getattr(cfg.TRAIN.SCHEDULER, "WARMUP_FACTOR", 0.01)

    # 1. Visual Branch Scheduler (Standard Step/Mstep)
    def visual_scheduler(epoch):
        if epoch < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs)
        
        if cfg.TRAIN.SCHEDULER.TYPE == 'step':
            decay_rate = getattr(cfg.TRAIN.SCHEDULER, "DECAY_RATE", 0.1)
            step_size = cfg.TRAIN.LR_DROP_EPOCH
            return decay_rate ** (epoch // step_size)
            
        elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
            decay_rate = getattr(cfg.TRAIN.SCHEDULER, "GAMMA", 0.1)
            milestones = cfg.TRAIN.SCHEDULER.MILESTONES
            count = sum([1 for m in milestones if epoch >= m])
            return decay_rate ** count
            
        return 1.0

    # 2. Text Branch Scheduler (Cosine Annealing for Self-Supervised Stability)
    def text_scheduler(epoch):
        if epoch < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs)
        
        import math
        T_max = cfg.TRAIN.EPOCH - warmup_epochs
        # Decay to near zero (1% of initial)
        eta_min_ratio = 0.1 
        
        curr_epoch = epoch - warmup_epochs
        if curr_epoch < 0: curr_epoch = 0
        if curr_epoch > T_max: curr_epoch = T_max
        
        # Cosine decay: 1.0 -> 0.01
        factor = eta_min_ratio + 0.5 * (1 - eta_min_ratio) * (1 + math.cos(math.pi * curr_epoch / T_max))
        return factor

    # Apply different schedulers to different parameter groups
    # Group 0: Visual/Other Params -> visual_scheduler
    # Group 1: Text Params -> text_scheduler
    if train_cls:
        # If only training cls, we only have one group usually, or logic differs. 
        # Assuming standard training flow here based on previous context.
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, visual_scheduler)
    else:
        # Check if we actually split the groups (only if text_encoder exists and we are in that branch)
        if len(optimizer.param_groups) == 2:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [visual_scheduler, text_scheduler])
        else:
            # Fallback for single group (e.g. if no text params found)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, visual_scheduler)
    
    return optimizer, lr_scheduler