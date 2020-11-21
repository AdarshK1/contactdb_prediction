from models.voxnet import DiverseVoxNet as VoxNet
from models.voxnet import VoxNetClassPred
from models.pointnet import DiversePointNet as PointNet
from models.losses import DiverseLoss
from voxel_dataset import VoxelDataset, VoxelPredictionDataset
from pointcloud_dataset import PointCloudDataset

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
# import visdom
from torch.utils.data import DataLoader
from torch import optim as toptim
import numpy as np
import os
import torch
import configparser
import argparse
import logging
from IPython.core.debugger import set_trace
import wandb
from datetime import datetime

osp = os.path


def create_plot_window(vis, xlabel, ylabel, title, win, env, trace_name):
    if not isinstance(trace_name, list):
        trace_name = [trace_name]

    vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
             name=trace_name[0],
             opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))
    for name in trace_name[1:]:
        vis.line(X=np.array([1]), Y=np.array([np.nan]), win=win, env=env,
                 name=name)


def train(data_dir, instruction, config_file, experiment_suffix=None,
          checkpoint_dir='.', device_id=0, weights_filename=None,
          include_sessions=None, exclude_sessions=None):
    # config
    # config = configparser.ConfigParser()
    # config.read(config_file)
    #
    # section = config['optim']
    # batch_size = section.getint('batch_size')
    # max_epochs = section.getint('max_epochs')
    # val_interval = section.getint('val_interval')
    # do_val = val_interval > 0
    # base_lr = section.getfloat('base_lr')
    # momentum = section.getfloat('momentum')
    # weight_decay = section.getfloat('weight_decay')
    #
    # section = config['misc']
    # log_interval = section.getint('log_interval')
    # shuffle = section.getboolean('shuffle')
    # num_workers = section.getint('num_workers')
    #
    # section = config['hyperparams']
    # n_ensemble = section.getint('n_ensemble')
    # diverse_beta = section.getfloat('diverse_beta')
    # pos_weight = section.getfloat('pos_weight')
    # droprate = section.getfloat('droprate')
    # lr_step_size = section.getint('lr_step_size', 10000)
    # lr_gamma = section.getfloat('lr_gamma', 1.0)

    voxnet_prediction = False

    hyperparameter_defaults = dict(
        num_workers=48,
        grid_size=64,
        random_rotation=180,
        n_ensemble=1,
        diverse_beta=1,
        pos_weight=10,
        droprate=0.0,
        lr_step_size=1000,
        lr_gamma=0.1,

        batch_size=30,
        max_epochs=1500,
        val_interval=10,
        weight_decay=0.00025,
        base_lr=0.001,
        momentum=0.9,

        log_interval=20,
        shuffle=True,
        fcn_size1=3488,
        fcn_size2=468,)

    dt = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    wandb.init(project="contactdb", name=dt, config=hyperparameter_defaults)
    config = wandb.config

    # cuda
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    else:
        devices = os.environ['CUDA_VISIBLE_DEVICES']
        devices = devices.split(',')[device_id]
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
    device = 'cuda:0'

    # create dataset and model
    model_name = config_file.split('/')[-1].split('.')[0]
    kwargs = dict(data_dir=data_dir, instruction=instruction,
                  include_sessions=include_sessions, exclude_sessions=exclude_sessions,
                  n_ensemble=config.n_ensemble)
    if 'voxnet_prediction' in model_name:
        print("doing pred")
        model = VoxNetClassPred(n_ensemble=config.n_ensemble, droprate=config.droprate, fcn_size1=config.fcn_size1,
                                fcn_size2=config.fcn_size2)
        # grid_size = config['hyperparams'].getint('grid_size')
        # random_rotation = config['hyperparams'].getfloat('random_rotation')

        grid_size = config.grid_size
        random_rotation = config.random_rotation

        train_dset = VoxelPredictionDataset(grid_size=grid_size,
                                            random_rotation=random_rotation, train=True, **kwargs)
        val_dset = VoxelPredictionDataset(grid_size=grid_size, random_rotation=0,
                                          train=False, **kwargs)
        voxnet_prediction = True
    elif 'voxnet' in model_name:
        model = VoxNet(n_ensemble=config.n_ensemble, droprate=config.droprate)
        grid_size = config['hyperparams'].getint('grid_size')
        random_rotation = config['hyperparams'].getfloat('random_rotation')
        train_dset = VoxelDataset(grid_size=grid_size,
                                  random_rotation=random_rotation, train=True, **kwargs)
        val_dset = VoxelDataset(grid_size=grid_size, random_rotation=0,
                                train=False, **kwargs)
    elif 'pointnet' in model_name:
        model = PointNet(n_ensemble=config.n_ensemble, droprate=config.droprate)
        n_points = config.n_points  # section.getint('n_points')
        random_rotation = config['hyperparams'].getfloat('random_rotation')
        random_scale = config['hyperparams'].getfloat('random_scale')
        train_dset = PointCloudDataset(n_points=n_points, train=True,
                                       random_rotation=random_rotation, random_scale=random_scale, **kwargs)
        val_dset = PointCloudDataset(n_points=n_points, train=False,
                                     random_rotation=0, random_scale=0, **kwargs)
    else:
        raise NotImplementedError

    # checkpointing
    exp_name = '{:s}_{:s}_diversenet'.format(instruction, model_name)
    if experiment_suffix:
        exp_name += '_{:s}'.format(experiment_suffix)

    def checkpoint_fn(engine: Engine):
        return -engine.state.avg_loss

    checkpoint_dir = osp.join(checkpoint_dir, exp_name)
    checkpoint_kwargs = dict(dirname=checkpoint_dir, filename_prefix='checkpoint',
                             score_function=checkpoint_fn, create_dir=True, require_empty=False,
                             save_as_state_dict=True)
    checkpoint_dict = {'model': model}

    # logging
    if not osp.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    log_filename = osp.join(checkpoint_dir, 'training_log.txt')
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()])
    logger = logging.getLogger()
    logger.info('Config from {:s}:'.format(config_file))
    with open(config_file, 'r') as f:
        for line in f:
            logger.info(line.strip())

    # load weights
    if weights_filename is not None:
        checkpoint = torch.load(osp.expanduser(weights_filename))
        checkpoint = {k: v for k, v in checkpoint.items() if 'conv4' not in k}
        model.load_state_dict(checkpoint, strict=False)
        logger.info('Loaded weights from {:s}'.format(weights_filename))
    model.to(device=device)

    # loss function
    if voxnet_prediction:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        loss_fn = DiverseLoss(beta=config.diverse_beta, pos_weight=config.pos_weight)

    loss_fn.to(device=device)
    # if do_val:
    #     val_loss_fn = DiverseLoss(beta=diverse_beta, train=False,
    #                               pos_weight=pos_weight)
    #     val_loss_fn.to(device=device)

    # optimizer
    optim = toptim.SGD(model.parameters(), lr=config.base_lr, weight_decay=config.weight_decay,
                       momentum=config.momentum)
    lr_scheduler = toptim.lr_scheduler.StepLR(optim, step_size=config.lr_step_size,
                                              gamma=config.lr_gamma)

    # dataloader
    train_dloader = DataLoader(train_dset, batch_size=config.batch_size, shuffle=config.shuffle,
                               pin_memory=True, num_workers=config.num_workers)
    if config.val_interval > 0:
        val_dloader = DataLoader(val_dset, batch_size=config.batch_size, shuffle=config.shuffle,
                                 pin_memory=True, num_workers=config.num_workers)

    # train and val loops
    def train_loop(engine: Engine, batch):
        geom, tex_targs = batch
        geom = geom.to(device=device, non_blocking=True)  # Nx3xP
        tex_targs = tex_targs.to(device=device, non_blocking=True)  # NxExP
        model.train()
        optim.zero_grad()
        tex_preds = model(geom)  # NxExP
        # print(tex_preds)
        if voxnet_prediction:
            # print("tex_preds", tex_preds.shape)
            # print("tex_preds", tex_preds)
            # print("tex_targs", tex_targs.shape)
            # print("tex_targs", tex_targs)
            loss = loss_fn(tex_preds, tex_targs)
            # print(loss.item())
        else:
            loss, _ = loss_fn(tex_preds, tex_targs)
        wandb.log({"train_loss": loss})
        loss.backward()
        optim.step()
        engine.state.train_loss = loss.item()
        return loss.item()

    trainer = Engine(train_loop)
    train_checkpoint_handler = ModelCheckpoint(score_name='train_loss',
                                               **checkpoint_kwargs)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, train_checkpoint_handler,
                              checkpoint_dict)

    if config.val_interval > 0:
        def val_loop(engine: Engine, batch):
            geom, tex_targs = batch
            geom = geom.to(device=device, non_blocking=True)
            tex_targs = tex_targs.to(device=device, non_blocking=True)
            if 'pointnet' not in model_name:
                # loss explodes for pointnet if model is in eval mode
                model.eval()
            with torch.no_grad():
                tex_preds = model(geom)
            if voxnet_prediction:
                # print("tex_preds", tex_preds.shape)
                # print("tex_targs", tex_targs.shape)
                loss = loss_fn(tex_preds, tex_targs)
            else:
                loss, _ = loss_fn(tex_preds, tex_targs)
            wandb.log({"val_loss": loss})
            engine.state.val_loss = loss.item()
            return loss.item()

        valer = Engine(val_loop)
        val_checkpoint_handler = ModelCheckpoint(score_name='val_loss',
                                                 **checkpoint_kwargs)
        valer.add_event_handler(Events.EPOCH_COMPLETED, val_checkpoint_handler,
                                checkpoint_dict)

    # callbacks
    # vis = visdom.Visdom()
    loss_win = 'loss'
    # create_plot_window(vis, '#Epochs', 'Loss', 'Training and Validation Loss',
    #                    win=loss_win, env=exp_name, trace_name=['train_loss', 'val_loss'])

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        it = (engine.state.iteration - 1) % len(train_dloader)
        engine.state.avg_loss = (engine.state.avg_loss * it + engine.state.output) / \
                                (it + 1)

        if it % config.log_interval == 0:
            logger.info("{:s} train Epoch[{:03d}/{:03d}] Iteration[{:04d}/{:04d}] "
                        "Loss: {:02.4f} lr: {:.4f}".
                        format(exp_name, engine.state.epoch, config.max_epochs, it + 1, len(train_dloader),
                               engine.state.output, lr_scheduler.get_lr()[0]))
            # epoch = engine.state.epoch - 1 + \
            #         float(it) / (len(train_dloader) - 1)

            # vis.line(X=np.array([epoch]), Y=np.array([engine.state.output]),
            #          update='append', win=loss_win, env=exp_name, name='train_loss')

    if config.val_interval > 0:
        @valer.on(Events.ITERATION_COMPLETED)
        def avg_loss_callback(engine: Engine):
            it = (engine.state.iteration - 1) % len(train_dloader)
            engine.state.avg_loss = (engine.state.avg_loss * it + engine.state.output) / \
                                    (it + 1)
            if it % config.log_interval == 0:
                logger.info("{:s} val Iteration[{:04d}/{:04d}] Loss: {:02.4f}"
                            .format(exp_name, it + 1, len(val_dloader), engine.state.output))

        # @valer.on(Events.EPOCH_COMPLETED)
        # def log_val_loss(engine: Engine):
        #     vis.line(X=np.array([trainer.state.epoch]),
        #              Y=np.array([engine.state.avg_loss]), update='append', win=loss_win,
        #              env=exp_name, name='val_loss')

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_val(engine: Engine):
            # vis.save([exp_name])
            if config.val_interval < 0:  # don't do validation
                return
            if engine.state.epoch % config.val_interval != 0:
                return
            valer.run(val_dloader)

        @trainer.on(Events.EPOCH_STARTED)
        def step_lr_scheduler(engine: Engine):
            lr_scheduler.step()

    def reset_avg_loss(engine: Engine):
        engine.state.avg_loss = 0

    trainer.add_event_handler(Events.EPOCH_STARTED, reset_avg_loss)
    if config.val_interval > 0:
        valer.add_event_handler(Events.EPOCH_STARTED, reset_avg_loss)

    # Ignite the torch!
    trainer.run(train_dloader, config.max_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default=osp.join('data', 'voxelized_meshes'))
    parser.add_argument('--checkpoint_dir',
                        default=osp.join('data', 'checkpoints'))
    parser.add_argument('--instruction', required=False, default="use")
    parser.add_argument('--config_file', required=False, default="configs/voxnet_prediction.ini")
    parser.add_argument('--weights_file', default=None)
    parser.add_argument('--suffix', default=None)
    parser.add_argument('--device_id', default=0)
    parser.add_argument('--include_sessions', default=None)
    parser.add_argument('--exclude_sessions', default=None)
    args = parser.parse_args()

    include_sessions = None
    if args.include_sessions is not None:
        include_sessions = args.include_sessions.split(',')
    exclude_sessions = None
    if args.exclude_sessions is not None:
        exclude_sessions = args.exclude_sessions.split(',')

    train(osp.expanduser(args.data_dir), args.instruction, args.config_file,
          experiment_suffix=args.suffix, device_id=args.device_id,
          checkpoint_dir=osp.expanduser(args.checkpoint_dir),
          weights_filename=args.weights_file, include_sessions=include_sessions,
          exclude_sessions=exclude_sessions)

    # train(osp.expanduser(osp.join('data', 'voxelized_meshes')), "use", "configs/voxnet_prediction.ini")
