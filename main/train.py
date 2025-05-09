import sys
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from src.teaser_trainer import TeaserTrainer
import os
from datasets.data_utils import load_dataloaders



def parse_args():
    conf = OmegaConf.load(sys.argv[1])  # sys.argv直接读命令行参数，以列表形式读进来；OmegaConf是用来load xxx.yaml文件的

    OmegaConf.set_struct(conf, True)

    sys.argv = [sys.argv[0]] + sys.argv[2:] # Remove the configuration file name from sys.argv

    conf.merge_with_cli()
    return conf

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

if __name__ == '__main__':
    # ----------------------- initialize configuration ----------------------- #
    config = parse_args()

    # ----------------------- initialize log directories ----------------------- #
    os.makedirs(config.train.log_path, exist_ok=True)
    train_images_save_path = os.path.join(config.train.log_path, 'train_images')
    os.makedirs(train_images_save_path, exist_ok=True)
    val_images_save_path = os.path.join(config.train.log_path, 'val_images')
    os.makedirs(val_images_save_path, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.train.log_path, 'config.yaml'))

    train_loader, val_loader = load_dataloaders(config)

    trainer = TeaserTrainer(config)
    trainer = trainer.to(config.device)

    if config.resume:
        trainer.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=config.device)

    # after loading, copy the base encoder 
    # this is used for regularization w.r.t. the base model as well as to compare the results    
    trainer.create_base_encoder()
    
    
    losses_AM_l1_train = AverageMeter()
    losses_AM_l1_val = AverageMeter()
    losses_AM_per_train = AverageMeter()
    losses_AM_per_val = AverageMeter()
    losses_AM_token_cycle_train = AverageMeter()
    losses_AM_token_cycle_val = AverageMeter()
    losses_AM_region_train = AverageMeter()
    losses_AM_region_val = AverageMeter()
    losses_AM_203_landmark_train = AverageMeter()
    losses_AM_203_landmark_val = AverageMeter()
    
    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        
        # restart everything at each epoch!
        trainer.configure_optimizers(len(train_loader))
        for phase in ['train', 'val']:
            loader = train_loader if phase == 'train' else val_loader
            
            for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
                if batch is None:
                    continue
                
                # set the train status of encoder and generator, alternately train them accroding to the batch_idx
                trainer.set_freeze_status(config, batch_idx, epoch)

                for key in batch:
                    batch[key] = batch[key].to(config.device)
                if phase == 'train':
                    outputs = trainer.step(batch, batch_idx, losses_AM_l1_train, losses_AM_per_train, losses_AM_token_cycle_train, losses_AM_region_train, losses_AM_203_landmark_train, phase=phase)
                if phase == 'val':
                    outputs = trainer.step(batch, batch_idx, losses_AM_l1_val, losses_AM_per_val, losses_AM_token_cycle_val, losses_AM_region_val, losses_AM_203_landmark_val, phase=phase)

                if batch_idx % config.train.visualize_every == 0:
                    with torch.no_grad():
                        visualizations = trainer.create_visualizations(batch, outputs)
                        trainer.save_visualizations(visualizations, f"{config.train.log_path}/{phase}_images/{epoch}_{batch_idx}.jpg", show_landmarks=True)
                                    


        if epoch % config.train.save_every == 0:
            trainer.save_model(trainer.state_dict(), os.path.join(config.train.log_path, 'model_{}.pt'.format(epoch)))
