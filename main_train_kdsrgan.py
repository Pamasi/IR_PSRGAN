import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
import torch

from utilss import utils_logger
from utilss import utils_image as util
from utilss import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from torchvision import models
from models import enhance_model_gan as net

from utilss.common import config_wandb
import wandb



def main(json_path='options/train_kdsrgan.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iterG, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iterD, init_path_D = option.find_last_checkpoint(opt['path']['models'], net_type='D')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netD'] = init_path_D
    current_step = max(init_iterG, init_iterD)

    # opt['path']['pretrained_netG'] = ''
    # current_step = 0
    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    wandb_run = config_wandb(opt)
    

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'Training seed {seed}')
    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            val_set = define_Dataset(dataset_opt)
            val_loader = DataLoader(val_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''


    
    model = net.ModelGAN(opt)

    model.init_train()

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    n_val = len(val_loader)
    n_train = len(train_loader)
    for _ in range(opt['train']['n_epoch']):  # keep running

        dict_log = {}
       
        for train_data in train_loader:

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                dict_log = {
                    'train/lr': model.current_learning_rate(),

                }

                logs = model.current_log()  # such as loss               
                for k, v in logs.items():  # merge log information into message
                    if f'train/{k}' in dict_log:
                        dict_log[f'train/{k}'] += v
                    else:
                        dict_log[f'train/{k}'] = v
          
        for k, v in dict_log.items():
            dict_log[f'train/{k}'] /=n_train
        
        wandb.log(dict_log)
 
            # -------------------------------
            # 5) save model
            # -------------------------------
        if current_step % opt['train']['checkpoint_save'] == 0:
            model.save(current_step)

        # -------------------------------
        # 6) testing
        # -------------------------------

        avg_psnr = 0.0
        avg_ssim = 0.0


        for val_data in val_loader:
            #image_name_ext = os.path.basename(val_data['L_path'][0])
            #img_name, ext = os.path.splitext(image_name_ext)

            #img_dir = os.path.join(opt['path']['images'], img_name)
            #util.mkdir(img_dir)

            model.feed_data(val_data)
            model.test()

            visuals = model.current_visuals()
            E_img = util.tensor2uint(visuals['E'])
            H_img = util.tensor2uint(visuals['H'])

            # -----------------------
            # save estimated image E
            # -----------------------
            #save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
            #util.imsave(E_img, save_img_path)

            # -----------------------
            # calculate PSNR & SSIM
            # -----------------------
            current_psnr = util.calculate_psnr(E_img, H_img, border=border)
            current_ssim = util.calculate_ssim(E_img, H_img, border=border)

            #logger.info('{:->4d}--> {:>10s} | {:<4.4f}dB | {:<4.4f}'.format(idx, image_name_ext, current_psnr, current_ssim))

            avg_psnr += current_psnr
            avg_ssim += current_ssim

        avg_psnr /= n_val
        avg_ssim /= n_val

        # testing log
        dict_log ={
            'val/avg_psnr':  avg_psnr,
            'val/avg_ssim':avg_ssim
        }
          
    print('Saving the final model.')
    model.save('ir_psrgan')
    print('End of training.')


if __name__ == '__main__':
    main()


