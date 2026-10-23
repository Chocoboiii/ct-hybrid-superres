import os.path
import math
import argparse
import random
import numpy as np
import logging
import csv
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from data.select_dataset import define_Dataset
from models.select_model import define_Model

def main(json_path='options/train_msrresnet_psnr.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path)
    parser.add_argument('--launcher', default='pytorch')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    if opt['dist']: init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)
    border = opt['scale']

    if opt['rank'] == 0: option.save(opt)
    opt = option.dict_to_nonedict(opt)

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    seed = opt['train']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_loader = DataLoader(train_set, batch_size=dataset_opt['dataloader_batch_size'], shuffle=dataset_opt['dataloader_shuffle'], num_workers=dataset_opt['dataloader_num_workers'], drop_last=True, pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    model = define_Model(opt)
    model.init_train()

    # --- ACTIVE LEARNING SETUP ---
    bad_batch_dir = os.path.join(opt['path']['root'], 'debug_bad_batches')
    csv_log_file = os.path.join(opt['path']['log'], 'detailed_metrics.csv')
    if opt['rank'] == 0:
        os.makedirs(bad_batch_dir, exist_ok=True)
        if not os.path.exists(csv_log_file):
            with open(csv_log_file, 'w', newline='') as f:
                csv.writer(f).writerow(['epoch', 'iter', 'lr', 'G_loss', 'train_psnr'])
    
    val_criterion = torch.nn.L1Loss()

    for epoch in range(1000000):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            model.update_learning_rate(current_step)
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if opt['rank'] == 0:
                try:
                    logs = model.current_log()
                    model.test()
                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])
                    train_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    
                    with open(csv_log_file, 'a', newline='') as f:
                        csv.writer(f).writerow([epoch, current_step, model.current_learning_rate(), logs['G_loss'], train_psnr])
                    
                    # DETECTOR DE LOTES MALOS (Warmup 1000)
                    if train_psnr < 19.0 and current_step > 1000:
                        save_name = f"iter{current_step}_psnr{int(train_psnr)}.png"
                        util.imsave(E_img, os.path.join(bad_batch_dir, save_name))
                        if 'H_path' in train_data:
                            with open(os.path.join(bad_batch_dir, "bad_files_list.txt"), "a") as log_file:
                                log_file.write(f"Iter {current_step} (PSNR {train_psnr:.2f}):\n")
                                for p in train_data['H_path']: log_file.write(f"{p}\n")
                                log_file.write("-" * 20 + "\n")
                except: pass

            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> G_loss: {:.4e}, Train_PSNR: {:.2f}dB'.format(epoch, current_step, model.current_learning_rate(), logs['G_loss'], train_psnr)
                logger.info(message)

            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                avg_psnr = 0.0; idx = 0
                for test_data in test_loader:
                    idx += 1
                    model.feed_data(test_data)
                    model.test()
                    visuals = model.current_visuals()
                    avg_psnr += util.calculate_psnr(util.tensor2uint(visuals['E']), util.tensor2uint(visuals['H']), border=border)
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average Val PSNR : {:.2f}dB\n'.format(epoch, current_step, avg_psnr / idx))

if __name__ == '__main__':
    main()