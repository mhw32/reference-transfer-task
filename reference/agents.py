import os
import sys
import copy
import torch
import dotmap
import logging
import numpy as np
from tqdm import tqdm
from itertools import chain
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from reference.utils import save_checkpoint
from reference.utils import AverageMeter
from reference.setup import print_cuda_statistics
from reference.datasets.chairs import ChairsInContext
from reference.models import Supervised


class BaseAgent(object):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        
        self._load_datasets()
        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset)
        self.val_loader, self.val_len = self._create_dataloader(self.val_dataset)

        self._choose_device()
        self._create_model()
        self.optims = self._create_optimizer()

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_val_iteration = 0
        
        # we need these to decide best loss
        self.current_loss = 0
        self.current_val_loss = 0
        self.best_val_loss = np.inf

        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _choose_device(self):
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        self.manual_seed = self.config.seed

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

    def _load_datasets(self):
        raise NotImplementedError

    def _create_dataloader(self, dataset):
        dataset_size = len(dataset)
        loader = DataLoader(dataset, batch_size=self.config.optim_params.batch_size, shuffle=True, 
                            num_workers=self.config.data_loader_workers)
        
        return loader, dataset_size

    def _create_model(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt as e:
            self.logger.info("Interrupt detected. Saving data...")
            self.backup()
            raise e

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.test()
            self.save_checkpoint()

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def test(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def backup(self):
        """
        Backs up the model upon interrupt
        """
        self.logger.info("Backing up current version of model...")
        self.save_checkpoint(filename='backup.pth.tar')

    def finalise(self):
        """
        Do appropriate saving after model is finished training
        """
        self.logger.info("Saving final versions of model...")
        self.save_checkpoint(filename='final.pth.tar')


class TrainAgent(BaseAgent):
    """Agent class to train reference game witness functions."""
    
    def _load_datasets(self):
        train_transforms = transforms.ToTensor()
        train_dataset = ChairsInContext(
            self.config.data_dir,
            data_size = self.config.data.data_size,
            vocab = None,
            split = 'train', 
            context_condition = self.config.data.context_condition,
            split_mode = self.config.data.split_mode, 
            image_size = self.config.data.image_size, 
            train_frac = 0.64,
            val_frac = 0.16,
            image_transform = train_transforms,
        )
        val_transforms = transforms.ToTensor()
        val_dataset = ChairsInContext(
            self.config.data_dir,
            data_size = self.config.data.data_size,
            vocab = train_dataset.vocab,
            split = 'val',  # NOTE: do not bleed test in
            context_condition = self.config.data.context_condition,
            split_mode = self.config.data.split_mode, 
            image_size = self.config.data.image_size, 
            train_frac = 0.64,
            val_frac = 0.16,
            image_transform = val_transforms,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.vocab = train_dataset.vocab
        self.vocab_size = len(self.vocab['w2i'])

    def _create_model(self):
        self.model = Supervised(
            # ---
            train_image_from_scratch = self.config.train_image_from_scratch,
            train_text_from_scratch = self.config.train_text_from_scratch,
            # ---
            n_pretrain_image = self.config.model.image.n_pretrain_image,
            n_pretrain_text = self.config.model.text.n_pretrain_text,
            # ---
            n_bottleneck = self.config.model.n_bottleneck,
            n_image_channels = self.config.model.image.n_image_channels,
            n_conv_filters = self.config.model.image.n_conv_filters,
            vocab_size = self.vocab_size,
            n_embedding = self.config.model.text.n_embedding,
            n_gru_hidden = self.config.model.text.n_gru_hidden,
            gru_bidirectional = self.config.model.text.gru_bidireqctional,
            n_gru_layers = self.config.model.text.n_gru_layers,
        )

    def _create_optimizer(self):
        self.optim = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optim.learning_rate,
            momentum=self.config.optim.momentum,
            weight_decay=self.config.optim.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optim, 
            mode = 'min',
            factor = 0.1,
            verbose = True,
            patience = self.config.optim.patience,
        )

    def train(self):
        """
        Train until patience runs out. Then lower the learning rate
        then keep training. Do that twice.
        """
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.validate()
            self.scheduler.step(self.current_val_loss)
            self.save_checkpoint()

    def train_one_epoch(self):
        num_batches = self.train_len // self.config.optim.batch_size
        tqdm_batch = tqdm(total=num_batches,
                            desc="[Epoch {}]".format(self.current_epoch))

        self.model.train()
        epoch_loss = AverageMeter()

        for chair_a, chair_b, chair_c, _, text_seq, text_len, label in self.train_loader:
            batch_size = chair_a.size(0)

            chair_a = chair_a.to(self.device)
            chair_b = chair_b.to(self.device)
            chair_c = chair_c.to(self.device)
            text_seq = text_seq.to(self.device)
            text_len = text_len.to(self.device)
            label = label.to(self.device)

            logit_a = self.model(chair_a, text_seq, text_len)
            logit_b = self.model(chair_b, text_seq, text_len)
            logit_c = self.model(chair_c, text_seq, text_len)

            logits = torch.cat([logit_a, logit_b, logit_c], dim=1)

            loss = F.cross_entropy(logits, label)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            epoch_loss.update(loss.item(), batch_size)
            tqdm_batch.set_postfix({"Train Loss": epoch_loss.avg})

            self.train_losses.append(epoch_loss.val)

            self.current_iteration += 1
            tqdm_batch.update()

        self.current_loss = epoch_loss.avg
        tqdm_batch.close()

    def validate(self):
        num_batches = self.val_len // self.config.optim.batch_size
        tqdm_batch = tqdm(total=num_batches, desc="[Val]")

        self.model.eval()

        epoch_loss = AverageMeter()
        num_correct = 0.
        num_total = 0.

        with torch.no_grad():
            for chair_a, chair_b, chair_c, _, text_seq, text_len, label in self.val_loader:
                batch_size = chair_a.size(0)

                chair_a = chair_a.to(self.device)
                chair_b = chair_b.to(self.device)
                chair_c = chair_c.to(self.device)
                text_seq = text_seq.to(self.device)
                text_len = text_len.to(self.device)
                label = label.to(self.device)
                
                logit_a = self.model(chair_a, text_seq, text_len)
                logit_b = self.model(chair_b, text_seq, text_len)
                logit_c = self.model(chair_c, text_seq, text_len)

                logits = torch.cat([logit_a, logit_b, logit_c], dim=1)

                loss = F.cross_entropy(logits, label)
                epoch_loss.update(loss.item(), batch_size)

                pred = F.softmax(logits, dim=1).max(1)
                num_correct += torch.sum(pred == label).item()
                num_total += batch_size

                tqdm_batch.set_postfix({
                    "Val Loss": epoch_loss.avg,
                    "Val Acc": num_correct / num_total,
                })
                tqdm_batch.update()

        tqdm_batch.close()

        self.current_val_iteration += 1
        self.current_val_loss = epoch_loss.avg
        
        self.val_losses.append(self.current_val_loss)
        self.val_accs.append(num_correct / num_total)

        # save if this was the best validation accuracy
        if self.current_val_loss <= self.best_val_loss:
            self.best_val_loss = self.current_val_loss

        return self.current_val_loss

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'train_loss': self.current_loss,
            'val_iteration': self.current_val_iteration,
            'val_loss': self.current_val_loss,
            'config': self.config,
        }
        is_best = ((self.current_val_loss == self.best_val_loss) or
                   not self.config.validate)
        save_checkpoint(out_dict, is_best, filename=filename,
                        folder=self.config.checkpoint_dir)
    
    def load_checkpoint(
            self, 
            filename, 
            checkpoint_dir = None, 
            load_epoch = True, 
            load_model = True, 
            load_optim = True,
        ):
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        filename = os.path.join(self.config.checkpoint_dir, filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')

            if load_epoch:
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.current_val_iteration = checkpoint['val_iteration']

            if load_model:
                model_state_dict = checkpoint['model_state_dict']
                self.model.load_state_dict(model_state_dict)

            if load_optim:
                optim_state_dict = checkpoint['optim_state_dict']
                self.optim.load_state_dict(optim_state_dict)

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(filename, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("Checkpoint doesnt exists: [{}]".format(filename))
            raise e


class EvaluateAgent(object):
    """Agent class to evaluate trained models."""

    def __init__(
            self, 
            checkpoint_dir, 
            checkpoint_name = 'model_best.pth.tar',
        ):
        
        super().__init__()
        
        self.logger = logging.getLogger("Agent")

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name))
        self.config = self.checkpoint.config
        self.agent = TrainAgent(self.config)
        self.agent.load_checkpoint(
            checkpoint_name, 
            checkpoint_dir = checkpoint_dir, 
            load_model = True,
            load_epoch = False, 
            load_optim = False,
        )

        self._load_datasets()
        self.test_loader, self.test_len = self._create_dataloader(self.test_dataset)
        self.test_losses = []
        self.test_accs = []

    def _load_datasets(self):
        tests_transforms = transforms.ToTensor()
        test_dataset = ChairsInContext(
            self.config.data_dir,
            data_size = self.config.data.data_size,
            vocab = self.agent.train_dataset.vocab,
            split = 'test',
            context_condition = self.config.data.context_condition,
            split_mode = self.config.data.split_mode, 
            image_size = self.config.data.image_size, 
            train_frac = 0.64,
            val_frac = 0.16,
            image_transform = test_transforms,
        )
        self.test_dataset = test_dataset
        self.vocab = self.agent.train_dataset.vocab
        self.vocab_size = len(self.vocab['w2i'])

    def _create_dataloader(self, dataset):
        dataset_size = len(dataset)
        loader = DataLoader(
            dataset, 
            batch_size = self.config.optim_params.batch_size, 
            shuffle = True, 
            num_workers = self.config.data_loader_workers,
        )
        
        return loader, dataset_size

    def test(self):
        num_batches = self.test_len // self.config.optim.batch_size
        tqdm_batch = tqdm(total=num_batches, desc="[Test]")

        self.model.eval()

        epoch_loss = AverageMeter()
        num_correct = 0.
        num_total = 0.

        with torch.no_grad():
            for chair_a, chair_b, chair_c, _, text_seq, text_len, label in self.test_loader:
                batch_size = chair_a.size(0)

                chair_a = chair_a.to(self.device)
                chair_b = chair_b.to(self.device)
                chair_c = chair_c.to(self.device)
                text_seq = text_seq.to(self.device)
                text_len = text_len.to(self.device)
                label = label.to(self.device)
                
                logit_a = self.model(chair_a, text_seq, text_len)
                logit_b = self.model(chair_b, text_seq, text_len)
                logit_c = self.model(chair_c, text_seq, text_len)

                logits = torch.cat([logit_a, logit_b, logit_c], dim=1)

                loss = F.cross_entropy(logits, label)
                epoch_loss.update(loss.item(), batch_size)

                pred = F.softmax(logits, dim=1).max(1)
                num_correct += torch.sum(pred == label).item()
                num_total += batch_size

                tqdm_batch.set_postfix({
                    "Test Loss": epoch_loss.avg,
                    "Test Acc": num_correct / num_total,
                })
                tqdm_batch.update()

        tqdm_batch.close()

        current_test_loss = epoch_loss.avg
        self.test_losses.append(current_test_loss)
        self.test_accs.append(num_correct / num_total)

        return current_test_loss

    def run(self):
        try:
            self.test()
        except KeyboardInterrupt as e:
            self.logger.info("Interrupt detected. Saving data...")
            raise e

    def finalise(self):
        self.logger.info("Finishing and saving metrics...")
        test_losses = np.array(self.test_losses)
        test_accs = np.array(self.test_accs)

        np.savez(
            os.path.join(self.checkpoint_dir, 'test_out.npz'),
            loss = test_losses, 
            acc = test_accs,
        )
