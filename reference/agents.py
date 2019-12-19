import os
import sys
import copy
import json
import torch
import pickle
import dotmap
import random
import logging
import numpy as np
from tqdm import tqdm
from dotmap import DotMap
from itertools import chain
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from reference.utils import save_checkpoint
from reference.utils import AverageMeter
from reference.setup import print_cuda_statistics
from reference.datasets.chairs import ChairsInContext
from reference.datasets.colors import ColorsInContext
from reference.datasets.refcoco import CocoInContext
from reference.models import Witness


class BaseAgent(object):

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        
        self._load_datasets()
        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset)
        self.val_loader, self.val_len = self._create_dataloader(self.val_dataset)

        self._choose_device()
        self._create_model()
        self._create_optimizer()

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
        random.seed(self.config.seed)
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
        loader = DataLoader(dataset, batch_size=self.config.optim.batch_size, shuffle=True, 
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
        for epoch in range(self.current_epoch, self.config.optim.epochs):
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
   
    def __init__(self, config, override_vocab = None):
        self.override_vocab = override_vocab
        super().__init__(config)

        if not self.config.train_image_from_scratch:
            assert self.config.pretrain_image_embedding_dir is not None
            
            pretrain_image_embedding_path = os.path.join(
                self.config.pretrain_root,
                self.config.dataset,
                self.config.pretrain_image_embedding_dir,
            )

            train_image_a_embedding_file = os.path.join(
                pretrain_image_embedding_path, 
                'train_image_a.npy',
            )
            train_image_b_embedding_file = os.path.join(
                pretrain_image_embedding_path, 
                'train_image_b.npy',
            )
            train_image_c_embedding_file = os.path.join(
                pretrain_image_embedding_path,
                'train_image_c.npy',
            )

            val_image_a_embedding_file = os.path.join(
                pretrain_image_embedding_path,
                'val_image_a.npy',
            )
            val_image_b_embedding_file = os.path.join(
                pretrain_image_embedding_path,
                'val_image_b.npy',
            )
            val_image_c_embedding_file = os.path.join(
                pretrain_image_embedding_path,
                'val_image_c.npy',
            )
            
            self.train_image_a_embeddings = np.load(train_image_a_embedding_file)
            self.train_image_b_embeddings = np.load(train_image_b_embedding_file)
            self.train_image_c_embeddings = np.load(train_image_c_embedding_file)

            self.val_image_a_embeddings = np.load(val_image_a_embedding_file)
            self.val_image_b_embeddings = np.load(val_image_b_embedding_file)
            self.val_image_c_embeddings = np.load(val_image_c_embedding_file)

            # if we have chosen a subset then, we need to properly subset these
            if self.config.data.data_size is not None:
                subset = self.train_dataset.subset_indices
                assert subset is not None
                self.train_image_a_embeddings = self.train_image_a_embeddings[subset]
                self.train_image_b_embeddings = self.train_image_b_embeddings[subset]
                self.train_image_c_embeddings = self.train_image_c_embeddings[subset]

        if not self.config.train_text_from_scratch:
            assert self.config.pretrain_text_embedding_dir is not None

            pretrain_text_embedding_path = os.path.join(
                self.config.pretrain_root,
                self.config.dataset,
                self.config.pretrain_text_embedding_dir,
            )

            train_embedding_file = os.path.join(
                pretrain_text_embedding_path,
                'train.npy',
            )
            val_embedding_file = os.path.join(
                pretrain_text_embedding_path,
                'val.npy',
            )

            self.train_text_embeddings = np.load(train_embedding_file)
            self.val_text_embeddings = np.load(val_embedding_file)

            if self.config.data.data_size is not None:
                subset = self.train_dataset.subset_indices
                assert subset is not None
                self.train_text_embeddings = self.train_text_embeddings[subset]

    def _load_datasets(self):

        if self.config.dataset == 'chairs_in_context':
            DatasetClass = ChairsInContext
        elif self.config.dataset == 'colors_in_context':
            DatasetClass = ColorsInContext
        elif self.config.dataset in ['refclef', 'refcoco', 'refcoco+']: 
            DatasetClass = CocoInContext
        else:
            raise Exception(f'Dataset {self.config.dataset} not supported.')
       
        train_dataset = DatasetClass(
            os.path.join(self.config.data_dir, self.config.dataset),
            data_size = self.config.data.data_size,
            image_size = self.config.data.image_size,
            vocab = self.override_vocab,
            split = 'train', 
            context_condition = self.config.data.context_condition,
            split_mode = self.config.data.split_mode, 
            train_frac = 0.80,
            val_frac = 0.10,
            image_transform = None,
            random_seed = self.config.seed,
        )
        val_dataset = DatasetClass(
            os.path.join(self.config.data_dir, self.config.dataset),
            image_size = self.config.data.image_size,
            vocab = train_dataset.vocab,
            split = 'val',  # NOTE: do not bleed test in
            context_condition = self.config.data.context_condition,
            split_mode = self.config.data.split_mode, 
            train_frac = 0.80,
            val_frac = 0.10,
            image_transform = None,
            random_seed = self.config.seed,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.vocab = train_dataset.vocab
        self.vocab_size = len(self.vocab['w2i'])

    def _create_model(self):
        self.model = Witness(
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
            gru_bidirectional = self.config.model.text.gru_bidirectional,
            n_gru_layers = self.config.model.text.n_gru_layers,
            # ---
            sneak_peak = self.config.model.text.sneak_peak,
        ).to(self.device)

    def _create_optimizer(self):
        if self.config.optim.optimizer == 'SGD':
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr = self.config.optim.learning_rate,
                momentum = self.config.optim.momentum,
                weight_decay = self.config.optim.weight_decay,
            )
        elif self.config.optim.optimizer == 'Adam':
            self.optim = torch.optim.Adam(
                self.model.parameters(),
                lr = self.config.optim.learning_rate,
                weight_decay = self.config.optim.weight_decay,
            )
        if self.config.optim.auto_schedule:
            self.scheduler = ReduceLROnPlateau(
                self.optim, 
                mode = 'min',
                factor = 0.1,
                verbose = True,
                patience = self.config.optim.patience,
            )
        else:
            self.scheduler = None

    def train(self):
        """
        Train until patience runs out. Then lower the learning rate
        then keep training. Do that twice.
        """
        for epoch in range(self.current_epoch, self.config.optim.epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            if epoch % self.config.optim.val_freq == 0:
                self.validate()
            if self.config.optim.auto_schedule:
                self.scheduler.step(self.current_val_loss)
            self.save_checkpoint()

    def train_one_epoch(self):
        num_batches = self.train_len // self.config.optim.batch_size
        tqdm_batch = tqdm(total=num_batches,
                            desc="[Epoch {}]".format(self.current_epoch))

        self.model.train()
        epoch_loss = AverageMeter()

        for index, image_a, image_b, image_c, text_seq, text_len, label in self.train_loader:
            batch_size = image_a.size(0)

            image_a = image_a.to(self.device)
            image_b = image_b.to(self.device)
            image_c = image_c.to(self.device)
            text_seq = text_seq.to(self.device)
            text_len = text_len.to(self.device)
            label = label.to(self.device)

            image_emb_a, image_emb_b, image_emb_c = None, None, None
            if not self.config.train_image_from_scratch:
                image_emb_a, image_emb_b, image_emb_c = extract_image_embeddings(
                    index, 
                    self.train_image_a_embeddings,
                    self.train_image_b_embeddings,
                    self.train_image_c_embeddings,
                    self.device,
                )

            text_emb = None
            if not self.config.train_text_from_scratch:
                text_emb = extract_text_embeddings(index, self.train_text_embeddings, self.device)

            logit_a = self.model(image_a, text_seq, text_len, image_emb = image_emb_a, text_emb = text_emb)
            logit_b = self.model(image_b, text_seq, text_len, image_emb = image_emb_b, text_emb = text_emb)
            logit_c = self.model(image_c, text_seq, text_len, image_emb = image_emb_c, text_emb = text_emb)

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
            for index, image_a, image_b, image_c, text_seq, text_len, label in self.val_loader:
                batch_size = image_a.size(0)

                image_a = image_a.to(self.device)
                image_b = image_b.to(self.device)
                image_c = image_c.to(self.device)
                text_seq = text_seq.to(self.device)
                text_len = text_len.to(self.device)
                label = label.to(self.device)

                image_emb_a, image_emb_b, image_emb_c = None, None, None
                if not self.config.train_image_from_scratch:
                    image_emb_a, image_emb_b, image_emb_c = extract_image_embeddings(
                        index, 
                        self.val_image_a_embeddings,
                        self.val_image_b_embeddings,
                        self.val_image_c_embeddings,
                        self.device,
                    )

                text_emb = None
                if not self.config.train_text_from_scratch:
                    text_emb = extract_text_embeddings(index, self.val_text_embeddings, self.device)

                logit_a = self.model(image_a, text_seq, text_len, image_emb = image_emb_a, text_emb = text_emb)
                logit_b = self.model(image_b, text_seq, text_len, image_emb = image_emb_b, text_emb = text_emb)
                logit_c = self.model(image_c, text_seq, text_len, image_emb = image_emb_c, text_emb = text_emb)

                logits = torch.cat([logit_a, logit_b, logit_c], dim=1)

                loss = F.cross_entropy(logits, label)
                epoch_loss.update(loss.item(), batch_size)

                pred = F.softmax(logits, dim=1)
                _, pred = torch.max(pred, 1)
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
            'vocab': self.train_dataset.vocab,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
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
        self.checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoints', checkpoint_name))
        self.config = self.checkpoint['config']
        self.agent = TrainAgent(self.config, override_vocab = self.checkpoint['vocab'])
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

        if not self.config.train_image_from_scratch:
            assert self.config.pretrain_image_embedding_dir is not None

            pretrain_image_embedding_path = os.path.join(
                self.config.pretrain_root,
                self.config.dataset,
                self.config.pretrain_image_embedding_dir,
            )
            
            test_image_a_embedding_file = os.path.join(
                pretrain_image_embedding_path,
                'test_image_a.npy',
            )
            test_image_b_embedding_file = os.path.join(
                pretrain_image_embedding_path,
                'test_image_b.npy',
            )
            test_image_c_embedding_file = os.path.join(
                pretrain_image_embedding_path,
                'test_image_c.npy',
            ) 
            
            self.test_image_a_embeddings = np.load(test_image_a_embedding_file)
            self.test_image_b_embeddings = np.load(test_image_b_embedding_file)
            self.test_image_c_embeddings = np.load(test_image_c_embedding_file)

        if not self.config.train_text_from_scratch:
            assert self.config.pretrain_text_embedding_dir is not None

            pretrain_text_embedding_path = os.path.join(
                self.config.pretrain_root,
                self.config.dataset,
                self.config.pretrain_text_embedding_dir,
            )

            test_embedding_file = os.path.join(
                pretrain_text_embedding_path,
                'test.npy',
            )

            self.test_text_embeddings = np.load(test_embedding_file)

    def _load_datasets(self):

        if self.config.dataset == 'chairs_in_context':
            DatasetClass = ChairsInContext
        elif self.config.dataset == 'colors_in_context':
            DatasetClass = ColorsInContext
        elif self.config.dataset in ['refclef', 'refcoco', 'refcoco+']: 
            DatasetClass = CocoInContext
        else:
            raise Exception(f'Dataset {self.config.dataset} not supported.')

        test_dataset = DatasetClass(
            os.path.join(self.config.data_dir, self.config.dataset),
            image_size = self.config.data.image_size,
            vocab = self.agent.train_dataset.vocab,
            split = 'test',
            context_condition = self.config.data.context_condition,
            split_mode = self.config.data.split_mode, 
            train_frac = 0.80,
            val_frac = 0.10,
            image_transform = None,
            random_seed = self.config.seed,
        )
        self.test_dataset = test_dataset
        self.vocab = self.agent.train_dataset.vocab
        self.vocab_size = len(self.vocab['w2i'])

    def _create_dataloader(self, dataset):
        dataset_size = len(dataset)
        loader = DataLoader(
            dataset, 
            batch_size = self.config.optim.batch_size, 
            shuffle = True, 
            num_workers = self.config.data_loader_workers,
        )
        
        return loader, dataset_size

    def test(self):
        num_batches = self.test_len // self.config.optim.batch_size
        tqdm_batch = tqdm(total=num_batches, desc="[Test]")

        self.agent.model.eval()

        epoch_loss = AverageMeter()
        num_correct = 0.
        num_total = 0.

        with torch.no_grad():
            for index, image_a, image_b, image_c, text_seq, text_len, label in self.test_loader:
                batch_size = image_a.size(0)

                image_a = image_a.to(self.agent.device)
                image_b = image_b.to(self.agent.device)
                image_c = image_c.to(self.agent.device)
                text_seq = text_seq.to(self.agent.device)
                text_len = text_len.to(self.agent.device)
                label = label.to(self.agent.device)

                image_emb_a, image_emb_b, image_emb_c = None, None, None
                if not self.config.train_image_from_scratch:
                    image_emb_a, image_emb_b, image_emb_c = extract_image_embeddings(
                        index, 
                        self.test_image_a_embeddings,
                        self.test_image_b_embeddings,
                        self.test_image_c_embeddings,
                        self.agent.device,
                    )

                text_emb = None
                if not self.config.train_text_from_scratch:
                    text_emb = extract_text_embeddings(index, self.test_text_embeddings, self.agent.device)

                logit_a = self.agent.model(image_a, text_seq, text_len, image_emb = image_emb_a, text_emb = text_emb)
                logit_b = self.agent.model(image_b, text_seq, text_len, image_emb = image_emb_b, text_emb = text_emb)
                logit_c = self.agent.model(image_c, text_seq, text_len, image_emb = image_emb_c, text_emb = text_emb)

                logits = torch.cat([logit_a, logit_b, logit_c], dim=1)

                loss = F.cross_entropy(logits, label)
                epoch_loss.update(loss.item(), batch_size)

                pred = F.softmax(logits, dim=1)
                _, pred = torch.max(pred, 1)
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


class FeatureAgent(object):
    """Agent class to extract features."""

    def __init__(
            self, 
            dataset,
            data_dir,
            context_condition = 'all',
            split_mode = 'easy',
            image_size = None,
            override_vocab = None, 
            batch_size = 128,
            gpu_device = 0, 
            cuda = True,
            seed = 42,
            image_transforms = None,
        ):
        self.dataset = dataset
        self.data_dir = data_dir
        self.context_condition = context_condition
        self.split_mode = split_mode
        self.batch_size = batch_size
        self.gpu_device = gpu_device
        self.cuda = cuda
        self.seed = seed
        self.image_size = image_size
        self.image_transforms = image_transforms
        self.override_vocab = override_vocab
        self._choose_device()

        self._load_datasets()
        self.train_loader, self.train_len = self._create_dataloader(self.train_dataset)
        self.val_loader, self.val_len = self._create_dataloader(self.val_dataset)
        self.test_loader, self.test_len = self._create_dataloader(self.test_dataset)
       
    def _choose_device(self):
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & self.cuda
        self.manual_seed = self.seed

        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.gpu_device)
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
 
    def _create_dataloader(self, dataset):
        dataset_size = len(dataset)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return loader, dataset_size

    def _load_datasets(self):

        if self.dataset == 'chairs_in_context':
            DatasetClass = ChairsInContext
        elif self.dataset == 'colors_in_context':
            DatasetClass = ColorsInContext
        elif self.dataset in ['refclef', 'refcoco', 'refcoco+']: 
            DatasetClass = CocoInContext
        else:
            raise Exception(f'Dataset {self.dataset} not supported.')
        
        train_dataset = DatasetClass(
            os.path.join(self.data_dir, self.dataset),
            image_size = self.image_size,
            vocab = self.override_vocab,
            split = 'train', 
            context_condition = self.context_condition,
            split_mode = self.split_mode, 
            train_frac = 0.80,
            val_frac = 0.10,
            image_transform = self.image_transforms,
        )
        val_dataset = DatasetClass(
            os.path.join(self.data_dir, self.dataset),
            image_size = self.image_size,
            vocab = train_dataset.vocab,
            split = 'val',
            context_condition = self.context_condition,
            split_mode = self.split_mode, 
            train_frac = 0.80,
            val_frac = 0.10,
            image_transform = self.image_transforms,
        )
        test_dataset = DatasetClass(
            os.path.join(self.data_dir, self.dataset),
            image_size = self.image_size,
            vocab = train_dataset.vocab,
            split = 'test',
            context_condition = self.context_condition,
            split_mode = self.split_mode, 
            train_frac = 0.80,
            val_frac = 0.10,
            image_transform = self.image_transforms,
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def extract_features(self, extract_fun, modality='image', split='train'):
        assert split in ['train', 'val', 'test']
        assert modality in ['image', 'text', 'encoded_text']

        if split == 'train':
            dataset = self.train_dataset
            data_loader = self.train_loader
        elif split == 'val':
            dataset = self.val_dataset
            data_loader = self.val_loader
        elif split == 'test':
            dataset = self.test_dataset
            data_loader = self.test_loader

        pbar = tqdm(total=len(data_loader))

        image_embs_a, image_embs_b, image_embs_c, text_embs = [], [], [], []

        for index, image_a, image_b, image_c, text_seq, text_len, _ in data_loader:
            image_a = image_a.to(self.device)
            image_b = image_b.to(self.device)
            image_c = image_c.to(self.device)
            
            if modality == 'image':
                image_emb_a = extract_fun(image_a)
                image_emb_b = extract_fun(image_b)
                image_emb_c = extract_fun(image_c)

                image_embs_a.append(image_emb_a)
                image_embs_b.append(image_emb_b)
                image_embs_c.append(image_emb_c)
            elif modality == 'text':
                raw_text = [dataset.__gettext__(ix.item()) for ix in index]
                text_emb = extract_fun(raw_text)

                text_embs.append(text_emb)
            elif modality == 'encoded_text':
                text_emb = extract_fun(text_seq, text_len)

                text_embs.append(text_emb)
        
            pbar.update()

        pbar.close()

        if modality == 'image':
            image_embs_a = torch.cat(image_embs_a, dim=0)
            image_embs_b = torch.cat(image_embs_b, dim=0)
            image_embs_c = torch.cat(image_embs_c, dim=0)

            return image_embs_a, image_embs_b, image_embs_c
        else:
            text_embs = torch.cat(text_embs, dim=0)

            return text_embs


def extract_image_embeddings(
        index, 
        image_a_embeddings,
        image_b_embeddings,
        image_c_embeddings,
        device,
    ):
    image_emb_a, image_emb_b, image_emb_c = [], [], []
    
    for ix in index:
        image_a_ix = image_a_embeddings[ix.item()]
        image_b_ix = image_b_embeddings[ix.item()] 
        image_c_ix = image_c_embeddings[ix.item()]
        image_emb_a.append(image_a_ix)
        image_emb_b.append(image_b_ix)
        image_emb_c.append(image_c_ix)
    
    image_emb_a = torch.from_numpy(np.stack(image_emb_a)).to(device)
    image_emb_b = torch.from_numpy(np.stack(image_emb_b)).to(device)
    image_emb_c = torch.from_numpy(np.stack(image_emb_c)).to(device)
    
    return image_emb_a, image_emb_b, image_emb_c


def extract_masked_image_embeddings(
        index, 
        full_image_embeddings,
        mask_image_embeddings,
        device,
    ):
    full_image_emb, mask_image_emb = [], []
    
    for ix in index:
        full_image_ix = full_image_embeddings[ix.item()]
        mask_image_ix = mask_image_embeddings[ix.item()] 
        full_image_emb.append(full_image_ix)
        mask_image_emb.append(mask_image_ix)
    
    full_image_emb = torch.from_numpy(np.stack(full_image_emb)).to(device)
    mask_image_emb = torch.from_numpy(np.stack(mask_image_emb)).to(device)
    
    return full_image_emb, mask_image_emb


def extract_text_embeddings(index, text_embeddings, device):
    text_emb = []
    for ix in index:
        text_emb_ix = text_embeddings[ix.item()]
        text_emb.append(text_emb_ix)
    text_emb = torch.from_numpy(np.stack(text_emb)).to(device)
    return text_emb


class MaskedTrainAgent(TrainAgent):

    def __init__(self, config, override_vocab = None):
        self.override_vocab = override_vocab
        super().__init__(config)

        if not self.config.train_image_from_scratch:
            assert self.config.pretrain_image_embedding_dir is not None
            
            pretrain_image_embedding_path = os.path.join(
                self.config.pretrain_root,
                self.config.dataset,
                self.config.pretrain_image_embedding_dir,
            )

            train_full_image_embedding_file = os.path.join(
                pretrain_image_embedding_path, 
                'train_full_image.npy',
            )
            train_mask_image_embedding_file = os.path.join(
                pretrain_image_embedding_path, 
                'train_mask_image.npy',
            )

            val_full_image_embedding_file = os.path.join(
                pretrain_image_embedding_path, 
                'val_full_image.npy',
            )
            val_mask_image_embedding_file = os.path.join(
                pretrain_image_embedding_path, 
                'val_mask_image.npy',
            )

            self.train_full_image_embeddings = np.load(train_full_image_embedding_file)
            self.train_mask_image_embeddings = np.load(train_mask_image_embedding_file)

            self.val_full_image_embeddings = np.load(val_full_image_embedding_file)
            self.val_mask_image_embeddings = np.load(val_mask_image_embedding_file)

            # if we have chosen a subset then, we need to properly subset these
            if self.config.data.data_size is not None:
                subset = self.train_dataset.subset_indices
                assert subset is not None
                self.train_full_image_embeddings = self.train_full_image_embeddings[subset]
                self.train_mask_image_embeddings = self.train_mask_image_embeddings[subset]

        if not self.config.train_text_from_scratch:
            assert self.config.pretrain_text_embedding_dir is not None

            pretrain_text_embedding_path = os.path.join(
                self.config.pretrain_root,
                self.config.dataset,
                self.config.pretrain_text_embedding_dir,
            )

            train_embedding_file = os.path.join(
                pretrain_text_embedding_path,
                'train.npy',
            )
            val_embedding_file = os.path.join(
                pretrain_text_embedding_path,
                'val.npy',
            )

            self.train_text_embeddings = np.load(train_embedding_file)
            self.val_text_embeddings = np.load(val_embedding_file)

            if self.config.data.data_size is not None:
                subset = self.train_dataset.subset_indices
                assert subset is not None
                self.train_text_embeddings = self.train_text_embeddings[subset]
    
    def train_one_epoch(self):
        num_batches = self.train_len // self.config.optim.batch_size
        tqdm_batch = tqdm(total=num_batches,
                            desc="[Epoch {}]".format(self.current_epoch))

        self.model.train()
        epoch_loss = AverageMeter()

        max_object = self.train_dataset.max_classes

        for index, full_image, mask_image_set, text_seq, text_len, label, num_object in self.train_loader:
            batch_size = full_image.size(0)

            full_image = full_image.to(self.device)
            mask_image_set = mask_image_set.to(self.device)
            text_seq = text_seq.to(self.device)
            text_len = text_len.to(self.device)
            label = label.to(self.device)

            full_image_emb, mask_image_set_emb = None, None
            if not self.config.train_image_from_scratch:
                full_image_emb, mask_image_set_emb = extract_masked_image_embeddings(
                    index, 
                    self.train_full_image_embeddings,
                    self.train_mask_image_embeddings,
                    self.device,
                )

            text_emb = None
            if not self.config.train_text_from_scratch:
                text_emb = extract_text_embeddings(index, self.train_text_embeddings, self.device)

            logits = []
            for j in range(max_object):
                mask_image = mask_image_set[:, j]
                
                mask_image_emb = None
                if not self.config.train_image_from_scratch:
                    mask_image_emb = mask_image_set_emb[:, j]

                logit_j = self.model(
                    full_image, 
                    mask_image, 
                    text_seq, 
                    text_len, 
                    full_image_emb = full_image_emb,
                    mask_image_emb = mask_image_emb,
                    text_emb = text_emb,
                )
                logits.append(logit_j)
            logits = torch.cat(logits, dim=1)

            # we have to compute elementwise loss
            loss = 0
            for i in range(batch_size):
                loss_i = F.cross_entropy(
                    logits[i, :num_object[i].item()].unsqueeze(0), 
                    label[i].unsqueeze(0),
                )
                loss = loss + loss_i
            loss = loss / float(batch_size)

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
            for index, full_image, mask_image_set, text_seq, text_len, label, num_object in self.val_loader:
                batch_size = full_image.size(0)

                full_image = full_image.to(self.device)
                mask_image_set = mask_image_set.to(self.device)
                text_seq = text_seq.to(self.device)
                text_len = text_len.to(self.device)
                label = label.to(self.device)

                full_image_emb, mask_image_set_emb = None, None
                if not self.config.train_image_from_scratch:
                    full_image_emb, mask_image_set_emb = extract_masked_image_embeddings(
                        index, 
                        self.train_full_image_embeddings,
                        self.train_mask_image_embeddings,
                        self.device,
                    )

                text_emb = None
                if not self.config.train_text_from_scratch:
                    text_emb = extract_text_embeddings(index, self.val_text_embeddings, self.device)

                logit_a = self.model(image_a, text_seq, text_len, image_emb = image_emb_a, text_emb = text_emb)
                logit_b = self.model(image_b, text_seq, text_len, image_emb = image_emb_b, text_emb = text_emb)
                logit_c = self.model(image_c, text_seq, text_len, image_emb = image_emb_c, text_emb = text_emb)

                logits = []
                for j in range(max_object):
                    mask_image = mask_image_set[:, j]
                    
                    mask_image_emb = None
                    if not self.config.train_image_from_scratch:
                        mask_image_emb = mask_image_set_emb[:, j]

                    logit_j = self.model(
                        full_image, 
                        mask_image, 
                        text_seq, 
                        text_len, 
                        full_image_emb = full_image_emb,
                        mask_image_emb = mask_image_emb,
                        text_emb = text_emb,
                    )
                    logits.append(logit_j)
                logits = torch.cat(logits, dim=1)

                # we have to compute elementwise loss
                loss = 0
                for i in range(batch_size):
                    loss_i = F.cross_entropy(
                        logits[i, :num_object[i].item()].unsqueeze(0), 
                        label[i].unsqueeze(0),
                    )
                    loss = loss + loss_i
                loss = loss / float(batch_size)
                
                epoch_loss.update(loss.item(), batch_size)

                pred = F.softmax(logits, dim=1)
                _, pred = torch.max(pred, 1)
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


class MaskedEvaluateAgent(EvaluateAgent):
    pass


class MaskedFeatureAgent(FeatureAgent):
    pass

