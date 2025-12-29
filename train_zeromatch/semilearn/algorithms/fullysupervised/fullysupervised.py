# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register('fullysupervised')
class FullySupervised(AlgorithmBase):
    """
        Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

    def train_step(self,  x_lb, y_lb, yp_lb):
        # inference and calculate sup/unsup losses
        with self.amp_cm():

            logits_x_lb = self.model(x_lb)['logits']
            # sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            sup_loss = self.ce_loss(logits_x_lb, yp_lb, reduction='mean') # training with pseudo-label.

        out_dict = self.process_out_dict(loss=sup_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item())
        return out_dict, log_dict

    
    def train(self):

        assert self.args.training_stage == 0

        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.call_hook("before_run")

        if self.start_epoch > 0:
            if self.ckpt_rng_states is not None:
                print("loaded rng state for epoch", self.start_epoch)
                load_rng_states(self.ckpt_rng_states)
                self.ckpt_rng_states = None
            
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            ################# for pseudo-label learning:
            # eval set is (part of ) train set - if train set reach 100% acc, stop training
            if self.best_eval_acc > 0.9999:
                print("fullysupervised train loop early stopping due to reaching 100 acc:", self.best_eval_acc)
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict['train_lb']:

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

## utils for rng
import torch
import numpy as np
import random

def load_rng_states(checkpoint_rng_state):
    random.setstate(checkpoint_rng_state["python"])
    np.random.set_state(checkpoint_rng_state["numpy"])
    torch.random.set_rng_state(checkpoint_rng_state["cpu"])

    if torch.cuda.is_available():
        # torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
        torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
        # try:
            # torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])

        # except Exception as e:
        #     print(
        #         f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
        #         "\nThis won't yield the same results as if the training had not been interrupted."
        #     )


ALGORITHMS['supervised'] = FullySupervised