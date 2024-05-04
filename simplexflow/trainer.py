"""
Here we define the trainer class (pytorch lightning trainer) for the simplexflow models.
"""

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from tqdm import tqdm

import lightning.pytorch as pl

from simplexflow.models import ContextUnet

PI = 3.141592653589


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class MnistTrainer(pl.LightningModule):
    """
    Trainer module for the MNIST model.
    """

    def __init__(self, hidden_dim=128, num_bins=4, nb_time_steps=100):
        """
        Args:
            hidden_dim (int): hidden dimension of the model
            num_bins (int): number of bins to discretize the data into
            nb_block (int): number of blocks in the model
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_bins = num_bins

        self.nb_time_steps = nb_time_steps

        # create the model
        self.model = ContextUnet(
            in_channels=num_bins,
            n_feat=hidden_dim,
            n_cfeat=10,
            height=28,
            nb_class=num_bins,
        )

        # create the loss function
        self.loss = nn.CrossEntropyLoss()

        self.apply(init_weights)

    def forward(self, data, t):
        """
        Forward pass of the model.
        """
        result_logit = self.model(data, t)

        return result_logit

    def compute_loss(self, logits, data, init_data):
        """
        Computes the loss.
        """
        return

    def compute_params_from_t(self, t):
        # we generate alpha_t = 1 - cos2(t * pi/2)
        alpha_t = 1 - torch.cos(t * PI / 2) ** 2
        alpha_t_dt = PI * torch.cos(PI / 2 * t) * torch.sin(PI / 2 * t)

        w_t = alpha_t_dt / (1 - alpha_t)

        # make w_t min of 0.005 and max of 1.5
        w_t = torch.clamp(w_t, min=0.005, max=1.5)

        return w_t, alpha_t, alpha_t_dt

    def training_step(self, batch, _):
        """
        Training step.
        """

        # we get the data from the batch
        data_onehot, data, labels = batch

        batch_size = data_onehot.shape[0]
        img_w = data_onehot.shape[1]

        # now we need to select a random time step between 0 and 1 for all the batch
        t = torch.rand(batch_size, 1).float()
        t = t.to(self.device)

        w_t, alpha_t, alpha_t_dt = self.compute_params_from_t(t)

        # we generate the prior dataset (gaussian noise)
        prior = torch.randn(batch_size, img_w, img_w, self.num_bins).to(self.device)
        prior = F.softmax(prior, dim=-1)

        alpha_t = alpha_t.unsqueeze(2).unsqueeze(3)

        gt = (1 - alpha_t) * prior + alpha_t * data_onehot

        result_logit = self.model(gt, t)

        loss = F.cross_entropy(result_logit, data, reduction="none")

        loss = loss * w_t.unsqueeze(2)
        loss = torch.mean(loss)

        return loss

    # on training end
    def on_train_epoch_end(self):
        # we should generate some images
        self.eval()
        with torch.no_grad():
            self.generate()
        self.train()

    def generate(self):
        """
        Method to generate some images.
        """
        # init the prior
        prior_t = torch.randn(1, 28, 28, self.num_bins).to(self.device)

        # softmax the prior
        prior_t = F.softmax(prior_t, dim=-1)

        for i in range(self.nb_time_steps):
            t = torch.ones((1, 1)).to(self.device)
            t = t * i / self.nb_time_steps

            w_t, alpha_t, alpha_t_dt = self.compute_params_from_t(t)

            g1_estimation = self.model(prior_t, t)

            # apply softmax to the logits
            g1_estimation = F.softmax(g1_estimation, dim=1)

            u_theta = w_t.unsqueeze(2).unsqueeze(3) * (g1_estimation.permute(0, 2, 3, 1) - prior_t)

            prior_t = prior_t + u_theta * 1 / self.nb_time_steps

        result_pred = torch.argmax(prior_t, dim=-1)

        self.save_image(result_pred, 100)

    def save_image(self, data, i):
        """
        Saves the image.
        """
        # plot the data
        plt.imshow(data.squeeze().cpu().numpy(), cmap="gray")

        # title
        plt.title(f"data = {i}")

        # save the figure
        plt.savefig(f"/teamspace/studios/this_studio/data_{i}.png")

        # close the figure
        plt.close()

    def configure_optimizers(self):
        """
        Configure the optimizer.
        """
        # create the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer
