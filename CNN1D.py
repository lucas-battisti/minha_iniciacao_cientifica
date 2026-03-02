# %% libraries

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn

import lightning as L
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

import pandas as pd
from sklearn.model_selection import train_test_split
import dt

# %% config
config = {
    "p": 1,

    "channels_1": 32,
    "kernel_1": 3,
    "maxpool_1_kernel": 2,

    "channels_2": 64,
    "kernel_2": 2,
    "maxpool_2_kernel": 2,

    "layer_1_size": 144,
    "layer_2_size": 144,

    "activation_function": nn.ReLU(),

    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 0
}


# %% DataTensor

class data(Dataset):
    def __init__(self, x: pd.DataFrame, e: pd.DataFrame, z: pd.DataFrame,
                 super_am=False, p=1):

        self.z = torch.tensor(z.values)

        if super_am:
            self.f = torch.tensor(x.fillna(0).values)
            self.e = torch.tensor(e.fillna(0).values)
            self.f = self.f.repeat(p, 1)
            self.e = self.e.repeat(p, 1)
            self.z = self.z.repeat(p, 1)
            self.cov = torch.normal(self.f, self.e).type(torch.float32)
        else:
            self.cov = torch.cat((torch.tensor(x.fillna(0).values),
                                  torch.tensor(e.fillna(0).values)), dim=1)

    def __getitem__(self, idx):
        return self.cov[idx], self.z[idx]

    def __len__(self):
        return len(self.z)


# %% 1D
def make_tensors(v_x: np.array, v_e: np.array,
                 version=['no', 'with', 'stack'],
                 super_am=False):
    if super_am and version != 'no':
        raise ValueError("super and no")

    n = len(v_x)
    p = len(v_x[0])

    if version == 'no':
        return torch.tensor(v_x.reshape((n, 1, p)))

    matriz = np.array([])

    for i in range(n):
        matriz = np.append(matriz, [v_x[i], v_e[i]])

    if version == 'with':
        return torch.tensor(matriz.reshape((n, 1, 2 * p)))
    if version == 'stack':
        return torch.tensor(matriz.reshape((n, 2, p)))


# %% dataclass
class data(Dataset):
    def __init__(self, x: pd.DataFrame, e: pd.DataFrame, z: pd.DataFrame,
                 version=['no', 'with', 'stack'],
                 super_am=False, p=1):

        if super_am and version != 'no':
            raise ValueError("super and not 'no'")

        self.z = torch.tensor(z.values)

        v_x = np.array(x.fillna(0))
        v_e = np.array(e.fillna(0))

        if super_am:
            self.z = self.z.repeat(p, 1)
            v_x = np.repeat(v_x, p, axis=0)
            v_e = np.repeat(v_e, p, axis=0)

            v_x = np.random.normal(v_x, v_e)

        self._1d = make_tensors(v_x, v_e, version=version)

        self._1d = self._1d.to(torch.float)

    def __getitem__(self, idx):
        return self._1d[idx], self.z[idx]

    def __len__(self):
        return len(self.z)


# %% DataModule

class DataModule(L.LightningDataModule):
    def __init__(self, x: pd.DataFrame, e: pd.DataFrame, z: pd.DataFrame,
                 set_size=[0.5, 0.25, 0.25], seed=2023,
                 config=None, version=['no', 'with', 'stack'], super_am=False):
        super().__init__()

        if config == None:
            config = {
                "p": 1,

                "channels_1": 32,
                "kernel_1": 3,
                "maxpool_1_kernel": 2,

                "channels_2": 64,
                "kernel_2": 2,
                "maxpool_2_kernel": 2,

                "layer_1_size": 144,
                "layer_2_size": 144,

                "activation_function": nn.ReLU(),

                "batch_size": 32,
                "lr": 1e-3,
                "weight_decay": 0
            }

        self.p = config["p"]
        self.batch_size = config["batch_size"]

        x_train, x_val = train_test_split(x, test_size=(1 - set_size[0]),
                                          random_state=seed)
        e_train, e_val = train_test_split(e, test_size=(1 - set_size[0]),
                                          random_state=seed)
        z_train, z_val = train_test_split(z, test_size=(1 - set_size[0]),
                                          random_state=seed)

        if len(set_size) == 3:
            x_val, x_test = train_test_split(x_val, test_size=set_size[2] / (1 - set_size[0]),
                                             random_state=seed)
            e_val, e_test = train_test_split(e_val, test_size=set_size[2] / (1 - set_size[0]),
                                             random_state=seed)
            z_val, z_test = train_test_split(z_val, test_size=set_size[2] / (1 - set_size[0]),
                                             random_state=seed)
        else:
            x_test, e_test, z_test = x_val, e_val, z_val

        self.train = data(x_train, e_train, z_train, version=version, super_am=super_am, p=self.p)
        self.val = data(x_val, e_val, z_val)
        self.test = data(x_test, e_test, z_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=True)

    # %% Module


class Module(L.LightningModule):
    def __init__(self, config=None,
                 loss_func=nn.MSELoss()):
        super().__init__()
        if config == None:
            config = {
                "p": 1,

                "channels_1": 32,
                "kernel_1": 3,
                "maxpool_1_kernel": 2,

                "channels_2": 64,
                "kernel_2": 2,
                "maxpool_2_kernel": 2,

                "layer_1_size": 144,
                "layer_2_size": 144,

                "activation_function": nn.ReLU(),

                "batch_size": 32,
                "lr": 1e-3,
                "weight_decay": 0
            }
        self.loss_func = loss_func

        self.channels_1 = config["channels_1"]
        self.kernel_1 = config["kernel_1"]
        self.maxpool_1_kernel = config["maxpool_1_kernel"]

        self.channels_2 = config["channels_2"]
        self.kernel_2 = config["kernel_2"]
        self.maxpool_2_kernel = config["maxpool_2_kernel"]

        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.activation_function = config["activation_function"]

        self.batch_size = ["batch_size"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

        self.conv1 = nn.Conv1d(1, self.channels_1, self.kernel_1)
        self.maxpool1 = nn.MaxPool1d(self.maxpool_1_kernel)

        self.conv2 = nn.Conv1d(self.channels_1, self.channels_2, self.kernel_2)
        self.maxpool2 = nn.MaxPool1d(self.maxpool_2_kernel)

        self.l1 = torch.nn.Linear(1000000, self.layer_1_size)
        self.l2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.l3 = torch.nn.Linear(self.layer_2_size, 1)

        self.test_step_targets = []
        self.test_step_outputs = []
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.activation_function(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.activation_function(x)

        x = nn.Flatten(1, 2)(x)

        a = x.size()[1]
        self.l1.in_features = a

        x = self.l1(x)
        x = self.activation_function(x)

        x = self.l2(x)
        x = self.activation_function(x)

        x = self.l3(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.loss_func(output, target)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.loss_func(output, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.loss_func(output, target)
        pred = target.tolist()[0]
        pred_hat = output.tolist()[0]
        self.test_step_targets.extend(pred)
        self.test_step_outputs.extend(pred_hat)
        return loss

    def on_train_epoch_end(self):

        pred = self.test_step_targets
        pred_hat = self.test_step_outputs

        desv_var = np.var(np.array(pred)-np.array(pred_hat))
        self.log("desv_var", desv_var)

        self.test_step_targets = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer


# %% Train
'''
def train(x: pd.DataFrame, e: pd.DataFrame, z: pd.DataFrame, config=None,
          set_size=[0.5, 0.25, 0.25], seed=2023, super=False,
          loss_func=nn.MSELoss(),
          num_epochs=100, num_gpus=0):
    dm = DataModule(x, e, z, set_size, seed, config, super)
    model = Module(config, loss_func)

    trainer = L.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False)

    trainer = prepare_trainer(trainer)
    trainer.fit(model, dm)
'''