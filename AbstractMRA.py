from typing import Type

import gymnasium as gym
import numpy as np
import torch
from MyModels import EpisodicMemUHN, WorkingMem
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from torch import nn
from torch.nn import functional as F


class EpisodicMemWrapper(nn.Module):
    def __init__(self, mem_module: Type[EpisodicMemUHN], mem_config: dict) -> None:
        super().__init__()
        self.mem = mem_module(**mem_config)

    def forward(
        self, x_t: torch.Tensor, h_t_prev: torch.Tensor, mem_state: list[torch.Tensor]
    ) -> torch.Tensor:
        keys, values, indices = mem_state
        return self.mem(torch.concat((x_t, h_t_prev), dim=2), keys, values)

    def write(
        self, x_t: torch.Tensor, h_t: torch.Tensor, mem_state: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        key_and_value = torch.concat((x_t, h_t), dim=-1)
        keys, values, indices = mem_state
        keys, values, indices = self.mem.write(
            key_and_value, h_t, keys, values, indices
        )
        return [keys, values, indices]

    def get_memory(self):
        return self.mem.get_memory()


class WorkingMemWrapper(nn.Module):
    def __init__(
        self, working_mem_module: Type[WorkingMem], working_mem_config: dict
    ) -> None:
        super().__init__()
        self.working_mem = working_mem_module(**working_mem_config)

    def forward(
        self, inputs: torch.Tensor, state: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        true_state = [s.transpose(0, 1) for s in state]
        RNN_outputs, actions, values, true_out_state = self.working_mem(
            inputs, true_state
        )
        out_state = [s.transpose(0, 1) for s in true_out_state]
        return RNN_outputs, actions, values, out_state

    def RNN_dim(self) -> int:
        return self.working_mem.RNN_dim()

    def get_initial_states(self) -> list[torch.Tensor]:
        return self.working_mem.get_initial_states()


class CPC_loss(nn.Module):
    def __init__(self, num_cpc_steps: int, feature_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty([num_cpc_steps, feature_size, hidden_size])
        )
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, features: torch.Tensor, hiddens: torch.Tensor) -> torch.Tensor:
        xwk = torch.einsum(
            "bti,kij,buj->bkut", features, self.weight, hiddens
        )  # Compute xwb[b,k,u,t]=X[b,t]@W[k]@H[b,u]
        xwk = -torch.log_softmax(xwk, dim=-1)
        mask = torch.zeros_like(xwk)
        B, S, T, T2 = mask.shape
        mask[
            [
                (b, k, t, t + k + 1)
                for b in range(B)
                for k in range(S)
                for t in range(T)
                if t + k + 1 < T2
            ]
        ] = 1
        xwk = xwk * mask
        xwk = xwk.sum(dim=(-1, -2)) / torch.clamp(mask.sum(dim=(-1, -2)), min=1)
        # first xwk.sum(dim=-1) reduce last dimension so that xwk[k,k,u]=-log(exp(X[b,u+k]@W[k]@H[b,u])/sum_t exp(exp(X[b,t]@W[k]@H[b,u])) when u+k<T2 and 0 otherwise
        # second xwk.sum(dim=-2)/mask.sum(dim=(-1,-2)) does a mean over the non-zero values
        xwk = xwk.sum(dim=-1)  # sum over all CPC steps
        xwk = xwk.mean()  # mean over batches
        return xwk


class AbstractMRA(RecurrentNetwork, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        feature_net: Type[nn.Module],
        feature_net_config: dict,
        mem_net: Type[EpisodicMemWrapper],
        mem_net_config: dict,
        working_mem_net: Type[WorkingMemWrapper],
        working_mem_net_config: dict,
        CPC_loss_config: dict,
    ) -> None:
        RecurrentNetwork.__init__(
            self,
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
        )
        nn.Module.__init__(self)
        self.feature_net = feature_net(**feature_net_config)
        self.mem_net = mem_net(**mem_net_config)
        self.working_mem_net = working_mem_net(**working_mem_net_config)
        self._values = None
        self.cpc_loss = CPC_loss(**CPC_loss_config)
        self.mem_state_len = len(self.mem_net.get_memory())

    def get_initial_state(self):
        states = []
        states.extend(self.mem_net.get_memory())  # keys and values for memory
        states.extend(
            self.working_mem_net.get_initial_states()
        )  # e.g. returns h and c for LSTM, set at 0
        return states

    def value_function(self):
        if self._values is None:
            raise RuntimeError("must call forward() first")
        return self._values.squeeze(-1).view(-1)  # Size (bat)

    def forward_rnn(self, inputs, state, seq_lens):
        original_obs = restore_original_dimensions(
            inputs, self.obs_space, tensorlib="torch"
        )
        mem_state = state[: self.mem_state_len]
        wm_state = state[self.mem_state_len :]
        picture_obs = original_obs[
            "RGB_INTERLEAVED"
        ]  # Size (batch_size, time, height, width, n_channels)
        batch_size = picture_obs.shape[0]
        time = picture_obs.shape[1]
        picture_obs4D = picture_obs.permute((0, 1, 4, 2, 3)).flatten(
            end_dim=-4
        )  # Size (batch_size * time, n_channels, height, width), Conv2D only supports 4D tensors
        x_flattened = self.feature_net(
            picture_obs4D
        )  # Size (batch_size * time, n_features)
        x_stacked = x_flattened.view(
            (batch_size, time, -1)
        )  # Size (batch_size, time, n_features)
        h_t_prev = wm_state[
            0
        ]  # TODO fix this hack # Size (batch_size, 1, working_mem_features)
        stacked_shape = list(h_t_prev.shape)
        stacked_shape[-2] = 0
        batch_size = stacked_shape[0]
        stacked_h_t = torch.zeros(
            stacked_shape
        )  # Size (batch_size, 0, working_mem_features)
        stacked_values = torch.zeros((batch_size, 0, 1))  # Size (batch_size, 0, 1)
        stacked_actions = torch.zeros(
            (batch_size, 0, self.num_outputs)
        )  # Size (batch_size, 0, num_outputs)
        for i in range(time):
            x_t = x_stacked[..., [i], :]
            m_t = self.mem_net(
                x_t, h_t_prev, mem_state
            )  # Size (batch_size, time, memory_features)
            (
                h_t,  # size (batch_size, 1, working_mem_features)
                actions,  # size (batch_size, 1, num_outputs)
                values,  # size (batch_size, 1, 1)
                wm_state,  # same size as state
            ) = self.working_mem_net(torch.concat((x_t, m_t), dim=-1), wm_state)
            mem_state = self.mem_net.write(x_t, h_t, mem_state)
            h_t_prev = h_t
            stacked_h_t = torch.concat([stacked_h_t, h_t], dim=-2)
            stacked_values = torch.concat([stacked_values, values], dim=-2)
            stacked_actions = torch.concat([stacked_actions, actions], dim=-2)
        self._values = stacked_values  # Size (batch_size, time, 1)
        if self.training:
            self.x_t = x_t  # size (batch_size, time, n_features)
            self.h_t = stacked_h_t  # size (batch_size, time, working_mem_features)
        return stacked_actions, [*mem_state, *wm_state]

    def custom_loss(self, policy_loss, loss_inputs):
        # state_in of size len(seq_lens)
        # state_out of size len(seq_lens)*max_seq_len
        # problems if rollout_fragment_length % max_seq_len != 0, will fill with zeros
        # data does not requires grad
        # actors aren't reset between seqs, nor between rollouts
        # batchsize is rounded to max_seq_lens multiples

        cpc_loss = self.cpc_loss(self.x_t, self.h_t)
        # cpc_loss = torch.Tensor([0])

        self.cpc_loss_metric = cpc_loss.item()
        self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

        return [loss_ + cpc_loss for loss_ in policy_loss]

    def metrics(self):
        return {
            "policy_loss": self.policy_loss_metric,
            "cpc_loss": self.cpc_loss_metric,
        }
