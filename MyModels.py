from typing import Optional, Type

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# region ConvNet


class ResidualBlock(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(3, 3),
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=(3, 3),
            padding=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.relu(input)
        output = self.conv1(output)
        output = F.relu(output)
        output = self.conv2(output)
        return input + output


class BigResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.residual = ResidualBlock(out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.maxpool(output)
        output = self.residual(output)
        return output


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.residual1 = BigResidualBlock(3, 16)
        self.residual2 = BigResidualBlock(16, 32)
        self.residual3 = BigResidualBlock(32, 32)
        self.fc = nn.LazyLinear(out_features=256)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.residual1(input)
        output = self.residual2(output)
        output = self.residual3(output)
        output = F.relu(output)
        output = output.flatten(start_dim=-3)
        output = self.fc(output)
        output = F.relu(output)
        return output


# endregion

# region WorkingMem


class WorkingMem(nn.Module):
    def __init__(
        self,
        RNN_layer: Type[nn.RNNBase],
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        additional_RNN_args: dict = {},
    ) -> None:
        super().__init__()
        self.action_lin = nn.Linear(in_features=hidden_dim, out_features=action_dim)
        self.value_lin = nn.Linear(in_features=hidden_dim, out_features=1)
        self.RNN = RNN_layer(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            **additional_RNN_args
        )

    def forward(
        self, inputs: torch.Tensor, state: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        if len(state) == 1:
            RNN_outputs, *out_state = self.RNN(inputs, state[0])
        else:
            RNN_outputs, out_state = self.RNN(inputs, state)
            out_state = list(out_state)
        actions = self.action_lin(RNN_outputs)
        value = self.value_lin(RNN_outputs)
        return RNN_outputs, actions, value, out_state

    def RNN_dim(self) -> int:
        return self.RNN.hidden_size

    def get_initial_states(self) -> list[torch.Tensor]:
        # TODO later, for modularity
        raise NotImplementedError


class WorkingLSTMMem(WorkingMem):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        additional_RNN_args: dict = {},
    ) -> None:
        super().__init__(
            RNN_layer=nn.LSTM,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            additional_RNN_args=additional_RNN_args,
        )
        self.RNN: nn.LSTM

    def get_initial_states(self) -> list[torch.Tensor]:
        dummy_input = torch.zeros((1, 1, self.RNN.input_size))
        h_0 = torch.zeros(
            self.RNN.get_expected_hidden_size(input=dummy_input, batch_sizes=None)
        ).squeeze(1)
        c_0 = torch.zeros(
            self.RNN.get_expected_cell_size(input=dummy_input, batch_sizes=None)
        ).squeeze(1)
        return [h_0, c_0]


# endregion

# region EpisodicMem


class LocalEpisodicMemUHN(nn.Module):
    def __init__(
        self,
        similarity_module: Type[nn.Module],
        sim_args: dict,
        separation_module: Type[nn.Module],
        sep_args: dict,
        projection_module: Type[nn.Module],
        proj_args: dict,
        key_dim: int,
        value_dim: int,
        has_slots: bool,
        mem_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sim = similarity_module(**sim_args)
        self.sep = separation_module(**sep_args)
        self.proj = projection_module(**proj_args)
        self.has_slots = has_slots
        if has_slots:
            if mem_size is None:
                raise ValueError("mem_size is set to None while an int was expected")
            else:
                self.mem_size = mem_size
                self.keys = nn.Parameter(
                    torch.zeros((mem_size, key_dim)), requires_grad=False
                )
                self.values = nn.Parameter(
                    torch.zeros((mem_size, value_dim)), requires_grad=False
                )
        else:
            self.mem_size = 0
            self.keys = nn.Parameter(torch.zeros((0, key_dim)), requires_grad=False)
            self.values = nn.Parameter(torch.zeros((0, value_dim)), requires_grad=False)
        self.unprocessed_seps = []
        self.last_written_slot = -1

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if self.mem_size == 0:
            keys = torch.zeros((0, self.keys.shape[1]))
            values = torch.zeros((0, self.values.shape[1]))
        else:
            keys = self.keys
            values = self.values
        sim = self.sim(query, keys)
        sep: torch.Tensor = self.sep(sim)
        self.unprocessed_seps.append(sep.detach())
        proj = sep @ self.proj(values)
        return proj

    def mem_writing_index(self) -> int:
        # TODO implement variants
        self.last_written_slot = (self.last_written_slot + 1) % self.mem_size
        return self.last_written_slot

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if self.has_slots:
            i = self.mem_writing_index()
            self.keys[..., i, :] = key.detach()
            self.values[..., i, :] = value.detach()
        else:
            self.keys.data = torch.concat((self.keys, key.unsqueeze(0)))
            self.values.data = torch.concat((self.values, value.unsqueeze(0)))
            self.mem_size += 1

    def get_memory(self) -> list[torch.Tensor]:
        return [self.keys, self.values]


class EpisodicMemUHN(nn.Module):
    def __init__(
        self,
        similarity_module: Type[nn.Module],
        sim_args: dict,
        separation_module: Type[nn.Module],
        sep_args: dict,
        projection_module: Type[nn.Module],
        proj_args: dict,
        key_dim: int,
        value_dim: int,
        has_slots: bool,
        mem_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sim = similarity_module(**sim_args)
        self.sep = separation_module(**sep_args)
        self.proj = projection_module(**proj_args)
        self.has_slots = has_slots
        # TODO remove local storage due to parametrization of memory
        if has_slots:
            if mem_size is None:
                raise ValueError("mem_size is set to None while an int was expected")
            else:
                self.mem_size = mem_size
                self.keys = nn.Parameter(
                    torch.zeros((mem_size, key_dim)), requires_grad=False
                )
                self.values = nn.Parameter(
                    torch.zeros((mem_size, value_dim)), requires_grad=False
                )
        else:
            self.mem_size = 0
            self.keys = nn.Parameter(torch.zeros((0, key_dim)), requires_grad=False)
            self.values = nn.Parameter(torch.zeros((0, value_dim)), requires_grad=False)
        self.unprocessed_seps = []

    def forward(self, query: torch.Tensor, keys, values) -> torch.Tensor:
        # TODO fix memory not working as a parameter
        sim = self.sim(query, keys)
        sep: torch.Tensor = self.sep(sim)
        # self.unprocessed_seps.append(sep.detach()) # TODO release when variants of writing index are implemented
        proj = sep @ self.proj(values)
        return proj

    def mem_writing_index(self, indices: torch.Tensor) -> torch.Tensor:
        # TODO implement variants
        return (indices + 1) % self.mem_size

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO fix indices not being overridable
        indices = indices.int()
        keys = torch.clone(keys)  # writing in place breaks backward pass
        values = torch.clone(values)  # writing in place breaks backward pass
        if self.has_slots:
            indices = self.mem_writing_index(indices)
            id_helper = torch.arange(0, indices.shape.numel())
            keys[id_helper, indices, :] = key.detach().squeeze()
            values[id_helper, indices, :] = value.detach().squeeze()
        else:
            keys = torch.concat((self.keys, key.unsqueeze(0)))
            values = torch.concat((self.values, value.unsqueeze(0)))
            self.mem_size += 1
        return keys, values, indices

    def get_memory(self) -> list[torch.Tensor]:
        return [
            self.keys,
            self.values,
            torch.Tensor([-1]),
        ]  # rajouter le last writing index ici


class LinearEncodeEpMemUHN(EpisodicMemUHN):
    def __init__(
        self,
        similarity_module: Type[nn.Module],
        sim_args: dict,
        separation_module: Type[nn.Module],
        sep_args: dict,
        projection_module: Type[nn.Module],
        proj_args: dict,
        key_dim: int,
        hidden_key_dim: int,
        value_dim: int,
        hidden_value_dim: int,
        recompute_memory: bool,
        has_slots: bool,
        mem_size: Optional[int] = None,
    ) -> None:
        if recompute_memory:
            super().__init__(
                similarity_module,
                sim_args,
                separation_module,
                sep_args,
                projection_module,
                proj_args,
                key_dim,
                value_dim,
                has_slots,
                mem_size,
            )
        else:
            super().__init__(
                similarity_module,
                sim_args,
                separation_module,
                sep_args,
                projection_module,
                proj_args,
                hidden_key_dim,
                hidden_value_dim,
                has_slots,
                mem_size,
            )
        self.key_lin = nn.Linear(key_dim, hidden_key_dim)
        self.value_lin = nn.Linear(value_dim, hidden_value_dim)
        self.recompute = recompute_memory

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if self.mem_size == 0:
            keys = torch.zeros((0, self.keys.shape[1]))
            values = torch.zeros((0, self.values.shape[1]))
        else:
            keys = self.keys
            values = self.values
        if self.recompute:
            keys = self.key_lin(keys)
            values = self.value_lin(values)
        sim = self.sim(query, keys)
        sep = self.sep(sim)
        # self.unprocessed_seps.append(sep.detach()) # TODO release when variants of writing index are implemented
        proj = sep @ self.proj(values)
        return proj

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.recompute:
            return super().write(key.detach(), value.detach(), keys, values, indices)
        else:
            return super().write(
                self.key_lin(key.detach()),
                self.value_lin(value.detach()),
                keys,
                values,
                indices,
            )


class MHNSimilarity(nn.Module):
    def __init__(self, query_dim: int, key_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty([query_dim, key_dim]))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "...ij,jk,...kl->...il", query, self.weight, torch.transpose(keys, -1, -2)
        )


class MHNSeparation(nn.Module):
    def __init__(self, temp: float) -> None:
        super().__init__()
        self.temp = temp

    def forward(self, sim: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.temp * sim, dim=-1)


class MHNProjection(nn.Module):
    def __init__(self, value_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(value_dim, proj_dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.lin(values)


# endregion
