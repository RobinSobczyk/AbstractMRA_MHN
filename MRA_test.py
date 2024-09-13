import AbstractMRA
import dmm2gym
import MyModels
import ray
import torch
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.tune.registry import register_env


def MRA_run():
    torch.autograd.set_detect_anomaly(True)  # better debugging on backward
    context = ray.init()
    print(context.dashboard_url)  # allows tracking from browser dashboard

    # allow dmm to be accessed from ray rl lib
    def env_creator(env_config):
        return dmm2gym.DmmPaperWrapper(**env_config)

    register_env("my_dmm_env", env_creator)

    # hyperparameters for MRA
    cnn_out_dim = 256
    hidden_dim = 256
    num_slots = 20
    temp = 1
    proj_dim = hidden_dim
    num_actions = 8
    key_dim = cnn_out_dim + hidden_dim
    CPC_steps = 3

    # Configure IMPALA with MRA
    impala_config = ImpalaConfig()
    impala_config = impala_config.resources(num_gpus=0)
    impala_config = impala_config.env_runners(
        num_gpus_per_env_runner=0, num_env_runners=1, rollout_fragment_length=60
    )
    impala_config = impala_config.learners(num_gpus_per_learner=0)
    impala_config = impala_config.framework(framework="torch")
    impala_config.model = {
        "custom_model": AbstractMRA.AbstractMRA,
        "custom_model_config": {
            "feature_net": MyModels.ConvNet,
            "feature_net_config": {},
            "mem_net": AbstractMRA.EpisodicMemWrapper,
            "mem_net_config": {
                "mem_module": MyModels.EpisodicMemUHN,
                "mem_config": {
                    "similarity_module": MyModels.MHNSimilarity,
                    "sim_args": {
                        "query_dim": cnn_out_dim + hidden_dim,
                        "key_dim": key_dim,
                    },
                    "separation_module": MyModels.MHNSeparation,
                    "sep_args": {
                        "temp": temp,
                    },
                    "projection_module": MyModels.MHNProjection,
                    "proj_args": {
                        "value_dim": hidden_dim,
                        "proj_dim": proj_dim,
                    },
                    "key_dim": key_dim,
                    "value_dim": hidden_dim,
                    "has_slots": True,
                    "mem_size": num_slots,
                },
            },
            "working_mem_net": AbstractMRA.WorkingMemWrapper,
            "working_mem_net_config": {
                "working_mem_module": MyModels.WorkingLSTMMem,
                "working_mem_config": {
                    "input_dim": cnn_out_dim + proj_dim,
                    "hidden_dim": hidden_dim,
                    "action_dim": num_actions,
                    "additional_RNN_args": {},
                },
            },
            "CPC_loss_config": {
                "num_cpc_steps": CPC_steps,
                "feature_size": cnn_out_dim,
                "hidden_size": hidden_dim,
            },
        },
        "max_seq_len": 20,
    }
    impala_config = impala_config.environment(
        env="my_dmm_env",
        env_config={
            "seed": 123,
            "level_name": "spot_diff_train",
        },
    )
    impala_config = impala_config.reporting(min_time_s_per_iteration=500)
    impala_config = impala_config.training(train_batch_size=600)
    # Instanciate
    algo = impala_config.build()
    # Train once
    algo.train()


if __name__ == "__main__":
    MRA_run()
