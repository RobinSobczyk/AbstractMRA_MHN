from typing import Any, Optional, SupportsFloat

import cv2
import dm_env.specs
import dm_memorytasks as dmm
import gymnasium as gym
import numpy as np


def convert_dmm_spec_to_gym_space(spec):
    if isinstance(spec, dict):
        spec = {k: convert_dmm_spec_to_gym_space(item) for k, item in spec.items()}
        return gym.spaces.Dict(spec)
    elif isinstance(spec, dm_env.specs.DiscreteArray):
        num_values = spec.num_values
        return gym.spaces.Discrete(n=num_values)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        dtype = spec.dtype
        shape = spec.shape
        return gym.spaces.Box(
            shape=shape,
            low=spec.minimum,
            high=spec.maximum,
            dtype=dtype,
        )
    elif isinstance(spec, dm_env.specs.Array):
        shape = spec.shape
        dtype = spec.dtype
        if np.issubdtype(dtype, np.floating):
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=dtype,
            )
        elif np.issubdtype(dtype, np.integer):
            return gym.spaces.Box(
                low=np.iinfo(dtype).min,
                high=np.iinfo(dtype).max,
                shape=shape,
                dtype=dtype,
            )
        else:
            raise NotImplementedError(
                f"No bounds found for unbounded array with type {dtype}"
            )
    else:
        raise NotImplementedError(type(spec))


def convert_dmm_spec_to_gym_space_override_for_action(spec):
    if isinstance(spec, dict):
        spec = {
            k: convert_dmm_spec_to_gym_space_override_for_action(item)
            for k, item in spec.items()
        }
        return gym.spaces.Dict(spec)
    elif isinstance(spec, dm_env.specs.DiscreteArray):
        num_values = spec.num_values
        return gym.spaces.Discrete(n=num_values)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        dtype = spec.dtype
        shape = spec.shape
        return gym.spaces.Box(
            shape=shape,
            low=-1,
            high=1,
            dtype=dtype,
        )
    elif isinstance(spec, dm_env.specs.Array):
        shape = spec.shape
        dtype = spec.dtype
        if np.issubdtype(dtype, np.floating):
            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=dtype,
            )
        elif np.issubdtype(dtype, np.integer):
            return gym.spaces.Box(
                low=np.iinfo(dtype).min,
                high=np.iinfo(dtype).max,
                shape=shape,
                dtype=dtype,
            )
        else:
            raise NotImplementedError(
                f"No bounds found for unbounded array with type {dtype}"
            )
    else:
        raise NotImplementedError(type(spec))


class ImageViewer:
    def __init__(self, window_name):
        self._window_name = window_name
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)

    def imshow(self, img):
        cv2.imshow(self._window_name, img[:, :, [2, 1, 0]])
        cv2.waitKey(0)

    def close(self):
        cv2.destroyWindow(self._window_name)


class DmmRawWrapper(gym.Env):
    def __init__(
        self,
        seed: int,
        level_name: str,
        from_docker: bool = True,
        path: Optional[str] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self._np_random = np.random.default_rng(seed)
        self._env: dmm._load_environment._MemoryTasksEnv
        self.level_name = level_name
        settings = dmm.EnvironmentSettings(seed, level_name, **kwargs)
        if from_docker:
            self._env = dmm.load_from_docker(settings)
        else:
            self._env = dmm.load_from_disk(path, settings)
        self.action_space = convert_dmm_spec_to_gym_space_override_for_action(
            self._env.action_spec()
        )
        self.observation_space = convert_dmm_spec_to_gym_space(
            self._env.observation_spec()
        )
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {"render_modes": [None, "human", "rgb_array"]}

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        timestep = self._env.step(action)
        if timestep is not None:
            observation = timestep.observation
            reward = timestep.reward
            done = timestep.last()
            truncated = False
            info = {}
            self.last_step = timestep
            return observation, reward, done, truncated, info
        else:
            raise TypeError(
                f"Env step returns None, possibly because env._num_actions_repeat is set to 0, current value : {self._env._num_action_repeats}"
            )

    def reset(self, seed: Optional[int] = None, **kwargs) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        timestep = self._env.reset()
        self.last_step = timestep
        return timestep.observation, {}

    def render(self):
        img = self.last_step.observation["RGB_INTERLEAVED"]
        if self.render_mode is None:
            return None
        elif self.render_mode == "human":
            if self.viewer is None:
                self.viewer = ImageViewer(self.level_name)
            self.viewer.imshow(img)
            return None
        elif self.render_mode == "rgb_array":
            return img
        else:
            raise NotImplementedError(
                f"No rendering is implemented for render mode {self.render_mode}"
            )

    def close(self):
        self._env.close()
        if self.viewer is not None:
            self.viewer.close()
        return super().close()


# move forward, move backward, strafe left, strafe right, look left, look right, look left while moving forward, look right while moving forward
def make_env_action(tab):
    env_action = {}
    env_action["LOOK_DOWN_UP"] = tab[0]  # 1 is Down
    env_action["LOOK_LEFT_RIGHT"] = tab[1]  # 1 is Right
    env_action["MOVE_BACK_FORWARD"] = tab[2]  # 1 is Forward
    env_action["STRAFE_LEFT_RIGHT"] = tab[3]  # 1 is Right
    return env_action


class DmmPaperWrapper(DmmRawWrapper):
    def __init__(
        self,
        seed: int,
        level_name: str,
        from_docker: bool = True,
        path: Optional[str] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(seed, level_name, from_docker, path, render_mode, **kwargs)
        self.action_space = gym.spaces.Discrete(n=8)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        action_id_to_env_actions = {
            0: [0, 0, 1, 0],  # move forward
            1: [0, 0, -1, 0],  # move backward
            2: [0, 0, 0, -1],  # strafe left
            3: [0, 0, 0, 1],  # strafe right
            4: [0, -1, 0, 0],  # look left
            5: [0, 1, 0, 0],  # look right
            6: [0, -1, 1, 0],  # look left while moving forward
            7: [0, 1, 1, 0],  # look right while moving forward
        }
        env_action = make_env_action(action_id_to_env_actions[np.argmax(action)])
        timestep = self._env.step(env_action)
        if timestep is not None:
            observation = timestep.observation
            reward = timestep.reward
            done = timestep.last()
            truncated = False
            info = {}
            self.last_step = timestep
            return observation, reward, done, truncated, info
        else:
            raise TypeError(
                f"Env step returns None, possibly because env._num_actions_repeat is set to 0, current value : {self._env._num_action_repeats}"
            )
