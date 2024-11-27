import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from open_pi0.models.pi0_gemma import Pi0GemmaProcessor


class CalvinH5Dataset(Dataset):
    def __init__(
        self,
        root_path: str,
        obs_seq_len: int = 2,
        action_seq_len: int = 50,
        use_relative_actions: bool = True,
        image_size: int = 224,
        repeat: int = 1,
        training: bool = True,
        preprocessor: Pi0GemmaProcessor = None,
        uncond_prob: float = 0.2,
    ):
        super().__init__()

        self.root_path = root_path
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.use_relative_actions = use_relative_actions
        self.image_size = image_size
        self.repeat = repeat
        self.training = training
        self.preprocessor = preprocessor
        self.uncond_prob = uncond_prob

        self.episode_padding_left = max(0, self.obs_seq_len - 1)
        self.episode_padding_right = max(0, self.action_seq_len - 1)

        self.transform = T.Compose(
            [
                T.Resize(image_size),
                T.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

        self._h5_fp: h5py.File | None = None
        self._ep_start_end_ids: list[tuple[int, int]] | None = None

    @property
    def h5_file(self):
        if self._h5_fp is None:
            self._h5_fp = h5py.File(self.root_path, "r")
        return self._h5_fp

    @property
    def ep_start_end_ids(self) -> list[tuple[int, int]]:
        if self._ep_start_end_ids is None:
            meta_group = self.h5_file["meta"]
            ep_start_end_ids: np.ndarray = meta_group["ep_start_end_ids"][:]
            frame_ids: np.ndarray = meta_group["frame_ids"][:]

            inv_frame_ids = {frame_id: i for i, frame_id in enumerate(frame_ids.tolist())}

            self._ep_start_end_ids = [
                (inv_frame_ids[start], inv_frame_ids[end] + 1)
                for start, end in ep_start_end_ids.tolist()
            ]
        return self._ep_start_end_ids

    def __len__(self):
        return len(self.ep_start_end_ids) * self.repeat

    def sample_horizon_indices(self, ep_start_idx: int, ep_end_idx: int, index: int) -> list[int]:
        episode_len = ep_end_idx - ep_start_idx
        horizon_len = self.obs_seq_len + self.action_seq_len - 1

        if horizon_len >= episode_len + self.episode_padding_left + self.episode_padding_right:
            start_idx = -self.episode_padding_left
        else:
            if self.training:
                start_idx = np.random.randint(
                    -self.episode_padding_left, episode_len + self.episode_padding_right - horizon_len + 1
                )
            else:
                # Use fixed seed for deterministic validation/test indices
                rng = np.random.default_rng(index)
                start_idx = rng.integers(
                    -self.episode_padding_left, episode_len + self.episode_padding_right - horizon_len + 1
                )

        indices = []
        for i in range(horizon_len):
            idx = min(max(0, start_idx + i), episode_len - 1)
            indices.append(ep_start_idx + idx)

        return indices

    def __getitem__(self, index: int):
        ep_index = index % len(self.ep_start_end_ids)

        if self.training and np.random.rand() < self.uncond_prob:
            uncond = True
        else:
            uncond = False

        if uncond:
            text = ""
        else:
            lang_group = self.h5_file["lang"]
            text_bytes: bytes = lang_group["text"][ep_index]
            text = text_bytes.decode("utf-8")

        ep_start_idx, ep_end_idx = self.ep_start_end_ids[ep_index]
        horizon_indices = self.sample_horizon_indices(ep_start_idx, ep_end_idx, index)

        action_group = self.h5_file["action"]
        actions_key = "rel_actions" if self.use_relative_actions else "actions"

        if uncond:
            action_dim = action_group[actions_key].shape[-1]
            actions = torch.zeros(self.action_seq_len, action_dim, dtype=torch.float32)
        else:
            action_indices = horizon_indices[self.obs_seq_len - 1:]
            action_indices_uniq, action_indices_inv = np.unique(action_indices, return_inverse=True)
            action_indices_uniq = action_indices_uniq.tolist()

            actions_array = action_group[actions_key][action_indices_uniq]
            actions = torch.from_numpy(actions_array).to(torch.float32)
            actions = actions[action_indices_inv]

        obs_indices = horizon_indices[:self.obs_seq_len]
        obs_indices_uniq, obs_indices_inv = np.unique(obs_indices, return_inverse=True)
        obs_indices_uniq = obs_indices_uniq.tolist()

        obs_group = self.h5_file["obs"]
        obs = {}

        if uncond:
            if self.preprocessor is None:
                rgb_static = torch.zeros(
                    self.obs_seq_len, 3, self.image_size, self.image_size, dtype=torch.float32
                )
                rgb_gripper = torch.zeros(
                    self.obs_seq_len, 3, self.image_size, self.image_size, dtype=torch.float32
                )
            else:
                rgb_static_shape = tuple(obs_group["rgb_static"].shape[1:])
                rgb_static = np.zeros((self.obs_seq_len,) + rgb_static_shape, dtype=np.uint8)
                rgb_gripper_shape = tuple(obs_group["rgb_gripper"].shape[1:])
                rgb_gripper = np.zeros((self.obs_seq_len,) + rgb_gripper_shape, dtype=np.uint8)

            robot_obs_dim = obs_group["robot_obs"].shape[-1]
            robot_obs = torch.zeros(self.obs_seq_len, robot_obs_dim, dtype=torch.float32)

            obs = {
                "rgb_static": rgb_static,
                "rgb_gripper": rgb_gripper,
                "robot_obs": robot_obs,
            }
        else:
            for obs_name in ["rgb_static", "rgb_gripper"]:
                obs_data = obs_group[obs_name][obs_indices_uniq]
                obs_data = obs_data[obs_indices_inv]

                if self.preprocessor is None:
                    obs_tensor = torch.from_numpy(obs_data).to(torch.float32)
                    obs_tensor.div_(255.0)
                    obs_tensor = obs_tensor.permute(0, 3, 1, 2)
                    obs_tensor = self.transform(obs_tensor)
                else:
                    obs_tensor = obs_data

                obs[obs_name] = obs_tensor

            robot_obs = obs_group["robot_obs"][obs_indices_uniq]
            robot_obs = robot_obs[obs_indices_inv]
            obs["robot_obs"] = torch.from_numpy(robot_obs).to(torch.float32)

        if self.preprocessor is not None:
            images = [x for x in obs["rgb_static"]]
            images += [x for x in obs["rgb_gripper"]]

            return self.preprocessor.prepare_for_traning_sample(
                images=images,
                instruction=text,
                propri_states=obs["robot_obs"][-1:],
                actions=actions,
                max_length=2048,    # TODO: make this a parameter
            )
        else:
            return {
                "instruction": text,
                "obs": obs,
                "actions": actions,
            }
