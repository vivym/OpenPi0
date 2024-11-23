import argparse
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def process_frame(frame_path: Path) -> dict | None:
    obj = np.load(frame_path)

    return {
        "rgb_static": obj["rgb_static"],
        "rgb_gripper": obj["rgb_gripper"],
        "depth_static": obj["depth_static"],
        "depth_gripper": obj["depth_gripper"],
        "rgb_tactile": obj["rgb_tactile"],
        "depth_tactile": obj["depth_tactile"],
        "actions": obj["actions"],
        "rel_actions": obj["rel_actions"],
        "robot_obs": obj["robot_obs"],
        "scene_obs": obj["scene_obs"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/CALVIN")
    parser.add_argument("--output_path", type=str, default="data/CALVIN")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root_dir = Path(args.input_path)
    assert input_root_dir.exists(), f"{input_root_dir} does not exist"

    for env_name in ["calvin_debug_dataset", "task_ABCD_D", "task_ABC_D", "task_D_D"]:
        env_dir = input_root_dir / env_name
        if not env_dir.exists():
            print(f"{env_dir} does not exist, skipping")
            continue

        output_dir = Path(args.output_path) / env_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in ["training", "validation"]:
            split_dir = env_dir / split
            assert split_dir.exists(), f"{split_dir} does not exist"

            print(f"Processing {split_dir}")

            output_path = output_dir / f"{split}.h5"
            if output_path.exists() and not args.overwrite:
                print(f"{output_path} exists, skipping")
                continue

            ann_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
            ann = np.load(ann_path, allow_pickle=True).item()
            texts = ann["language"]["ann"]
            tasks = ann["language"]["task"]
            ep_start_end_ids = ann["info"]["indx"]

            frame_ids = []
            for start_idx, end_idx in ep_start_end_ids:
                frame_ids.extend(list(range(start_idx, end_idx + 1)))

            frame_ids = sorted(set(frame_ids))
            frame_ids_array = np.array(frame_ids, dtype=np.int64)

            frame_paths = [split_dir / f"episode_{frame_idx:07d}.npz" for frame_idx in frame_ids]

            with h5py.File(output_path, mode="w", locking=True) as f:
                meta_group = f.create_group("meta")
                lang_group = f.create_group("lang")
                obs_group = f.create_group("obs")
                action_group = f.create_group("action")

                ep_start_end_ids = np.array(ep_start_end_ids, dtype=np.int64)
                meta_group.create_dataset("ep_start_end_ids", data=ep_start_end_ids)
                meta_group.create_dataset("frame_ids", data=frame_ids_array)

                lang_group.create_dataset("text", data=texts, dtype=h5py.string_dtype())
                lang_group.create_dataset("task_type", data=tasks, dtype=h5py.string_dtype())

                obs_keys = [
                    "rgb_static",
                    "rgb_gripper",
                    "depth_static",
                    "depth_gripper",
                    "rgb_tactile",
                    "depth_tactile",
                    "robot_obs",
                    "scene_obs",
                ]
                action_keys = ["actions", "rel_actions"]
                obs_list = {
                    key: []
                    for key in obs_keys
                }
                action_list = {
                    key: []
                    for key in action_keys
                }

                bsz = 10000

                obs_group.create_dataset(
                    "rgb_static",
                    (bsz, 200, 200, 3),
                    maxshape=(None, 200, 200, 3),
                    dtype=np.uint8,
                )

                obs_group.create_dataset(
                    "rgb_gripper",
                    (bsz, 84, 84, 3),
                    maxshape=(None, 84, 84, 3),
                    dtype=np.uint8,
                )

                obs_group.create_dataset(
                    "rgb_tactile",
                    (bsz, 160, 120, 6),
                    maxshape=(None, 160, 120, 6),
                    dtype=np.uint8,
                )

                obs_group.create_dataset(
                    "depth_static",
                    (bsz, 200, 200),
                    maxshape=(None, 200, 200),
                    dtype=np.float32,
                )

                obs_group.create_dataset(
                    "depth_gripper",
                    (bsz, 84, 84),
                    maxshape=(None, 84, 84),
                    dtype=np.float32,
                )

                obs_group.create_dataset(
                    "depth_tactile",
                    (bsz, 160, 120, 2),
                    maxshape=(None, 160, 120, 2),
                    dtype=np.float32,
                )

                obs_group.create_dataset(
                    "robot_obs",
                    (bsz, 15),
                    maxshape=(None, 15),
                    dtype=np.float64,
                )

                obs_group.create_dataset(
                    "scene_obs",
                    (bsz, 24),
                    maxshape=(None, 24),
                    dtype=np.float64,
                )

                action_group.create_dataset(
                    "actions",
                    (bsz, 7),
                    maxshape=(None, 7),
                    dtype=np.float64,
                )

                action_group.create_dataset(
                    "rel_actions",
                    (bsz, 7),
                    maxshape=(None, 7),
                    dtype=np.float64,
                )

                offset = 0
                with mp.Pool(32) as p:
                    for results in tqdm(p.imap(process_frame, frame_paths), total=len(frame_paths), desc="Processing frames"):
                        for key in obs_keys:
                            obs_list[key].append(results[key])
                        for key in action_keys:
                            action_list[key].append(results[key])

                        if len(obs_list["rgb_static"]) >= bsz:
                            for key in obs_keys:
                                obs_data = np.stack(obs_list[key], axis=0)
                                ds = obs_group[key]
                                ds.resize(offset + obs_data.shape[0], axis=0)
                                ds[offset : offset + obs_data.shape[0]] = obs_data
                                obs_list[key] = []
                            for key in action_keys:
                                action_data = np.stack(action_list[key], axis=0)
                                ds = action_group[key]
                                ds.resize(offset + action_data.shape[0], axis=0)
                                ds[offset : offset + action_data.shape[0]] = action_data
                                action_list[key] = []
                            offset += obs_data.shape[0]

                if len(obs_list["rgb_static"]) > 0:
                    for key in obs_keys:
                        obs_data = np.stack(obs_list[key], axis=0)
                        ds = obs_group[key]
                        ds.resize(offset + obs_data.shape[0], axis=0)
                        ds[offset : offset + obs_data.shape[0]] = obs_data
                    for key in action_keys:
                        action_data = np.stack(action_list[key], axis=0)
                        ds = action_group[key]
                        ds.resize(offset + action_data.shape[0], axis=0)
                        ds[offset : offset + action_data.shape[0]] = action_data


if __name__ == "__main__":
    main()
