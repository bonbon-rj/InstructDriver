from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from torch.nn.utils.rnn import pad_sequence

from src.utils.conversion import to_device, to_numpy, to_tensor


@dataclass
class NuplanFeature(AbstractModelFeature):
    data: Dict[str, Any]

    @classmethod
    def collate(cls, feature_list: List[NuplanFeature]) -> NuplanFeature:
        batch_data = {}
        for key in ["agent", "map"]:
            batch_data[key] = {
                k: pad_sequence(
                    [f.data[key][k] for f in feature_list], batch_first=True
                )
                for k in feature_list[0].data[key].keys()
            }
        for key in ["current_state", "origin", "angle"]:
            batch_data[key] = torch.stack([f.data[key] for f in feature_list], dim=0)

        return NuplanFeature(data=batch_data)

    def to_feature_tensor(self) -> NuplanFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_tensor(v)
        return NuplanFeature(data=new_data)

    def to_numpy(self) -> NuplanFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_numpy(v)
        return NuplanFeature(data=new_data)

    def to_device(self, device: torch.device) -> NuplanFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_device(v, device)
        return NuplanFeature(data=new_data)

    def serialize(self) -> Dict[str, Any]:
        return self.data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> NuplanFeature:
        return NuplanFeature(data=data)

    def unpack(self) -> List[AbstractModelFeature]:
        raise NotImplementedError

    def is_valid(self) -> bool:
        return self.data["polylines"].shape[0] > 0

    @classmethod
    def normalize(
        self, data, first_time=False, radius=None, hist_steps=21
    ) -> NuplanFeature:
        cur_state = data["current_state"]
        center_xy, center_angle = cur_state[:2].copy(), cur_state[2].copy()

        rotate_mat = np.array(
            [
                [np.cos(center_angle), -np.sin(center_angle)],
                [np.sin(center_angle), np.cos(center_angle)],
            ],
            dtype=np.float64,
        )

        data["current_state"][:3] = 0

        data["ego"]["position"] = np.matmul(data["ego"]["position"] - center_xy, rotate_mat)
        data["ego"]["velocity"] = np.matmul(data["ego"]["velocity"], rotate_mat)
        data["ego"]["heading"] -= center_angle
        data["agents"]["position"] = np.matmul(data["agents"]["position"] - center_xy, rotate_mat)
        data["agents"]["velocity"] = np.matmul(data["agents"]["velocity"], rotate_mat)
        data["agents"]["heading"] -= center_angle

        data["map"]["point_position"] = np.matmul(
            data["map"]["point_position"] - center_xy, rotate_mat
        )
        data["map"]["point_vector"] = np.matmul(data["map"]["point_vector"], rotate_mat)
        data["map"]["point_orientation"] -= center_angle

        data["map"]["polygon_center"][..., :2] = np.matmul(
            data["map"]["polygon_center"][..., :2] - center_xy, rotate_mat
        )
        data["map"]["polygon_center"][..., 2] -= center_angle
        data["map"]["polygon_position"] = np.matmul(
            data["map"]["polygon_position"] - center_xy, rotate_mat
        )
        data["map"]["polygon_orientation"] -= center_angle

        ego_target_position = (data["ego"]["position"][:, hist_steps:] - data["ego"]["position"][:, hist_steps - 1][:, None])
        ego_target_heading = (data["ego"]["heading"][:, hist_steps:]- data["ego"]["heading"][:, hist_steps - 1][:, None])
        ego_target = np.concatenate([ego_target_position, ego_target_heading[..., None]], -1)
        ego_target[~data["ego"]["valid_mask"][:, hist_steps:]] = 0
        data["ego"]["target"] = ego_target
        agents_target_position = (data["agents"]["position"][:, hist_steps:] - data["agents"]["position"][:, hist_steps - 1][:, None])
        agents_target_heading = (data["agents"]["heading"][:, hist_steps:]- data["agents"]["heading"][:, hist_steps - 1][:, None])
        agents_target = np.concatenate([agents_target_position, agents_target_heading[..., None]], -1)
        agents_target[~data["agents"]["valid_mask"][:, hist_steps:]] = 0
        data["agents"]["target"] = agents_target

        if first_time:
            point_position = data["map"]["point_position"]
            x_max, x_min = radius, -radius
            y_max, y_min = radius, -radius
            valid_mask = (
                (point_position[:, 0, :, 0] < x_max)
                & (point_position[:, 0, :, 0] > x_min)
                & (point_position[:, 0, :, 1] < y_max)
                & (point_position[:, 0, :, 1] > y_min)
            )
            valid_polygon = valid_mask.any(-1)
            data["map"]["valid_mask"] = valid_mask

            for k, v in data["map"].items():
                data["map"][k] = v[valid_polygon]

            data["origin"] = center_xy
            data["angle"] = center_angle

        return NuplanFeature(data=data)
