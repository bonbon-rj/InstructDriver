from typing import List, Type
import ast
import torch
import torch.distributed as dist

from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _get_velocity_and_acceleration,
    _se2_vel_acc_to_ego_state,
)

from data_utils import DataUtil
from pathlib import Path
import os

import numba
import numpy as np
import traceback

from accessory.data.alpaca import format_prompt


@numba.njit
def rotate_round_z_axis(points: np.ndarray, angle: float):
    rotate_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return points @ rotate_mat


class InstructDriver(AbstractPlanner):

    def __init__(self, torch_module_wrapper):
        self.torch_module_wrapper = torch_module_wrapper

        self.output = ''
        self.model = None
        self.args = None

        self.future_horizon = 8
        self.step_interval = 0.5
        self.pose_num = int(self.future_horizon / self.step_interval)
        # self.future_horizon = self.pose_num * self.step_interval
        self.motionless_trajectory = [[0.0,0.0,0.0] for _ in range(self.pose_num)]


    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        # model initialization
        from llm_patches.llm_singleton import LLMSingleton
        self.model, self.args = LLMSingleton.get_llm_and_args()

        # data process initialization
        self.datautil = DataUtil()

        # simulation initialization
        self.planner_feature_builder = self.torch_module_wrapper.get_list_of_required_feature()[0]
        self.initialization = initialization

    def name(self) -> str:
        return self.__class__.__name__
    
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> List[AbstractTrajectory]:
        ego_history = current_input.history.ego_states
        ego_state = ego_history[-1]

        json_instruction = self.datautil.get_instruction()
        feature_data = self.planner_feature_builder.get_features_from_simulation(current_input, self.initialization)
        json_input = self.datautil.get_input_for_iter(self.planner_feature_builder, feature_data, -1)

        if dist.get_rank() == 0:
            self.master_func(json_instruction, json_input)
        else:
            self.worker_func()
        
        if len(self.output)!=0:
            output_trajectory_description = self.output
            self.output = ''
            try:
                output_trajectory_str = self.output2traj(output_trajectory_description)
                output_trajectory_list = ast.literal_eval(output_trajectory_str)
                final_ret_traj = self.list2traj(ego_history, output_trajectory_list)
            except Exception:
                return self.list2traj(ego_history, self.motionless_trajectory)
        return final_ret_traj


    @ torch.inference_mode()
    def generate(self, prompt, question_input, system_prompt, max_gen_len, gen_t, top_p):
        image = None

        # text output
        _prompt = format_prompt({"instruction":prompt, "input":question_input}, system_prompt)

        dist.barrier()
        dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p])
        if self.args.quant:
            results = self.model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
        else:
            with torch.cuda.amp.autocast(dtype=self.args.target_dtype):
                results = self.model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
        text_output = results[0].strip()
        return text_output
    
    def worker_func(self):

        while True:
            dist.barrier()

            input_data = [None for _ in range(5)]
            dist.broadcast_object_list(input_data)
            _prompt, image, max_gen_len, gen_t, top_p = input_data
            with torch.cuda.amp.autocast(dtype=self.args.target_dtype):
                _ = self.model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p, )
        
    def master_func(self, json_instruction, json_input):

        system_prompt = "alpaca"
        max_gen_len = self.args.max_gen_len
        gen_t = 0.0
        top_p = 0.75
        try:
            output = self.generate(json_instruction, json_input, system_prompt, max_gen_len, gen_t, top_p)
            # output = self.model.generate(json_instruction, json_input, system_prompt, max_gen_len, gen_t, top_p)
        except:
            output = 'Error'
        
        self.output = output


    @staticmethod
    def to_global(local_trajectory, ego_state):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading

        global_position = (
            rotate_round_z_axis(np.ascontiguousarray(local_trajectory[..., :2]), -angle)
            + origin
        )
        global_heading = local_trajectory[..., 2] + angle

        global_trajectory = np.concatenate(
            [global_position, global_heading[..., None]], axis=1
        )
        return global_trajectory

    def list2traj(self, ego_history, input_list):

        ego_state = ego_history[-1]
        global_trajectory = self.to_global(np.array(input_list), ego_state)
        
        states = [StateSE2.deserialize(pose) for pose in global_trajectory]

        timesteps = _get_fixed_timesteps(ego_state, len(input_list) * self.step_interval, self.step_interval)
        velocities, accelerations = _get_velocity_and_acceleration(states, ego_history, timesteps)
        output_states = [
            _se2_vel_acc_to_ego_state(state, velocity, acceleration, timestep, ego_state.car_footprint.vehicle_parameters)
            for state, velocity, acceleration, timestep in zip(states, velocities, accelerations, timesteps)
        ]
        output_states.insert(0, ego_state)

        return InterpolatedTrajectory(output_states)
    
    @staticmethod
    def output2traj(output_str):
        match_str = "Trajectory:"
        return output_str[output_str.rfind(match_str) + len(match_str):]
    



