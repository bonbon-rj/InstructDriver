import os
from pathlib import Path
import math
import numpy as np
import copy
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from collections import Counter

class DataUtil:
    def __init__(self, 
                 future_horizon=8.0, future_sample_interval=0.5, 
                 historical_horizon=2.0, historical_sample_interval=0.5, 
                 predict_second=1.0,
                 raw_sample_interval=0.1, 
                 accuracy=2):
    
        # used to form the input
        self.future_horizon = future_horizon # s
        self.future_sample_interval = future_sample_interval
        self.future_pose_num = int(self.future_horizon / self.future_sample_interval)
        self.historical_horizon = historical_horizon # s
        self.historical_sample_interval = historical_sample_interval
        self.historical_pose_num = int(self.historical_horizon / self.historical_sample_interval)
        self.predict_second = predict_second

        # raw data
        # raw_future_horizon = 8
        self.raw_sample_interval = raw_sample_interval

        self.accuracy = accuracy


    def get_instruction(self):
        instruction_file = Path(os.path.dirname(os.path.realpath(__file__))) / 'instruction.txt'

        with open(str(instruction_file), 'r', encoding='utf-8') as f:
            data = f.read()
            return data.format(self.predict_second, 
                            self.future_horizon, self.future_pose_num, self.future_sample_interval, 
                            self.historical_horizon, self.historical_pose_num, 
                            self.predict_second,
                            self.future_pose_num, self.future_pose_num, self.future_pose_num)
    @staticmethod
    def get_point_orientation_description(point, mid_point=[0,0]):
        threshold = 0.3
        xp, yp = point
        x0, y0 = mid_point
        if abs(xp - x0) > threshold:
            if xp>x0:
                if abs(yp - y0) < threshold:
                    return "front"
                elif yp > y0:
                    return "left front"
                elif yp <= y0:
                    return "right front"
            else:
                if abs(yp - y0) < threshold:
                    return "rear"
                elif yp > y0:
                    return "left rear"
                elif yp <= y0:
                    return "right rear"
        else:
            if yp > y0:
                return "left"
            else:
                return "right"

    @staticmethod
    def traj_to_action(traj_list):
        
        straight_threshold = 0.1

        x, y, yaw = traj_list[0]
        xd, yd, yawd = traj_list[-1]

        angle_to_destination = math.atan2(yd - y, xd - x) # -pi pi
        angle_diff = angle_to_destination - yaw

        distance_threshold = 0.05
        acceleration_threshold = 0.05 
        yaw_mean_threshold = 0.02 

        traj_list = np.array(traj_list)

        x_diff = np.diff(traj_list[:,0])
        y_diff = np.diff(traj_list[:,1])
        distance_mean = np.mean(np.sqrt(x_diff*x_diff+y_diff*y_diff))
        if distance_mean < distance_threshold:
            meta_action_str = "Keep in place"
        else:
            x_diff_diff_mean = np.mean(np.diff(x_diff))
            y_diff_diff_mean = np.mean(np.diff(y_diff))
            yaw_mean = np.mean(np.diff(traj_list[:, 2]))

            if abs(angle_diff) > math.pi /2:
                if angle_diff > 0:
                    if x_diff_diff_mean > acceleration_threshold:
                        meta_action_str = "Decelerate to turn left and make a U-turn."
                    elif x_diff_diff_mean > -acceleration_threshold:
                        meta_action_str = "Uniform to turn left and make a U-turn."
                    else:
                        meta_action_str = "Accelerates to turn left and make a U-turn."
                
                    meta_action_str = 'Turn left and make a U-turn.'
                else:
                    if x_diff_diff_mean > acceleration_threshold:
                        meta_action_str = "Decelerate to turn right and make a U-turn."
                    elif x_diff_diff_mean > -acceleration_threshold:
                        meta_action_str = "Uniform to turn right and make a U-turn."
                    else:
                        meta_action_str = "Accelerates to turn right and make a U-turn."
            else:
                # left
                if yaw_mean > yaw_mean_threshold:
                    if y_diff_diff_mean > acceleration_threshold:
                        meta_action_str = "Decelerate to make a left turn"
                    elif y_diff_diff_mean > -acceleration_threshold:
                        meta_action_str = "Uniform to make a left turn"
                    else:
                        meta_action_str = "Accelerates to make a left turn"
                # stright
                elif yaw_mean > -yaw_mean_threshold:
                    if x_diff_diff_mean > acceleration_threshold:
                        meta_action_str = "Accelerates to go straight"
                    elif x_diff_diff_mean > -acceleration_threshold:
                        meta_action_str = "Uniform to go straight"
                    else:
                        meta_action_str = "Decelerate to go straight"
                # right
                else:
                    if y_diff_diff_mean > acceleration_threshold:
                        meta_action_str = "Accelerates to make a right turn"
                    elif y_diff_diff_mean > -acceleration_threshold:
                        meta_action_str = "Uniform to make a right turn"
                    else:
                        meta_action_str = "Decelerate to make a right turn"
        return meta_action_str
    
    @staticmethod
    def lane_to_action(lane_list):
        lane_list = np.array(lane_list)
        deltas = np.diff(lane_list, axis=0) 
        angles = np.arctan2(deltas[:, 1], deltas[:, 0]) 
        
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        straight_angle_threshold = 0.02
        if -straight_angle_threshold <= mean_angle < straight_angle_threshold:
            return 'Go straight'
        elif straight_angle_threshold <= mean_angle < np.pi / 2:
            return 'Turn left'
        elif - np.pi / 2 <= mean_angle < -straight_angle_threshold:
            return 'Turn right'
        else:
            return 'Make a U-turn'

    @staticmethod
    def pose_to_des_action(pose, des):
        straight_threshold = 0.1

        x, y, yaw = pose
        xd, yd = des

        angle_to_destination = math.atan2(yd - y, xd - x) # -pi pi
        angle_diff = angle_to_destination - yaw

        action = ''
        if abs(angle_diff) > math.pi /2:
            if angle_diff > 0:
                action = 'Turn left and make a U-turn.'
            else:
                action = 'Turn right and make a U-turn.'
        else:
            if angle_diff > straight_threshold:
                action = 'Turn left.'
            elif angle_diff > -straight_threshold:
                action = 'Go straight.'
            else:
                action = 'Turn right.'
        return action



    @staticmethod
    def get_rectangle_from_pose(pose, width, length):

        xm, ym, heading = pose

        if math.cos(heading) == 0:
            x1 = xm + width / 2
            y1 = ym - length / 2
            x2 = xm + width / 2
            y2 = ym + length / 2
            x3 = xm - width / 2
            y3 = ym + length / 2
            x4 = xm - width / 2
            y4 = xm - length / 2
        elif math.sin(heading) == 0:
            x1 = xm + length / 2
            y1 = ym - width / 2
            x2 = xm + length / 2
            y2 = ym + width / 2
            x3 = xm - length / 2
            y3 = ym + width / 2
            x4 = xm - length / 2
            y4 = ym - width / 2
        else:
            k = math.tan(heading)
            theta = heading

            # middle_line: y=k1(x-xm)+ym   long_line: y=k1x+b1+yb1 y=k1x+b1-yb1
            k1 = k
            b1 = ym - k1 * xm
            yb1 = (width / 2) / math.cos(theta)

            # middle_line: y=k2(x-xm)+ym   long_line: y=k2x+b2+yb2 y=k2x+b2-yb2
            k2 = -1 / k1
            b2 = ym - k2 * xm
            yb2 = (length / 2) / math.sin(theta)

            # four points
            x1 = ((b2 - yb2) - (b1 + yb1)) / (k1 - k2)
            y1 = k1 * x1 + b1 + yb1
            x2 = ((b2 + yb2) - (b1 + yb1)) / (k1 - k2)
            y2 = k1 * x2 + b1 + yb1
            x3 = ((b2 + yb2) - (b1 - yb1)) / (k1 - k2)
            y3 = k1 * x3 + b1 - yb1
            x4 = ((b2 - yb2) - (b1 - yb1)) / (k1 - k2)
            y4 = k1 * x4 + b1 - yb1

        points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        return points

    def round_list(self, lst):
        return [round(data, self.accuracy) for data in lst]

    def round_2d_list(self, lsts):
        return [self.round_list(lst) for lst in lsts]

    def round_vertices(self, vertices):
        return [self.round_list(point) for point in vertices]

    @staticmethod
    def point_in_quad(point, quad):
        x, y = point

        inside = False
        for i in range(len(quad)):
            x1, y1 = quad[i]
            x2, y2 = quad[(i + 1) % len(quad)]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside

        return inside

    def get_input_for_iter(self, feature_builder, feature_data, idx):


        objects_types = feature_builder.interested_objects_types
        polygon_types = feature_builder.polygon_types

        # ego
        ego_data = copy.deepcopy(feature_data.data['ego'])

        # offset
        ego_position_offset = ego_data['position'][:,idx,:][0]
        ego_data['position'] = ego_data['position'] - ego_position_offset
        ego_heading_offset = ego_data['heading'][:,idx][0]
        ego_data['heading'] = ego_data['heading'] - ego_heading_offset

        xm, ym = ego_data['position'][:,idx,:][0]
        heading = ego_data['heading'][:,idx][0]
        w, h = ego_data['shape'][:,idx,:][0]
        vx, vy = ego_data['velocity'][:,idx,:][0]
        ax, ay = ego_data['acceleration'][:,idx,:][0]

        ego_pose = [xm, ym, heading]
        ego_vertice = self.get_rectangle_from_pose(ego_pose, w, h)
        ego_velocity = [vx, vy]
        ego_acceleration = [ax, ay]
        ego_predict_position = [
            xm + vx * self.predict_second + 0.5 * ax * self.predict_second * self.predict_second,
            ym + vy * self.predict_second + 0.5 * ay * self.predict_second * self.predict_second
        ]

        # agents
        agents_data = copy.deepcopy(feature_data.data['agents'])
        # offset
        agents_data['position'] = agents_data['position'] - ego_position_offset
        agents_data['heading'] = agents_data['heading'] - ego_heading_offset
        agents_category = []
        for category_index in agents_data['category']:
            category = objects_types[category_index].fullname
            agents_category.append(category)

        agents_num = len(agents_data['position'])
        agents_shapes = []
        agents_poses = []
        agents_velocitys = []
        agents_predictions = []


        for i in range(agents_num):
            xm, ym = agents_data['position'][:,idx,:][i]
            heading = agents_data['heading'][:,idx][i]
            w, h = agents_data['shape'][:,idx,:][i]
            vx, vy = agents_data['velocity'][:,idx,:][i]
            pose = [xm, ym, heading]
            vertices = self.get_rectangle_from_pose(pose, w, h)

            agents_shapes.append(vertices)
            agents_poses.append(pose)     
            agents_velocitys.append([vx,vy])
            agents_predictions.append([xm + vx * self.predict_second, ym + vy * self.predict_second])

        assert agents_num == len(agents_category) ==  len(agents_shapes) ==  len(agents_poses) ==  len(agents_velocitys) ==  len(agents_predictions) 

        input_str = ''
        input_str += 'Perception:\n'
        for i in range(agents_num):
            if (abs(agents_poses[i][0]) > 1e5) or (abs(agents_poses[i][1]) > 1e5):
                continue
            
            input_str += "- "
            input_str += agents_category[i] if self.check_agents_category(agents_category[i]) else self.get_agents_category_from_shape(agents_data['shape'][:,idx,:][i])
            input_str += ":\n"
            input_str += "-- Shape: "
            input_str += str(self.round_vertices(agents_shapes[i]))
            input_str += "\n"
            input_str += "-- Pose: "
            input_str += str(self.round_list(agents_poses[i]))
            input_str += "\n"
            input_str += "-- Velocity: "
            input_str += str(self.round_list(agents_velocitys[i]))
            input_str += "\n"
            input_str += "-- Prediction: "
            input_str += str(self.round_list(agents_predictions[i]))
            input_str += "\n"

        input_str += '\n'

        input_str += 'Ego-States:\n'
        input_str += "- Shape: "
        input_str += str(self.round_vertices(ego_vertice))
        input_str += "\n"
        input_str += "- Pose: "
        input_str += str(self.round_list(ego_pose))
        input_str += "\n"
        input_str += "- Velocity: "
        input_str += str(self.round_list(ego_velocity))
        input_str += "\n"
        input_str += "- Acceleration: "
        input_str += str(self.round_list(ego_acceleration))
        input_str += "\n"

        input_str += '\n'

        input_str += 'Historical Trajectory:\n'
        cnt = 1
        step = int(self.historical_sample_interval / self.raw_sample_interval)
        history_trajectory = []
        for _ in range(self.historical_pose_num):
            index = idx - (step * cnt) 

            xm, ym = ego_data['position'][:,index,:][0]
            heading = ego_data['heading'][:,index][0]
            history_trajectory.append(self.round_list([xm, ym, heading]))
            cnt += 1
        history_trajectory.reverse()
        input_str += str(history_trajectory)
        input_str += '\n'

        input_str += '\n'
   
        map_data = copy.deepcopy(feature_data.data['map'])
        
        self.current_nearest_line_idx = self.from_position_get_nearest_line_idx(map_data, ego_pose[:2])


        traffic_status_on_route = map_data["polygon_tl_status"][np.where(map_data['polygon_on_route'])]
        traffic_status_on_route = traffic_status_on_route[np.where(traffic_status_on_route != TrafficLightStatusType.UNKNOWN.value)]
        traffic_status_on_route_counts = Counter(traffic_status_on_route.tolist())
        if len(traffic_status_on_route_counts)==0:
            self.traffic_status = TrafficLightStatusType.UNKNOWN
        else:
            self.traffic_status = TrafficLightStatusType(max(traffic_status_on_route_counts, key=lambda status: traffic_status_on_route_counts[status]))         
        self.next_nearest_line_idx = self.from_position_get_nearest_line_idx(map_data, ego_predict_position)

        input_str += 'Map Insights:\n'
        input_str += f'- Traffic Light Status: {self.traffic_status.name}\n'
        input_str += f'- Speed Limit: {str(round(map_data["polygon_speed_limit"][self.current_nearest_line_idx], self.accuracy))}\n'
        self.current_lane_centerline = self.round_2d_list(self.simply_line(map_data["point_position"][self.current_nearest_line_idx, 0, :, :].tolist()))
        input_str += f'- Current Lane Centerline: {str(self.current_lane_centerline)}\n'

        nearby_lane_boundaries = []
        current_nearest_line_idxs = [self.current_nearest_line_idx, self.next_nearest_line_idx]
        for i in current_nearest_line_idxs:
            for j in [1,2]:
                nearby_lane_boundaries.append(map_data["point_position"][i, j, :, :].tolist())
        nearby_lane_boundaries = self.remove_duplicate_lines(nearby_lane_boundaries)
        input_str += '- Nearby Lane Boundaries:\n'
        for boundary in nearby_lane_boundaries:
            input_str += str(self.round_2d_list(self.simply_line(boundary))) + '\n'

        return input_str

    def get_output_iter(self, feature_builder, feature_data, history_idx, idx):
        objects_types = feature_builder.interested_objects_types

        ego_target_data = copy.deepcopy(feature_data.data['ego']['target'])

        last_index = (idx - history_idx) - 1 
        if last_index > 0:
            # offset
            pose_offset = ego_target_data[:,last_index,:][0]
            ego_target_data = ego_target_data - pose_offset

        output_str = ''

        cnt = 1
        step = int(self.future_sample_interval / self.raw_sample_interval)
        trajectory = []
        for _ in range(self.future_pose_num):
            index = (idx - history_idx) + (step * cnt) - 1
            pose = ego_target_data[:,index,:][0]
            trajectory.append(self.round_list(pose))
            cnt += 1

        agents_data = copy.deepcopy(feature_data.data['agents'])
        ego_data = copy.deepcopy(feature_data.data['ego'])
        map_data = copy.deepcopy(feature_data.data['map'])

        ## offset
        ego_position_offset = ego_data['position'][:,idx,:][0]
        ego_data['position'] = ego_data['position'] - ego_position_offset
        ego_heading_offset = ego_data['heading'][:,idx][0]
        ego_data['heading'] = ego_data['heading'] - ego_heading_offset
        # offset
        agents_data['position'] = agents_data['position'] - ego_position_offset
        agents_data['heading'] = agents_data['heading'] - ego_heading_offset    

        caution_threshold = 3
        immediate_attention_threshold = 1.5
        # [xm, ym, pred_xm, pred_ym, pred_shape], dir([xm, ym, pred_xm, pred_ym, id, dis, points])
        later_ego_lst, later_nearly_lst = self.get_nearly_zone_agents_dir(objects_types,ego_data,agents_data,idx, self.predict_second, caution_threshold)

        output_str += 'Thoughts:\n'
        output_str += f'1.Preliminary Action: Follow Current Lane Centerline {self.current_lane_centerline}, {str(self.lane_to_action(self.current_lane_centerline))}.\n'
        output_str += f"2.Collisions Considerations:\n"
        output_str += f'ego-vehicle at {str(self.round_list(later_ego_lst[0:2]))} to {str(self.round_list(later_ego_lst[2:4]))}, '
        output_str += f'predicted Shape is {str(self.round_2d_list(later_ego_lst[-1]))}.\n'
        output_str += f'- {self.predict_second}s later near ego-vehicle:\n'
        if len(later_nearly_lst) == 0:
            output_str += "No object poses a potential hazard, safe.\n"
        else: 
            for later_dir in later_nearly_lst:
                for key, value in later_dir.items():   
                    object_type, object_id, object_pred_x, object_pred_y, object_id, object_r, points = value
                    ego_pred_x, ego_pred_y = later_ego_lst[2:4]

                    left_right_threshold = 1.5
                    left, right = False, False
                    error = ego_pred_y - object_pred_y
                    if error < -left_right_threshold:
                        left = True
                    elif error < left_right_threshold:
                        pass
                    else:
                        right = True

                    
                    if object_r < immediate_attention_threshold:
                        output_str += f"!!! {key} [{object_id}] at {str(self.round_list(value[0:2]))} to {str(self.round_list(value[2:4]))}, "
                        if left:
                            output_str += f"located on the left."
                        elif right:
                            output_str += f"located on the right."
                        else:
                            output_str += f"located in front."      

                        output_str += f" Its {self.get_point_orientation_description(points[1],value[2:4])} near {self.get_point_orientation_description(points[0])} of ego-vehicle at a distance of {round(object_r, self.accuracy)}m."
                        output_str += f" Immediate action required for {self.get_point_orientation_description(points[0])} in critical state!\n"
                    else:  
                        output_str += f"! {key} [{object_id}] at {str(self.round_list(value[0:2]))} to {str(self.round_list(value[2:4]))}, "
                        if left:
                            output_str += f"located on the left."
                        elif right:
                            output_str += f"located on the right."
                        else:
                            output_str += f"located in front."

                        output_str += f" Its {self.get_point_orientation_description(points[1],value[2:4])} near {self.get_point_orientation_description(points[0])} of ego-vehicle at a distance of {round(object_r, self.accuracy)}m."
                        output_str += f" Vigilance needed on {self.get_point_orientation_description(points[0])}.\n"

        ego_xm, ego_ym = ego_data['position'][:,idx,:][0]
        ego_vx, ego_vy = ego_data['velocity'][:,idx,:][0]
        ego_heading = ego_data['heading'][:,idx][0]
        ego_w, ego_h = ego_data['shape'][:,idx,:][0]
        ego_pose = [ego_xm, ego_ym, ego_heading]
        ego_vertice = self.get_rectangle_from_pose(ego_pose, ego_w, ego_h)

        ego_ax, ego_ay = ego_data['acceleration'][:,idx,:][0]
        traffic_status = self.traffic_status
        output_str += "3.Transportation Considerations:\n"
        if traffic_status == TrafficLightStatusType.GREEN:
            output_str += '- Traffic Light: Green. Continue with caution.\n'
        elif traffic_status == TrafficLightStatusType.YELLOW:
            output_str += '- Traffic Light: Yellow. Reduce speed and prepare to halt.\n'
        elif traffic_status == TrafficLightStatusType.RED:
            output_str += '- Traffic Light: Red. Come to a complete stop at the stop line.\n'
        else:
            output_str += '- Traffic Light: Unknown. Proceed with caution and adhere to prevailing traffic rules.\n'
        speed_now = round(math.sqrt(ego_vx*ego_vx+ego_vy*ego_vy), self.accuracy)
        speed_limit = round(map_data["polygon_speed_limit"][self.current_nearest_line_idx], self.accuracy)
        
        
        if speed_limit <= 0.1:
            output_str += f"- Speed Limit: The current roadway has no speed limit requirements.\n"
        else:
            if speed_now > speed_limit:
                exceed_amount = speed_now - speed_limit
                output_str += f"- Speed Limit: Exceeded: Current speed of {speed_now} m/s surpasses the speed limit of {speed_limit} m/s by {exceed_amount} m/s. Deceleration required to comply with regulations.\n"
            else:
                speed_limit_threshold = 0.5
                speed_difference = speed_limit - speed_now
                if speed_difference< speed_limit_threshold:
                    output_str += f"- Speed Limit: Alert: Current speed at {speed_now} m/s is just {speed_difference} m/s under the speed limit of {speed_limit} m/s. No further acceleration allowed.\n"
                else:    
                    output_str += f"- Speed Limit: Safe: Currently traveling at {speed_now} m/s, which is within the speed limit of {speed_limit} m/s. Continue monitoring speed and proceed with caution.\n"
        
        def fit_curve(x, y, degree):
            coeffs = np.polyfit(x, y, degree)
            p = np.poly1d(coeffs)
            return p
        def vertice_intersect_curve(vertice, curve):
            last_above = None

            for p in vertice:
                x0, y0 = p
                y1 = curve(x0) 
                above = y1 > y0 

                if last_above is not None and above != last_above:
                    return True

                last_above = above 

            return False

        left_bound = map_data["point_position"][self.current_nearest_line_idx, 1, :, :].tolist()
        left_x, left_y = zip(*left_bound)
        left_curve = fit_curve(left_x, left_y, 2)
        left_contact = False
        right_contact = False
        if vertice_intersect_curve(ego_vertice, left_curve):
            left_contact = True
        right_bound = map_data["point_position"][self.current_nearest_line_idx, 2, :, :].tolist()
        right_x, right_y = zip(*right_bound)
        right_curve = fit_curve(right_x, right_y, 2)
        if vertice_intersect_curve(ego_vertice, right_curve):
            right_contact = True

        if left_contact:
            output_str += "- Borderline: Detected proximity or contact with the left boundary line."
            output_str += "Initiate corrective steering to the right, unless a lane change maneuver is currently underway.\n"
        elif right_contact:
            output_str += "- Borderline: Detected proximity or contact with the right boundary line."
            output_str += "Initiate corrective steering to the left, unless a lane change maneuver is currently underway.\n"
        else:
            output_str += "- Borderline: No proximity or contact detected with the left and right boundary line. Maintain current trajectory.\n"

        output_str += f'4.Final Action: {self.traj_to_action(trajectory)}.\n'
        output_str += '\n'
        
        output_str += 'Trajectory:\n'
        output_str += str(trajectory) 
        output_str += '\n'

        return output_str


    def get_nearly_zone_agents_dir(self, objects_types, ego_data, agents_data, idx, pred_sec, threshold):
        # If pred_sec is 0, it means now

        agents_id = [i for i in range(len(agents_data['position']))]
        agents_category = []
        for category_index in agents_data['category']:
            category = objects_types[category_index].fullname
            agents_category.append(category)
            
        ego_xm, ego_ym = ego_data['position'][:,idx,:][0]
        ego_vx, ego_vy = ego_data['velocity'][:,idx,:][0]
        ego_ax, ego_ay = ego_data['acceleration'][:,idx,:][0]
        ego_heading = ego_data['heading'][:,idx][0]
        ego_pred_xm = ego_xm + ego_vx * self.predict_second + 0.5 * ego_ax * self.predict_second * self.predict_second
        ego_pred_ym = ego_ym + ego_vy * self.predict_second + 0.5 * ego_ay * self.predict_second * self.predict_second
        ego_pred_pose = [ego_pred_xm, ego_pred_ym, ego_heading] #  Assuming that heading is unchanged
        ego_pred_rect = self.get_rectangle_from_pose(ego_pred_pose,*(ego_data['shape'][:,idx,:][0]))

        nearly_lst = []
        agents_num = len(agents_data['position'])
        for i in range(agents_num):
            agents_xm, agents_ym = agents_data['position'][:,idx,:][i]
            if (abs(agents_xm) > 1e5) or (abs(agents_ym) > 1e5):
                continue
            agents_vx, agents_vy = agents_data['velocity'][:,idx,:][i]
            agents_heading = agents_data['heading'][:,idx][i]
            agents_w, agents_h = agents_data['shape'][:,idx,:][i]

            agents_pred_xm, agents_pred_ym = [agents_xm + agents_vx * pred_sec, agents_ym + agents_vy * pred_sec]
            agents_pred_pose = [agents_pred_xm, agents_pred_ym, agents_heading]  #  Assuming that heading is unchanged
            agents_pred_rect = self.get_rectangle_from_pose(agents_pred_pose, agents_w, agents_h)
            
            min_dis, closest_point = self.get_closest_point_of_2rect(self.rect_to_8points(ego_pred_rect), self.rect_to_8points(agents_pred_rect))
            if min_dis < threshold:
                category = agents_category[i] if self.check_agents_category(agents_category[i]) else self.get_agents_category_from_shape(agents_data['shape'][:,idx,:][i])
                nearly_dir = {}
                nearly_dir[category] = [agents_xm, agents_ym, agents_pred_xm, agents_pred_ym, agents_id[i], min_dis, closest_point]
                nearly_lst.append(nearly_dir)
        return [ego_xm, ego_ym, ego_pred_xm, ego_pred_ym, ego_pred_rect], nearly_lst

    @staticmethod
    def check_agents_category(category):
        return (category != TrackedObjectType.EGO.fullname)

    @staticmethod
    def rect_to_8points(rect):
        middle_point = []
        for i in range(len(rect)):
            xs, ys = rect[i]
            xe, ye = rect[(i + 1) % len(rect)] 
            middle_point.append([(xs+xe)/2, (ys+ye)/2])
        return rect + middle_point
    
    @staticmethod
    def get_agents_category_from_shape(shape):
        w, h = shape
        area = w * h
        if area < 1:
            return TrackedObjectType.PEDESTRIAN.fullname
        elif area < 4:
            return TrackedObjectType.BICYCLE.fullname
        else:
            return TrackedObjectType.VEHICLE.fullname
    
    @staticmethod
    def save_list_to_json(lst, save_path):
        import json

        with open(save_path, 'w') as file:
            file.write("[\n")
            length = len(lst)
            for i, data in enumerate(lst):
                if i != length - 1:
                    file.write(json.dumps(data) + ",\n")
                else:
                    file.write(json.dumps(data) + "\n]")
    @staticmethod
    def simply_line(line):
        line_point_step = 4
        line_start = line[0]
        line_end = line[-1]

        return [line_start] + line[1:-1:line_point_step] + [line_end]
    
    @staticmethod
    def from_position_get_nearest_line_idx(map_data, position):
        polygon_nums = len(map_data['polygon_on_route'])
        nearest_line_idx = -1
        min_dis = float("inf")
        for i in range(polygon_nums):
            if map_data['polygon_on_route'][i]:       
                x2, y2 = position
                center_line = map_data["point_position"][i,0,:,:]
                for p in center_line:
                    x1, y1 = p
                    dis = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
                    if dis < min_dis:
                        min_dis = dis
                        nearest_line_idx = i
        return nearest_line_idx
    

    def from_pose_get_nearest_line_idxs(self, map_data, pose):

        def line_in_quad(line, quad): 
            success_cnt = 0
            for p in line:
                if self.point_in_quad(p,quad):
                    success_cnt +=1
            if success_cnt >= 3:
                return True
            return False

        range_width = 20
        range_length = 24
        range_rect = self.get_rectangle_from_pose(pose,range_width,range_length)

        polygon_nums = len(map_data['polygon_on_route'])
        nearest_line_idxs = []
        for i in range(polygon_nums):
            if map_data['polygon_on_route'][i]:       

                lines = map_data["point_position"][i,:,:,:]
                all_line_in_quad = True
                for line in lines:
                    if not line_in_quad(line, range_rect):
                        all_line_in_quad = False
                if all_line_in_quad:
                    nearest_line_idxs.append(i)
        return nearest_line_idxs
    
    @staticmethod 
    def remove_duplicate_lines(lines):
        def check_line_same(line1,line2):
            num = len(line1) if len(line1) > len(line2) else len(line2)
            cnt = 0
            for i in range(num):
                x1,y1 = line1[i]
                x2,y2 = line2[i]
                dis = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
                if dis < 0.25:
                    cnt += 1
            if cnt / num > 0.75:
                return True
            return False

        def check_line_in_lines(line,lines):
            have_same = False 
            for l in lines:
                if check_line_same(line, l):
                    have_same = True
            return have_same

        new_lines = []
        for line in lines:
            if not check_line_in_lines(line, new_lines):
                new_lines.append(line)
        return new_lines

    
    @staticmethod 
    def get_closest_point_of_2rect(rect1, rect2):
        min_dis = float('inf')
        closest_point = []
        for p1 in rect1:
            for p2 in rect2:
                x1, y1 = p1
                x2, y2 = p2
                dis = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
                if dis < min_dis:
                    min_dis = dis
                    closest_point = [p1,p2]
        return min_dis, closest_point


