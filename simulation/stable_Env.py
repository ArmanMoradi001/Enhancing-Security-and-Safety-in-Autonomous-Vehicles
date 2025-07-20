import time
import random
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from simulation.settings import *

class CarlaEnvironment:
    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True):
        """Initialize the CARLA environment."""
        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action  # Note: 'continuous' misspelled as in original
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.town = town
        
        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Lists for tracking actors and observations
        self.sensor_list = []
        self.actor_list = []
        self.walker_list = []
        self.create_pedestrians()

        # ################################################# DoS attack and mitigation attributes ############################################
        self.attack_state = False
        self.noisy_observation = False
        self.mitigation_action = 0
        self.last_valid_sensors = None
        self.belief_attack = 0.2  # Initial belief of attack probability

        # Simulation variables
        self.timesteps = 0
        self.rotation = 0.0
        self.previous_location = None
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.target_speed = 22.0  # km/h
        self.max_speed = 25.0
        self.min_speed = 15.0
        self.max_distance_from_center = 3.0
        self.throttle = 0.0
        self.previous_steer = 0.0
        self.velocity = 0.0
        self.distance_from_center = 0.0
        self.angle = 0.0
        self.distance_covered = 0.0
        self.episode_start_time = 0.0
        self.collision_history = []
        self.image_obs = None
        self.navigation_obs = None
        self.fresh_start = True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0

    def reset(self):
        """Reset the environment to start a new episode."""
        try:
            # Clean up existing actors and sensors
            if self.actor_list or self.sensor_list:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()

            # Reset DoS attack and mitigation states
            self.attack_state = False
            self.noisy_observation = False
            self.mitigation_action = 0
            self.last_valid_sensors = None
            self.belief_attack = 0.2

            # Spawn vehicle
            vehicle_bp = self.get_vehicle(CAR_NAME)
            if self.town == "Town07":
                transform = self.map.get_spawn_points()[38]
                self.total_distance = 750
            elif self.town == "Town02":
                transform = self.map.get_spawn_points()[1]
                self.total_distance = 780
            else:
                transform = random.choice(self.map.get_spawn_points())
                self.total_distance = 250

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            if self.vehicle is None:
                raise RuntimeError("Failed to spawn vehicle")
            self.actor_list.append(self.vehicle)

            # Set up sensors
            self.camera_obj = CameraSensor(self.vehicle)
            while not self.camera_obj.front_camera:
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            # Reset simulation variables
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.throttle = 0.0
            self.previous_steer = 0.0
            self.velocity = 0.0
            self.distance_from_center = 0.0
            self.angle = 0.0
            self.distance_covered = 0.0

            # Initialize waypoints
            if self.fresh_start:
                self.current_waypoint_index = 0
                self.route_waypoints = []
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    if self.town == "Town07":
                        next_waypoint = current_waypoint.next(1.0)[0] if x < 650 else current_waypoint.next(1.0)[-1]
                    elif self.town == "Town02":
                        next_waypoint = current_waypoint.next(1.0)[-1] if x < 650 else current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]
                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
            else:
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            # Initial observation
            normalized_velocity = self.velocity / self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = self.angle / np.deg2rad(20)
            self.navigation_obs = np.array([
                self.throttle,
                self.velocity,
                normalized_velocity,
                normalized_distance_from_center,
                normalized_angle,
                self.belief_attack
            ])

            time.sleep(0.5)
            self.collision_history.clear()
            self.episode_start_time = time.time()
            return [self.image_obs, self.navigation_obs]

        except Exception as e:
            print(f"Error during reset: {e}")
            self.cleanup()
            raise

    def step(self, action_idx):
        """Execute one step in the environment."""
        try:
            self.timesteps += 1
            self.fresh_start = False

            ############################################# Determine true attack state (hidden)
            self.attack_state = random.random() < 0.2

            #################################################### Simulate noisy observation
            self.noisy_observation = random.random() < (0.9 if self.attack_state else 0.1)

            # Update belief state (P(attack | observation history))
            p_o_given_attack = 0.9 if self.noisy_observation else 0.1
            p_o_given_no_attack = 0.1 if self.noisy_observation else 0.9
            p_o = (p_o_given_attack * self.belief_attack) + (p_o_given_no_attack * (1 - self.belief_attack))
            self.belief_attack = (p_o_given_attack * self.belief_attack) / p_o if p_o > 0 else self.belief_attack

            # Apply throttle based on mitigation action
            self.mitigation_action = action_idx[2]
            throttle = float((action_idx[1] + 1.0) / 2)  # Base throttle (0 to 1)
            if self.mitigation_action == 1 and self.belief_attack > 0.5:  # Slow down
                throttle = max(0.0, throttle * 0.5)
            elif self.mitigation_action == 2 and self.belief_attack > 0.5:  # Use prior data
                if self.last_valid_sensors is not None:
                    self.image_obs = self.last_valid_sensors

            # Apply vehicle control
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            if self.continous_action_space:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle = max(min(throttle, 1.0), 0.0)
                self.vehicle.apply_control(carla.VehicleControl(
                    steer=self.previous_steer * 0.7 + steer * 0.3,
                    throttle=self.throttle * 0.7 + throttle * 0.3
                ))
                self.previous_steer = steer
                self.throttle = throttle
            else:
                steer = self.action_space[action_idx[0]]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(
                        steer=self.previous_steer * 0.7 + steer * 0.3,
                        throttle=1.0
                    ))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(
                        steer=self.previous_steer * 0.7 + steer * 0.3
                    ))
                self.previous_steer = steer
                self.throttle = 1.0

            # Traffic light handling
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            # Update state variables
            self.collision_history = self.collision_obj.collision_data
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.location = self.vehicle.get_location()

            # Waypoint tracking
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(
                    self.vector(wp.transform.get_forward_vector())[:2],
                    self.vector(self.location - wp.transform.location)[:2]
                )
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            self.current_waypoint = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index + 1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(
                self.vector(self.current_waypoint.transform.location),
                self.vector(self.next_waypoint.transform.location),
                self.vector(self.location)
            )
            self.center_lane_deviation += self.distance_from_center

            # Calculate angle deviation
            fwd = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle = self.angle_diff(fwd, wp_fwd)

            # Update checkpoint and last valid sensors
            if not self.fresh_start and self.checkpoint_frequency is not None:
                self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency
            if not self.attack_state:
                self.last_valid_sensors = self.image_obs

            # Compute rewards
            done = False
            reward = 0
            if (len(self.collision_history) != 0 or
                self.distance_from_center > self.max_distance_from_center or
                (self.episode_start_time + 10 < time.time() and self.velocity < 1.0) or
                self.velocity > self.max_speed):
                done = True
                reward = -10
            else:
                # Driving reward (R_d)
                centering_penalty = self.distance_from_center / self.max_distance_from_center
                angle_penalty = abs(self.angle / np.deg2rad(20))
                if self.velocity > self.target_speed:
                    speed_penalty = (self.velocity - self.target_speed) / (self.max_speed - self.target_speed)
                else:
                    speed_penalty = (self.target_speed - self.velocity) / (self.target_speed - self.min_speed) * 0.5
                R_d = 1.0 - 0.4 * centering_penalty - 0.4 * angle_penalty - 0.2 * speed_penalty
                R_d = max(R_d, 0.0)
                waypoint_progress = self.current_waypoint_index - self.checkpoint_waypoint_index
                R_d += 0.1 * waypoint_progress

                # Safety reward (R_s)
                if self.attack_state:
                    R_s = 1.0 if self.mitigation_action in [1, 2] else -1.0
                else:
                    R_s = -0.5 if self.mitigation_action in [1, 2] else 0.1

                reward = 0.7 * R_d + 0.3 * R_s

            # Check termination
            if self.timesteps >= 7500 or self.current_waypoint_index >= len(self.route_waypoints) - 2:
                done = True
                if self.current_waypoint_index >= len(self.route_waypoints) - 2:
                    self.fresh_start = True
                    if self.checkpoint_frequency is not None:
                        if self.checkpoint_frequency < self.total_distance // 2:
                            self.checkpoint_frequency += 2
                        else:
                            self.checkpoint_frequency = None
                            self.checkpoint_waypoint_index = 0

            # Retrieve camera observation
            while not self.camera_obj.front_camera:
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)

            # Construct navigation observation with belief
            normalized_velocity = self.velocity / self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([
                self.throttle,
                self.velocity,
                normalized_velocity,
                normalized_distance_from_center,
                normalized_angle,
                self.belief_attack
            ])

            # Cleanup on episode end
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps if self.timesteps > 0 else 0.0
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                for sensor in self.sensor_list:
                    sensor.destroy()
                self.remove_sensors()
                for actor in self.actor_list:
                    actor.destroy()

            return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation]

        except Exception as e:
            print(f"Error during step: {e}")
            self.cleanup()
            raise

    def create_pedestrians(self):
        """Spawn pedestrians in the environment."""
        try:
            walker_spawn_points = []
            for _ in range(NUMBER_OF_PEDESTRIAN):
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc:
                    spawn_point.location = loc
                    walker_spawn_points.append(spawn_point)

            for spawn_point in walker_spawn_points:
                walker_bp = random.choice(self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                walker = self.world.try_spawn_actor(walker_bp, spawn_point)
                if walker:
                    walker_controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.extend([walker_controller, walker])

            all_actors = self.world.get_actors([actor.id for actor in self.walker_list])
            for i in range(0, len(self.walker_list), 2):
                all_actors[i].start()
                all_actors[i].go_to_location(self.world.get_random_location_from_navigation())

        except Exception as e:
            print(f"Error creating pedestrians: {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.walker_list.clear()

    def set_other_vehicles(self):
        """Spawn NPC vehicles."""
        try:
            for _ in range(NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(bp_vehicle, spawn_point)
                if other_vehicle:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except Exception as e:
            print(f"Error setting other vehicles: {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])

    def change_town(self, new_town):
        """Change the simulation town."""
        self.world = self.client.load_world(new_town)
        self.map = self.world.get_map()
        self.town = new_town

    def get_world(self):
        """Return the current world."""
        return self.world

    def get_blueprint_library(self):
        """Return the blueprint library."""
        return self.blueprint_library

    def angle_diff(self, v0, v1):
        """Calculate angle difference between two vectors."""
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle <= -np.pi:
            angle += 2 * np.pi
        return angle

    def distance_to_line(self, A, B, p):
        """Calculate distance from point p to line defined by A and B."""
        num = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        return num / denom if denom > 0 else np.linalg.norm(p - A)

    def vector(self, v):
        """Convert CARLA objects to numpy arrays."""
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])
        return np.zeros(3)

    def get_discrete_action_space(self):
        """Define discrete steering action space."""
        return np.array([-0.50, -0.30, -0.10, 0.0, 0.10, 0.30, 0.50])

    def get_vehicle(self, vehicle_name):
        """Get vehicle blueprint with random color."""
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint

    def set_vehicle(self, vehicle_bp, spawn_points):
        """Spawn a vehicle at a random spawn point."""
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if self.vehicle:
            self.actor_list.append(self.vehicle)

    def remove_sensors(self):
        """Reset sensor references."""
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = []
        self.wrong_maneuver = None

    def cleanup(self):
        """Clean up all actors and sensors on error."""
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
        self.sensor_list.clear()
        self.actor_list.clear()
        self.walker_list.clear()
        self.remove_sensors()
        if self.display_on:
            pygame.quit()