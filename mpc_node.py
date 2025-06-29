import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaTrafficLightStatusList, CarlaTrafficLightInfoList
from derived_object_msgs.msg import ObjectArray  # 🔧 FIXED: Single reliable import
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from transforms3d.euler import quat2euler, euler2quat
from mpc_controller.mpc_solver import solve_corner_mpc_with_debug
import numpy as np
import time
import math

class CornerCapableMPCNode(Node):
    """
    🚀🚧 ENHANCED MPC Node with STEERING COMMITMENT + TRAFFIC LIGHT + FIXED OBSTACLE AVOIDANCE
    🔧 SURGICAL FIX: Reliable single-source obstacle detection without breaking existing logic
    """
    def __init__(self):
        super().__init__('corner_capable_mpc_node')
        self.get_logger().info('🚀🚧 ENHANCED STEERING COMMITMENT + TRAFFIC LIGHT + FIXED OBSTACLE AVOIDANCE MPC Controller Node started!')
        self.get_logger().info('   🔄 Enhanced corner handling with steering commitment logic')
        self.get_logger().info('   🎯 Anti-oscillation steering for smooth turns')
        self.get_logger().info('   🚦 ENHANCED traffic light detection with debugging and fallback')
        self.get_logger().info('   🚧 FIXED: Reliable obstacle detection for safe distance keeping')
        self.get_logger().info('   🔧 SURGICAL: Minimal changes to preserve working logic')
        
        # Vehicle parameters
        self.L = 2.5
        self.Ts = 0.1
        self.N = 10
        
        # Lane parameters - Updated for Town 10
        self.lane_center_y = 0.0           # Will be updated to target_lane_center
        self.target_lane_center = 0.0      # 🔧 NEW: Target lane center
        self.lane_width = 3.5              # Standard lane width
        self.lane_heading = 0.0            # Will be auto-detected from initial heading
            
        # State variables
        self.current_state = None
        self.initial_position_logged = False
        self.last_control_time = 0.0
        self.control_period = 0.1
        
        # Goal management (UNCHANGED - WORKING)
        self.goal_state = None
        self.user_goal_locked = False
        self.goal_set_method = "none"
        self.waypoints = []
        self.current_waypoint_index = 0
        self.goal_tolerance = 1.2
        self.waypoint_tolerance = 2.5
        
        # Turn classification (UNCHANGED - WORKING)
        self.turn_type = "UNKNOWN"
        self.turn_complexity = 0.0
        self.corner_mode = "NORMAL"
        
        # Steering commitment (UNCHANGED - WORKING)
        self.steering_commitment_active = False
        self.commitment_start_time = None
        self.commitment_target_heading = None
        self.commitment_direction = None
        self.last_commitment_status = ""
        
        # Traffic Light Integration (UNCHANGED - WORKING)
        self.traffic_light_positions = {}
        self.traffic_light_states = {}
        self.nearest_traffic_light_id = None
        self.traffic_light_status = "unknown"
        self.traffic_light_distance = float('inf')
        self.traffic_light_position = None
        self.last_traffic_light_update = 0.0
        self.traffic_light_timeout = 2.0
        
        # Traffic Light State Machine (UNCHANGED - WORKING)
        self.traffic_light_state = "NORMAL"
        self.red_light_stop_position = None
        self.stop_line_distance = 3.0
        self.red_light_approach_speed = 0.0
        
        # Traffic Light Control Parameters (UNCHANGED - WORKING)
        self.emergency_brake_distance = 2.0
        self.comfortable_brake_distance = 25.0
        self.proceed_through_distance = 8.0
        
        # Traffic Light Debug (UNCHANGED - WORKING)
        self.debug_info_callback_count = 0
        self.debug_status_callback_count = 0
        self.debug_last_status_time = 0.0
        self.debug_last_info_time = 0.0
        
        # === 🚧 FIXED: RELIABLE OBSTACLE DETECTION VARIABLES ===
        self.obstacles = []
        self.nearby_obstacles = []
        self.critical_obstacles = []
        
        # 🔧 FIXED: Conservative detection parameters
        self.detection_range = 20.0         # Reduced from 50.0m for reliability
        self.safety_distance = 2.0
        self.critical_distance = 6.0        # Reduced from 10.0m
        self.emergency_distance = 2.5       # Reduced from 5.0m
        
        # Avoidance state
        self.obstacle_avoidance_active = False
        self.avoidance_type = "none"
        self.target_obstacle = None
        
        # Performance metrics
        self.obstacles_detected_count = 0
        self.obstacle_emergency_brakes = 0
        self.obstacle_avoidance_maneuvers = 0
        
        # Control history
        self.last_delta = 0.0
        self.last_a = 0.0
        self.control_history = []
        
        # Performance monitoring (UNCHANGED - WORKING)
        self.solve_times = []
        self.success_count = 0
        self.total_count = 0
        self.goal_reached_count = 0
        
        # Steering commitment metrics (UNCHANGED - WORKING)
        self.commitment_activation_count = 0
        self.commitment_success_count = 0
        self.total_commitment_time = 0.0
        self.oscillation_prevention_count = 0
        
        # Traffic light metrics (UNCHANGED - WORKING)
        self.red_light_stops = 0
        self.yellow_light_decisions = 0
        self.emergency_brakes = 0
        self.smooth_traffic_stops = 0
        self.yellow_proceed_count = 0
        self.yellow_stop_count = 0
        
        # Other metrics (UNCHANGED - WORKING)
        self.corner_count = 0
        self.straight_count = 0
        self.sharp_turn_count = 0
        self.smooth_turn_count = 0
        self.distance_to_goal = float('inf')
        self.goal_start_time = None
        self.last_goal_update_time = 0.0
        self.mpc_failure_count = 0
        self.fallback_usage_count = 0
        self.smooth_control_count = 0
        self.aggressive_control_count = 0
        self.goal_stop_count = 0
        
        # Disable Carla autopilot
        self.autopilot_pub = self.create_publisher(Bool, '/carla/ego_vehicle/enable_autopilot', 1)
        self.create_timer(1.0, self.disable_autopilot_once)
        
        # Subscribers
        self.sub_odom = self.create_subscription(
            Odometry, 
            '/carla/ego_vehicle/odometry', 
            self.steering_commitment_odom_callback,
            10
        )
        
        # Traffic Light Integration (UNCHANGED - WORKING)
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        try:
            self.sub_traffic_light_status = self.create_subscription(
                CarlaTrafficLightStatusList,
                '/carla/traffic_lights/status',
                self.enhanced_traffic_light_status_callback,
                qos_profile
            )
            self.get_logger().info("✅ Subscribed to traffic light status with reliable QoS")
            
            self.sub_traffic_light_info = self.create_subscription(
                CarlaTrafficLightInfoList,
                '/carla/traffic_lights/info', 
                self.enhanced_traffic_light_info_callback,
                qos_profile
            )
            self.get_logger().info("✅ Subscribed to traffic light info with reliable QoS")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to subscribe to traffic light topics: {e}")
        
        # === 🚧 FIXED: SINGLE RELIABLE OBSTACLE DETECTION ===
        self.obstacle_data_source = "none"
        
        try:
            self.sub_objects = self.create_subscription(
                ObjectArray,
                '/carla/objects',
                self.fixed_objects_callback,  # 🔧 FIXED: Single reliable callback
                10
            )
            self.obstacle_data_source = "/carla/objects"
            self.get_logger().info("✅ Using reliable /carla/objects for obstacle detection")
        except Exception as e:
            self.get_logger().warn(f"⚠️ Obstacle detection disabled: {e}")
            self.obstacle_data_source = "none"
        
        # 🔧 REMOVED: Multiple conflicting subscriptions that caused false positives
        
        # Goal and waypoint subscribers (UNCHANGED - WORKING)
        self.sub_goal = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.corner_goal_callback,
            10
        )
        
        self.sub_waypoints = self.create_subscription(
            Path,
            '/waypoints',
            self.waypoints_callback,
            10
        )
        
        self.sub_simple_goal = self.create_subscription(
            PointStamped,
            '/simple_goal',
            self.simple_goal_callback,
            10
        )
        
        # Publishers (UNCHANGED - WORKING)
        self.pub_control = self.create_publisher(
            CarlaEgoVehicleControl, 
            '/carla/ego_vehicle/vehicle_control_cmd', 
            10
        )
        
        self.pub_debug_goal = self.create_publisher(
            PoseStamped,
            '/debug/corner_goal',
            10
        )
        
        self.pub_debug_turn_info = self.create_publisher(
            String,
            '/debug/turn_info',
            10
        )
        
        self.pub_debug_commitment = self.create_publisher(
            String,
            '/debug/steering_commitment',
            10
        )
        
        self.pub_traffic_light_status = self.create_publisher(
            String,
            '/debug/traffic_light_control',
            10
        )
        
        self.pub_obstacle_status = self.create_publisher(
            String,
            '/debug/obstacle_status',
            10
        )
        
        # Status and monitoring timers (UNCHANGED - WORKING)
        self.create_timer(5.0, self.print_enhanced_status)
        self.create_timer(1.0, self.corner_goal_monitor)
        self.create_timer(5.0, self.debug_traffic_light_status)
        
        self.get_logger().info("🎯 Ready for goal input via /goal_pose, /simple_goal, or /waypoints")
        self.get_logger().info("💡 Example: ros2 topic pub /goal_pose geometry_msgs/PoseStamped ...")
        self.get_logger().info(f"🚧 Reliable obstacle detection source: {self.obstacle_data_source}")
    
    # === 🚧 FIXED: RELIABLE OBSTACLE DETECTION METHODS ===
    
    def fixed_objects_callback(self, msg):
        """🔧 SURGICAL: Reliable obstacle callback using ObjectArray - preserves existing logic"""
        if self.current_state is None:
            return
        
        ego_x, ego_y, ego_theta = self.current_state[0], self.current_state[1], self.current_state[2]
        
        # 🔒 PRESERVE: Keep same obstacle reset pattern as existing code
        self.obstacles.clear()
        self.nearby_obstacles.clear()
        self.critical_obstacles.clear()
        self.obstacle_avoidance_active = False
        self.avoidance_type = "none"
        self.target_obstacle = None
        
        if not hasattr(msg, 'objects') or not msg.objects:
            return
        
        # 🔧 FIXED: Reliable parsing for ObjectArray (based on validated test data)
        classification_map = {
            4: 'pedestrian', 5: 'bike', 6: 'car', 7: 'truck', 
            8: 'motorcycle', 9: 'other_vehicle', 10: 'barrier', 11: 'sign'
        }
        
        valid_obstacles = 0
        
        for obj in msg.objects:
            try:
                # 🔧 FIXED: Reliable data extraction (confirmed working from test data)
                obj_id = obj.id
                obj_type = classification_map.get(obj.classification, 'unknown')
                obj_x = obj.pose.position.x
                obj_y = obj.pose.position.y
                obj_z = obj.pose.position.z
                
                # Skip unknown objects and ensure proper classification
                if obj_type == 'unknown' or not obj.object_classified:
                    continue
                
                # Calculate distance
                distance = np.sqrt((obj_x - ego_x)**2 + (obj_y - ego_y)**2)
                
                # 🔒 PRESERVE: Use same distance thresholds as before
                if distance < 1.0 or distance > self.detection_range:
                    continue
                
                # 🔒 PRESERVE: Use same obstacle_info structure as existing code
                obstacle_info = {
                    'id': obj_id,
                    'type': obj_type,
                    'x': obj_x,
                    'y': obj_y,
                    'z': obj_z,
                    'distance': distance,
                    'is_ahead': self.is_obstacle_ahead(ego_x, ego_y, ego_theta, obj_x, obj_y),
                    'size': self.estimate_obstacle_size(obj_type),
                    'threat_level': 0.0,
                    'relative_angle': self.calculate_relative_angle(ego_x, ego_y, ego_theta, obj_x, obj_y)
                }
                
                self.obstacles.append(obstacle_info)
                valid_obstacles += 1
                
                # 🔒 PRESERVE: Use same threat categorization as existing code
                if obstacle_info['is_ahead']:
                    if distance <= self.emergency_distance:
                        obstacle_info['threat_level'] = 1.0
                        self.critical_obstacles.append(obstacle_info)
                    elif distance <= self.critical_distance:
                        obstacle_info['threat_level'] = 0.7
                        self.critical_obstacles.append(obstacle_info)
                    elif distance <= self.detection_range * 0.5:
                        obstacle_info['threat_level'] = 0.3
                        self.nearby_obstacles.append(obstacle_info)
                        
            except Exception as e:
                continue
        
        # Update obstacle count
        self.obstacles_detected_count = valid_obstacles
        
        # 🔒 PRESERVE: Use existing avoidance update method
        if self.critical_obstacles:
            self.update_safe_obstacle_avoidance()
        
        # 🔒 PRESERVE: Same logging pattern as before
        log_counter = getattr(self, '_obstacle_log_counter', 0) + 1
        self._obstacle_log_counter = log_counter
        
        if log_counter % 30 == 1:
            self.get_logger().info(
                f"🚧 RELIABLE OBSTACLES: valid={valid_obstacles}, critical={len(self.critical_obstacles)} "
                f"[ObjectArray source]"
            )
            
            if self.critical_obstacles:
                closest = min(self.critical_obstacles, key=lambda x: x['distance'])
                self.get_logger().info(
                    f"   Closest: {closest['type']} at {closest['distance']:.1f}m "
                    f"(ID: {closest['id']}, threat: {closest['threat_level']:.1f})"
                )
        
        # 🔒 PRESERVE: Use existing publish method
        self.publish_obstacle_status()

    def update_safe_obstacle_avoidance(self):
        """🔧 FIXED: Safe obstacle avoidance with extra validation to prevent false positives"""
        
        # Double-check critical obstacles are truly valid
        truly_critical = []
        for obs in self.critical_obstacles:
            # Extra strict validation to prevent false positives
            if (obs['distance'] >= 1.5 and          # Minimum safe distance
                obs['distance'] <= 25.0 and         # Maximum reasonable distance  
                obs['is_ahead'] and                  # Must be ahead
                obs['type'] != 'unknown' and         # Must have valid type
                obs['type'] != '' and                # Must not be empty
                isinstance(obs['id'], int) and       # Must have valid ID
                obs['id'] > 0):                      # Must not be ego (ID=0)
                truly_critical.append(obs)
        
        if not truly_critical:
            return  # No truly critical obstacles
        
        # Find most dangerous from validated list
        most_dangerous = max(truly_critical, key=lambda x: x['threat_level'])
        
        # Activate avoidance only if distance is reasonable
        self.obstacle_avoidance_active = True
        self.target_obstacle = most_dangerous
        
        if most_dangerous['distance'] <= self.emergency_distance:
            self.avoidance_type = "emergency_brake"
            self.obstacle_emergency_brakes += 1
            self.get_logger().warn(
                f"🚨 VALIDATED EMERGENCY: {most_dangerous['type']} (ID:{most_dangerous['id']}) "
                f"at {most_dangerous['distance']:.1f}m"
            )
        elif most_dangerous['distance'] <= self.critical_distance:
            self.avoidance_type = "brake"
            self.get_logger().info(
                f"🚧 VALIDATED BRAKE: {most_dangerous['type']} (ID:{most_dangerous['id']}) "
                f"at {most_dangerous['distance']:.1f}m"
            )
        
        self.obstacle_avoidance_maneuvers += 1

    def is_obstacle_ahead(self, ego_x, ego_y, ego_theta, obs_x, obs_y):
        """Check if obstacle is ahead of ego vehicle (UNCHANGED - WORKING)"""
        # Vector from ego to obstacle
        dx = obs_x - ego_x
        dy = obs_y - ego_y
        
        # Ego vehicle forward direction
        ego_forward_x = np.cos(ego_theta)
        ego_forward_y = np.sin(ego_theta)
        
        # Dot product to check if obstacle is ahead
        dot_product = dx * ego_forward_x + dy * ego_forward_y
        
        # Consider ahead if within 60° cone
        distance = np.sqrt(dx**2 + dy**2)
        if distance > 0:
            cos_angle = dot_product / distance
            return cos_angle > 0.5  # 60° cone ahead
        
        return False

    def calculate_relative_angle(self, ego_x, ego_y, ego_theta, obs_x, obs_y):
        """Calculate relative angle of obstacle (UNCHANGED - WORKING)"""
        dx = obs_x - ego_x
        dy = obs_y - ego_y
        angle_to_obstacle = np.arctan2(dy, dx)
        relative_angle = self.normalize_angle(angle_to_obstacle - ego_theta)
        return relative_angle

    def estimate_obstacle_size(self, actor_type):
        """Estimate obstacle size for safety calculations (UNCHANGED - WORKING)"""
        if not isinstance(actor_type, str):
            return 2.5
            
        actor_type_lower = actor_type.lower()
        
        if 'vehicle' in actor_type_lower or actor_type_lower in ['car', 'truck', 'motorcycle']:
            if actor_type_lower in ['truck']:
                return 8.0
            elif actor_type_lower in ['motorcycle', 'bike']:
                return 2.0
            else:
                return 4.5
        elif actor_type_lower in ['pedestrian']:
            return 1.0
        elif actor_type_lower in ['barrier', 'sign']:
            return 3.0
        else:
            return 2.5

    def publish_obstacle_status(self):
        """Publish obstacle status for debugging (UNCHANGED - WORKING)"""
        if hasattr(self, 'pub_obstacle_status'):
            status_msg = String()
            
            if self.obstacle_avoidance_active and self.target_obstacle:
                status_msg.data = (
                    f"{self.avoidance_type}|{self.target_obstacle['type']}|"
                    f"{self.target_obstacle['distance']:.1f}|{len(self.critical_obstacles)}|"
                    f"{len(self.nearby_obstacles)}|{self.obstacles_detected_count}"
                )
            else:
                status_msg.data = (
                    f"none|none|inf|{len(self.critical_obstacles)}|"
                    f"{len(self.nearby_obstacles)}|{self.obstacles_detected_count}"
                )
            
            self.pub_obstacle_status.publish(status_msg)
    
    # === 🚦 ENHANCED: CARLA Traffic Light Integration Methods (UNCHANGED - WORKING) ===
    
    def enhanced_traffic_light_status_callback(self, msg):
        """🚦 ENHANCED: Traffic light status callback (UNCHANGED - WORKING)"""
        try:
            self.debug_status_callback_count += 1
            self.debug_last_status_time = time.time()
            
            if self.debug_status_callback_count <= 5 or self.debug_status_callback_count % 50 == 0:
                self.get_logger().info(f"🚦 STATUS CALLBACK #{self.debug_status_callback_count}: {len(msg.traffic_lights)} lights")
            
            for i, tl_status in enumerate(msg.traffic_lights):
                self.traffic_light_states[tl_status.id] = tl_status.state
                
                if i < 3 and self.debug_status_callback_count <= 3:
                    state_names = ["RED", "YELLOW", "GREEN", "OFF", "UNKNOWN"]
                    state_str = state_names[min(tl_status.state, 4)]
                    self.get_logger().info(f"🚦 TL {tl_status.id}: state={tl_status.state} ({state_str})")
            
            self.find_relevant_traffic_light()
            self.update_traffic_light_state_machine()
            
        except Exception as e:
            self.get_logger().error(f"❌ Error in status callback: {e}")
    
    def enhanced_traffic_light_info_callback(self, msg):
        """🚦 ENHANCED: Traffic light info callback (UNCHANGED - WORKING)"""
        try:
            self.debug_info_callback_count += 1
            self.debug_last_info_time = time.time()
            
            if self.debug_info_callback_count <= 5 or self.debug_info_callback_count % 50 == 0:
                self.get_logger().info(f"🚦 INFO CALLBACK #{self.debug_info_callback_count}: {len(msg.traffic_lights)} positions")
            
            for i, tl_info in enumerate(msg.traffic_lights):
                pos = tl_info.transform.position
                self.traffic_light_positions[tl_info.id] = (pos.x, pos.y, pos.z)
                
                if i < 3 and self.debug_info_callback_count <= 3:
                    self.get_logger().info(f"🚦 TL {tl_info.id}: pos=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})")
            
            self.find_relevant_traffic_light()
            
        except Exception as e:
            self.get_logger().error(f"❌ Error in info callback: {e}")
    
    def debug_traffic_light_status(self):
        """🚦 DEBUG: Print traffic light debugging info (UNCHANGED - WORKING)"""
        current_time = time.time()
        
        status_age = current_time - self.debug_last_status_time if self.debug_last_status_time > 0 else float('inf')
        info_age = current_time - self.debug_last_info_time if self.debug_last_info_time > 0 else float('inf')
        
        self.get_logger().info(f"🚦 DEBUG STATUS:")
        self.get_logger().info(f"   Status callbacks: {self.debug_status_callback_count} (last: {status_age:.1f}s ago)")
        self.get_logger().info(f"   Info callbacks: {self.debug_info_callback_count} (last: {info_age:.1f}s ago)")
        self.get_logger().info(f"   Positions stored: {len(self.traffic_light_positions)}")
        self.get_logger().info(f"   States stored: {len(self.traffic_light_states)}")
        self.get_logger().info(f"   Current TL: {self.traffic_light_status} at {self.traffic_light_distance:.1f}m")
        self.get_logger().info(f"   TL State: {self.traffic_light_state}")
        
        if len(self.traffic_light_states) > 0 and len(self.traffic_light_positions) == 0:
            self.get_logger().info("🔄 Using ENHANCED FALLBACK mode: States available but no positions")
        elif len(self.traffic_light_states) == 0:
            self.get_logger().warn("⚠️ No traffic light data received!")
        
        if len(self.traffic_light_states) > 0:
            sample_states = list(self.traffic_light_states.items())[:3]
            state_names = ["RED", "YELLOW", "GREEN", "OFF", "UNKNOWN"]
            for tl_id, state in sample_states:
                state_str = state_names[min(state, 4)]
                self.get_logger().info(f"   Sample TL {tl_id}: {state_str}")
    
    def find_relevant_traffic_light(self):
        """🎯 ENHANCED: Find nearest traffic light (UNCHANGED - WORKING)"""
        if self.current_state is None:
            return
           
        x_curr, y_curr, theta_curr = self.current_state[0], self.current_state[1], self.current_state[2]
        
        try:
            if len(self.traffic_light_positions) == 0:
                if len(self.traffic_light_states) > 0:
                    self.enhanced_fallback_traffic_light_detection(x_curr, y_curr, theta_curr)
                    return
                else:
                    self.reset_traffic_light_state()
                    return
            
            self.position_based_traffic_light_detection(x_curr, y_curr, theta_curr)
            
        except Exception as e:
            self.get_logger().error(f"❌ Error in find_relevant_traffic_light: {e}")
            self.reset_traffic_light_state()
    
    def enhanced_fallback_traffic_light_detection(self, x_curr, y_curr, theta_curr):
        """🚦 ENHANCED: Smart fallback detection (UNCHANGED - WORKING)"""
        current_speed = self.current_state[3] if len(self.current_state) > 3 else 0.0
        
        red_lights = [(tl_id, 0) for tl_id, state in self.traffic_light_states.items() if state == 0]
        yellow_lights = [(tl_id, 1) for tl_id, state in self.traffic_light_states.items() if state == 1]
        green_lights = [(tl_id, 2) for tl_id, state in self.traffic_light_states.items() if state == 2]
        
        if red_lights:
            chosen_id, state_int = red_lights[0]
            if current_speed > 8.0:
                estimated_distance = 35.0
            elif current_speed > 4.0:
                estimated_distance = 25.0
            else:
                estimated_distance = 15.0
            chosen_data = (chosen_id, "red", estimated_distance, "RED_PRIORITY_ENHANCED")
            
        elif yellow_lights:
            chosen_id, state_int = yellow_lights[0]
            estimated_distance = min(20.0, max(12.0, current_speed * 2.0))
            chosen_data = (chosen_id, "yellow", estimated_distance, "YELLOW_PRIORITY_ENHANCED")
            
        elif green_lights:
            chosen_id, state_int = green_lights[0]
            estimated_distance = min(50.0, max(20.0, current_speed * 3.0))
            chosen_data = (chosen_id, "green", estimated_distance, "GREEN_AVAILABLE_ENHANCED")
        else:
            self.reset_traffic_light_state()
            return
        
        chosen_id, chosen_state, estimated_distance, reason = chosen_data
        
        self.nearest_traffic_light_id = chosen_id
        self.traffic_light_status = chosen_state
        self.traffic_light_distance = estimated_distance
        self.traffic_light_position = None
        self.last_traffic_light_update = time.time()
        
        log_counter = getattr(self, '_enhanced_fallback_log_counter', 0) + 1
        self._enhanced_fallback_log_counter = log_counter
        
        if log_counter % 15 == 1:
            total_lights = len(self.traffic_light_states)
            self.get_logger().info(
                f"🚦 ENHANCED FALLBACK: TL {chosen_id} ({chosen_state.upper()}) "
                f"at ~{estimated_distance:.1f}m [speed-based: {current_speed:.1f}m/s]"
            )
            self.get_logger().info(f"   Available: {len(red_lights)}R, {len(yellow_lights)}Y, {len(green_lights)}G (total: {total_lights})")

    def is_traffic_light_ahead(self, vehicle_x, vehicle_y, vehicle_heading, tl_x, tl_y):
        """🎯 Check if traffic light is ahead (UNCHANGED - WORKING)"""
        vehicle_dir_x = math.cos(vehicle_heading)
        vehicle_dir_y = math.sin(vehicle_heading)
        
        to_tl_x = tl_x - vehicle_x
        to_tl_y = tl_y - vehicle_y
        
        distance = math.sqrt(to_tl_x**2 + to_tl_y**2)
        if distance > 0:
            to_tl_x /= distance
            to_tl_y /= distance
        else:
            return False
        
        dot_product = vehicle_dir_x * to_tl_x + vehicle_dir_y * to_tl_y
        is_ahead = dot_product > 0.7
        
        return is_ahead

    def position_based_traffic_light_detection(self, x_curr, y_curr, theta_curr):
        """🎯 ENHANCED: Position-based detection (UNCHANGED - WORKING)"""
        relevant_lights = []
        total_lights = len(self.traffic_light_positions)
        ahead_count = 0
        filtered_count = 0
        
        for tl_id in self.traffic_light_positions.keys():
            if tl_id in self.traffic_light_states:
                tl_x, tl_y, tl_z = self.traffic_light_positions[tl_id]
                tl_state_int = self.traffic_light_states[tl_id]
                
                if not self.is_traffic_light_ahead(x_curr, y_curr, theta_curr, tl_x, tl_y):
                    filtered_count += 1
                    continue
                
                ahead_count += 1
                
                distance = math.sqrt((tl_x - x_curr)**2 + (tl_y - y_curr)**2)
                
                if distance > 100.0:
                    continue
                if distance < 3.0:
                    continue
                
                state_str = ["red", "yellow", "green", "unknown", "unknown"][min(tl_state_int, 4)]
                
                if state_str == "red":
                    priority = distance
                elif state_str == "yellow":
                    priority = 1000 + distance
                elif state_str == "green":
                    priority = 2000 + distance
                else:
                    priority = 9999
                
                relevant_lights.append({
                    'id': tl_id,
                    'distance': distance,
                    'state': state_str,
                    'position': (tl_x, tl_y),
                    'priority': priority
                })
        
        if relevant_lights:
            relevant_lights.sort(key=lambda x: x['priority'])
            best_light = relevant_lights[0]
            
            self.nearest_traffic_light_id = best_light['id']
            self.traffic_light_status = best_light['state']
            self.traffic_light_distance = best_light['distance']
            self.traffic_light_position = best_light['position']
            self.last_traffic_light_update = time.time()
            
            log_counter = getattr(self, '_position_log_counter', 0) + 1
            self._position_log_counter = log_counter
            
            if log_counter % 20 == 1 or best_light['state'] in ["red", "yellow"]:
                tl_x, tl_y = best_light['position']
                self.get_logger().info(
                    f"🎯 AHEAD-ONLY FILTER: TL {best_light['id']} ({best_light['state'].upper()}) "
                    f"at {best_light['distance']:.1f}m, pos=({tl_x:.1f}, {tl_y:.1f})"
                )
                self.get_logger().info(
                    f"   📊 Filter stats: {total_lights} total → {ahead_count} ahead → {len(relevant_lights)} valid"
                )
        else:
            self.reset_traffic_light_state()
    
    def reset_traffic_light_state(self):
        """🚦 Reset traffic light state (UNCHANGED - WORKING)"""
        self.nearest_traffic_light_id = None
        self.traffic_light_status = "unknown"
        self.traffic_light_distance = float('inf')
        self.traffic_light_position = None
        self.traffic_light_state = "NORMAL"
    
    def update_traffic_light_state_machine(self):
        """🚦 ENHANCED: Traffic light state machine (UNCHANGED - WORKING)"""
        if self.current_state is None:
            return
        
        current_speed = self.current_state[3]
        
        if time.time() - self.last_traffic_light_update > self.traffic_light_timeout:
            self.traffic_light_status = "unknown"
            self.traffic_light_state = "NORMAL"
            return
        
        if self.traffic_light_status == "red":
            if self.traffic_light_distance > 30.0:
                self.traffic_light_state = "NORMAL"
            elif self.traffic_light_distance > 5.0:
                self.traffic_light_state = "APPROACHING_RED"
                if not hasattr(self, '_red_light_logged'):
                    self.red_light_stops += 1
                    self._red_light_logged = True
                    self.get_logger().info(f"🚦 RED LIGHT DETECTED: Starting controlled approach at {self.traffic_light_distance:.1f}m")
            else:
                self.traffic_light_state = "STOPPED_AT_RED"
                if self.red_light_stop_position is None:
                    self.red_light_stop_position = (self.current_state[0], self.current_state[1])
                    self.get_logger().info(f"🔴 STOPPED AT RED LIGHT: Position logged")
        
        elif self.traffic_light_status == "yellow":
            stopping_distance = self.calculate_stopping_distance(current_speed)
            comfort_margin = 5.0
            
            time_to_intersection = self.traffic_light_distance / max(current_speed, 0.1)
            estimated_yellow_remaining = 3.0
            
            self.yellow_light_decisions += 1
            
            if self.traffic_light_distance < self.proceed_through_distance:
                self.traffic_light_state = "PROCEEDING_YELLOW"
                self.yellow_proceed_count += 1
                self.get_logger().info(f"🟡 YELLOW DECISION: PROCEED (too close: {self.traffic_light_distance:.1f}m)")
            elif self.traffic_light_distance > (stopping_distance + comfort_margin):
                self.traffic_light_state = "APPROACHING_RED"
                self.yellow_stop_count += 1
                self.get_logger().info(f"🟡 YELLOW DECISION: STOP (safe stop: {self.traffic_light_distance:.1f}m > {stopping_distance + comfort_margin:.1f}m)")
            elif time_to_intersection < estimated_yellow_remaining:
                self.traffic_light_state = "PROCEEDING_YELLOW"
                self.yellow_proceed_count += 1
                self.get_logger().info(f"🟡 YELLOW DECISION: PROCEED (time: {time_to_intersection:.1f}s < {estimated_yellow_remaining:.1f}s)")
            else:
                self.traffic_light_state = "APPROACHING_RED"
                self.yellow_stop_count += 1
                self.get_logger().info(f"🟡 YELLOW DECISION: STOP (border case - safety first)")
        
        elif self.traffic_light_status == "green":
            if self.traffic_light_distance < 50.0:
                self.traffic_light_state = "PROCEEDING_GREEN"
            else:
                self.traffic_light_state = "NORMAL"
            
            self.red_light_stop_position = None
            if hasattr(self, '_red_light_logged'):
                delattr(self, '_red_light_logged')
        
        else:
            self.traffic_light_state = "NORMAL"
    
    def calculate_stopping_distance(self, current_speed):
        """🚦 Calculate safe stopping distance (UNCHANGED - WORKING)"""
        reaction_time = 0.5
        max_deceleration = 4.0
        
        reaction_distance = current_speed * reaction_time
        braking_distance = (current_speed ** 2) / (2 * max_deceleration)
        
        return reaction_distance + braking_distance + 2.0
    
    # === GOAL MANAGEMENT METHODS (UNCHANGED - WORKING) ===
    
    def debug_initial_position_once(self):
        """🔧 Debug initial position (UNCHANGED - WORKING)"""
        if self.current_state is not None and not self.initial_position_logged:
            x, y, yaw, v = self.current_state
            heading_deg = np.degrees(yaw)
            
            self.get_logger().info("\n" + "="*70)
            self.get_logger().info("🗺️ ENHANCED STEERING COMMITMENT + TRAFFIC LIGHT + FIXED OBSTACLE AVOIDANCE DEBUG - INITIAL POSITION")
            self.get_logger().info("="*70)
            self.get_logger().info(f"📍 Vehicle Position: ({x:.1f}, {y:.1f})")
            self.get_logger().info(f"🧭 Vehicle Heading: {heading_deg:.1f}° ({self.get_cardinal_direction(heading_deg)})")
            self.get_logger().info(f"🚗 Vehicle Speed: {v:.1f} m/s")
            
            # 🔧 NEW: Determine target lane instead of using center
            self.target_lane_center = self.determine_target_lane(heading_deg)
            self.lane_heading = yaw
            
            self.get_logger().info(f"🔍 COORDINATE DEBUG:")
            self.get_logger().info(f"   Current vehicle Y: {y:.1f}m")
            self.get_logger().info(f"   Target lane Y: {self.target_lane_center:.1f}m") 
            self.get_logger().info(f"   Lane deviation: {y - self.target_lane_center:.1f}m")
            self.get_logger().info(f"   Expected pull direction: {'UP' if y < self.target_lane_center else 'DOWN'}")
            
            self.suggest_corner_tests(x, y, heading_deg)
            
            if not self.user_goal_locked and self.goal_state is None:
                self.get_logger().info("🎯 No user goal detected - setting default corner test goal")
                self.auto_set_corner_goal(x, y, heading_deg)
                self.goal_set_method = "auto"
            else:
                if self.user_goal_locked:
                    self.get_logger().info(f"🔒 USER GOAL LOCKED: {self.goal_state} - skipping auto-set")
                else:
                    self.get_logger().info(f"🎯 Existing goal detected: {self.goal_state} - skipping auto-set")
            
            self.initial_position_logged = True
            
    def get_cardinal_direction(self, heading_deg):
        """Convert heading to cardinal direction (UNCHANGED - WORKING)"""
        heading_normalized = heading_deg % 360
        
        if heading_normalized < 22.5 or heading_normalized >= 337.5:
            return "East"
        elif 22.5 <= heading_normalized < 67.5:
            return "Northeast"
        elif 67.5 <= heading_normalized < 112.5:
            return "North"
        elif 112.5 <= heading_normalized < 157.5:
            return "Northwest"
        elif 157.5 <= heading_normalized < 202.5:
            return "West"
        elif 202.5 <= heading_normalized < 247.5:
            return "Southwest"
        elif 247.5 <= heading_normalized < 292.5:
            return "South"
        elif 292.5 <= heading_normalized < 337.5:
            return "Southeast"
        else:
            return "Unknown"
    
    def suggest_corner_tests(self, x_curr, y_curr, heading_deg):
        """Suggest corner tests (UNCHANGED - WORKING)"""
        self.get_logger().info("\n🎯 SUGGESTED ENHANCED STEERING COMMITMENT + TRAFFIC LIGHT + FIXED OBSTACLE AVOIDANCE TESTS for Town 10:")
        
        heading_normalized = heading_deg % 360
        
        if abs(heading_normalized) < 45 or abs(heading_normalized - 360) < 45:
            self.get_logger().info(f"   🔄 90° Right Turn: [{x_curr + 80:.1f}, {y_curr + 80:.1f}, 90°] (East→North)")
            self.get_logger().info(f"   🔄 90° Left Turn:  [{x_curr + 80:.1f}, {y_curr - 80:.1f}, -90°] (East→South)")
            self.get_logger().info(f"   🔄 U-Turn:        [{x_curr + 80:.1f}, {y_curr:.1f}, 180°] (East→West)")
            self.get_logger().info(f"   ➡️ Straight:      [{x_curr + 60:.1f}, {y_curr:.1f}, 0°] (Straight ahead)")
        elif abs(heading_normalized - 90) < 45:
            self.get_logger().info(f"   🔄 90° Right Turn: [{x_curr + 80:.1f}, {y_curr + 80:.1f}, 0°] (North→East)")
            self.get_logger().info(f"   🔄 90° Left Turn:  [{x_curr - 80:.1f}, {y_curr + 80:.1f}, 180°] (North→West)")
            self.get_logger().info(f"   🔄 U-Turn:        [{x_curr:.1f}, {y_curr + 80:.1f}, -90°] (North→South)")
            self.get_logger().info(f"   ➡️ Straight:      [{x_curr:.1f}, {y_curr + 60:.1f}, 90°] (Straight ahead)")
        elif abs(heading_normalized - 180) < 45 or abs(heading_normalized + 180) < 45:
            self.get_logger().info(f"   🔄 90° Right Turn: [{x_curr - 80:.1f}, {y_curr + 80:.1f}, 90°] (West→North)")
            self.get_logger().info(f"   🔄 90° Left Turn:  [{x_curr - 80:.1f}, {y_curr - 80:.1f}, -90°] (West→South)")
            self.get_logger().info(f"   🔄 U-Turn:        [{x_curr - 80:.1f}, {y_curr:.1f}, 0°] (West→East)")
            self.get_logger().info(f"   ➡️ Straight:      [{x_curr - 60:.1f}, {y_curr:.1f}, 180°] (Straight ahead)")
        elif abs(heading_normalized + 90) < 45 or abs(heading_normalized - 270) < 45:
            self.get_logger().info(f"   🔄 90° Right Turn: [{x_curr - 80:.1f}, {y_curr - 80:.1f}, 180°] (South→West)")
            self.get_logger().info(f"   🔄 90° Left Turn:  [{x_curr + 80:.1f}, {y_curr - 80:.1f}, 0°] (South→East)")
            self.get_logger().info(f"   🔄 U-Turn:        [{x_curr:.1f}, {y_curr - 80:.1f}, 90°] (South→North)")
            self.get_logger().info(f"   ➡️ Straight:      [{x_curr:.1f}, {y_curr - 60:.1f}, -90°] (Straight ahead)")
        
        self.get_logger().info("💡 Enhanced steering commitment will prevent oscillating turns!")
        self.get_logger().info("💡 Enhanced traffic light control will ensure safe stops!")
        self.get_logger().info("💡 FIXED obstacle avoidance will maintain safe distances!")
        self.get_logger().info("💡 To test, publish to /goal_pose or update set_corner_test_goal()")
        self.get_logger().info("="*70)
    
    def auto_set_corner_goal(self, x_curr, y_curr, heading_deg):
        """🔧 Auto-set goal (UNCHANGED - WORKING)"""
        if self.user_goal_locked:
            self.get_logger().info("🔒 User goal locked - skipping auto-set")
            return
        
        goal_x = 250.0
        goal_y = -0.1
        goal_heading = 0.00
        
        turn_description = f"MANUAL GOAL ({goal_x}, {goal_y})"
        
        self.goal_state = [goal_x, goal_y, goal_heading]
        self.goal_start_time = time.time()
        self.goal_set_method = "auto"
        
        self.get_logger().info(f"\n🎯 MANUAL GOAL SET: {turn_description}")
        self.get_logger().info(f"   Goal: ({goal_x:.1f}, {goal_y:.1f}, {np.degrees(goal_heading):.1f}°)")
        self.get_logger().info("   🚀 Enhanced MPC will use steering commitment to prevent oscillations")
        self.get_logger().info("   🚦 Enhanced traffic light control will handle red/yellow/green lights")
        self.get_logger().info("   🚧 FIXED obstacle avoidance will maintain safe distances")
        self.get_logger().info("   💡 Publish to /goal_pose to override with your own goal")
        self.publish_debug_goal()
    
    def corner_goal_callback(self, msg):
        """🔧 Enhanced goal callback (UNCHANGED - WORKING)"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        
        q = msg.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
        
        if abs(x) > 1000 or abs(y) > 1000:
            self.get_logger().warn(f"⚠️ Goal position out of bounds: ({x:.1f}, {y:.1f}) - ignored")
            return
        
        self.goal_state = [x, y, yaw]
        self.goal_start_time = time.time()
        self.current_waypoint_index = 0
        self.user_goal_locked = True
        self.goal_set_method = "user"
        
        self.reset_steering_commitment_state()
        
        if self.current_state is not None:
            self.analyze_corner_type()
            
            dist = np.sqrt((x - self.current_state[0])**2 + (y - self.current_state[1])**2)
            self.get_logger().info(f"🔒 USER GOAL LOCKED: ({x:.1f}, {y:.1f}, {np.degrees(yaw):.1f}°)")
            self.get_logger().info(f"   Distance: {dist:.1f}m, Turn type: {self.turn_type}, Complexity: {self.turn_complexity:.2f}")
            self.get_logger().info(f"   🚀 Enhanced steering commitment ready for anti-oscillation control")
            self.get_logger().info(f"   🚦 Enhanced traffic light control ready for smart stops")
            self.get_logger().info(f"   🚧 FIXED obstacle avoidance ready for safe distance keeping")
            self.get_logger().info(f"   🔒 Auto-goal override disabled")
        else:
            self.get_logger().info(f"🔒 USER GOAL LOCKED: ({x:.1f}, {y:.1f}, {np.degrees(yaw):.1f}°)")
        
        self.publish_debug_goal()
    
    def simple_goal_callback(self, msg):
        """🔧 Enhanced simple goal callback (UNCHANGED - WORKING)"""
        x = msg.point.x
        y = msg.point.y
        
        if abs(x) > 1000 or abs(y) > 1000:
            self.get_logger().warn(f"⚠️ Simple goal out of bounds: ({x:.1f}, {y:.1f}) - ignored")
            return
        
        goal_yaw = self.lane_heading
        
        self.goal_state = [x, y, goal_yaw]
        self.goal_start_time = time.time()
        self.current_waypoint_index = 0
        self.user_goal_locked = True
        self.goal_set_method = "user_simple"
        
        self.reset_steering_commitment_state()
        
        if self.current_state is not None:
            self.analyze_corner_type()
            dist = np.sqrt((x - self.current_state[0])**2 + (y - self.current_state[1])**2)
            self.get_logger().info(f"🔒 SIMPLE USER GOAL LOCKED: ({x:.1f}, {y:.1f})")
            self.get_logger().info(f"   Distance: {dist:.1f}m, Turn type: {self.turn_type}")
            self.get_logger().info(f"   🔒 Auto-goal override disabled")
        else:
            self.get_logger().info(f"🔒 SIMPLE USER GOAL LOCKED: ({x:.1f}, {y:.1f})")
        
        self.publish_debug_goal()
    
    def waypoints_callback(self, msg):
        """Handle waypoint path (UNCHANGED - WORKING)"""
        self.waypoints = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.waypoints.append((x, y))
        
        self.current_waypoint_index = 0
        self.get_logger().info(f"🗺️ NEW ENHANCED STEERING COMMITMENT + TRAFFIC LIGHT + FIXED OBSTACLE AVOIDANCE WAYPOINTS: {len(self.waypoints)} points received")
        
        if len(self.waypoints) > 0:
            final_x, final_y = self.waypoints[-1]
            
            self.goal_state = [final_x, final_y, self.lane_heading]
            self.goal_start_time = time.time()
            self.user_goal_locked = True
            self.goal_set_method = "waypoint"
            
            self.reset_steering_commitment_state()
            
            self.get_logger().info(f"🔒 WAYPOINT GOAL LOCKED: ({final_x:.1f}, {final_y:.1f})")
    
    def reset_steering_commitment_state(self):
        """🚀 Reset steering commitment state (UNCHANGED - WORKING)"""
        self.steering_commitment_active = False
        self.commitment_start_time = None
        self.commitment_target_heading = None
        self.commitment_direction = None
        self.last_commitment_status = ""
        self.get_logger().info("🔄 Steering commitment state reset for new goal")
    
    def update_steering_commitment_tracking(self):
        """🚀 Update steering commitment tracking (UNCHANGED - WORKING)"""
        solver_state = getattr(solve_corner_mpc_with_debug, '_steering_commitment_state', None)
        
        if solver_state is not None:
            was_active = self.steering_commitment_active
            self.steering_commitment_active = solver_state.get('is_committed', False)
            
            if not was_active and self.steering_commitment_active:
                self.commitment_activation_count += 1
                self.commitment_start_time = time.time()
                self.commitment_target_heading = solver_state.get('target_heading', None)
                
                direction_sign = solver_state.get('commitment_direction', 0.0)
                if direction_sign > 0:
                    self.commitment_direction = "LEFT"
                elif direction_sign < 0:
                    self.commitment_direction = "RIGHT"
                else:
                    self.commitment_direction = "NONE"
                
                self.get_logger().info(f"🚀 STEERING COMMITMENT ACTIVATED #{self.commitment_activation_count}")
                self.get_logger().info(f"   Direction: {self.commitment_direction}")
                if self.commitment_target_heading is not None:
                    self.get_logger().info(f"   Target heading: {np.degrees(self.commitment_target_heading):.1f}°")
            
            elif was_active and not self.steering_commitment_active:
                if self.commitment_start_time is not None:
                    commitment_duration = time.time() - self.commitment_start_time
                    self.total_commitment_time += commitment_duration
                    self.commitment_success_count += 1
                    
                    self.get_logger().info(f"🏁 STEERING COMMITMENT COMPLETED #{self.commitment_success_count}")
                    self.get_logger().info(f"   Duration: {commitment_duration:.1f}s")
                    self.get_logger().info(f"   Total commitment time: {self.total_commitment_time:.1f}s")
                
                self.commitment_start_time = None
                self.commitment_target_heading = None
                self.commitment_direction = None
        
        if self.steering_commitment_active:
            duration = time.time() - self.commitment_start_time if self.commitment_start_time else 0.0
            status = f"ACTIVE|{self.commitment_direction}|{duration:.1f}s"
            if self.commitment_target_heading is not None:
                status += f"|{np.degrees(self.commitment_target_heading):.1f}°"
        else:
            status = f"INACTIVE|activations:{self.commitment_activation_count}|successes:{self.commitment_success_count}"
        
        if status != self.last_commitment_status:
            commitment_msg = String()
            commitment_msg.data = status
            self.pub_debug_commitment.publish(commitment_msg)
            self.last_commitment_status = status
    
    def analyze_corner_type(self):
        """🔧 Corner type analysis (UNCHANGED - WORKING)"""
        if self.current_state is None or self.goal_state is None:
            return
        
        x_curr, y_curr, theta_curr, v_curr = self.current_state
        x_goal, y_goal = self.goal_state[0], self.goal_state[1]
        theta_goal = self.goal_state[2] if len(self.goal_state) > 2 else None
        
        dx = x_goal - x_curr
        dy = y_goal - y_curr
        direction_to_goal = np.arctan2(dy, dx)
        
        heading_to_goal_error = self.normalize_angle(direction_to_goal - theta_curr)
        
        if theta_goal is not None:
            final_heading_error = self.normalize_angle(theta_goal - theta_curr)
            total_heading_change = abs(np.degrees(heading_to_goal_error)) + abs(np.degrees(final_heading_error))
        else:
            final_heading_error = 0.0
            total_heading_change = abs(np.degrees(heading_to_goal_error))
        
        if total_heading_change < 8.0:
            self.turn_type = "STRAIGHT"
            self.turn_complexity = 0.0
        elif total_heading_change < 20.0:
            self.turn_type = "GENTLE_TURN"
            self.turn_complexity = total_heading_change / 90.0
        elif total_heading_change < 50.0:
            self.turn_type = "NORMAL_TURN"  
            self.turn_complexity = total_heading_change / 90.0
        elif total_heading_change < 100.0:
            self.turn_type = "SHARP_TURN"
            self.turn_complexity = min(1.0, total_heading_change / 90.0)
        else:
            self.turn_type = "U_TURN"
            self.turn_complexity = 1.0
        
        dist_to_goal = np.sqrt(dx**2 + dy**2)
        
        if dist_to_goal < 3.0:
            self.corner_mode = "FINAL_APPROACH"
        elif self.turn_type in ["SHARP_TURN", "U_TURN"] and dist_to_goal < 15.0:
            self.corner_mode = "TURNING"
        elif self.turn_type != "STRAIGHT" and dist_to_goal < 25.0:
            self.corner_mode = "APPROACH"
        else:
            self.corner_mode = "NORMAL"
    
    def publish_debug_goal(self):
        """Publish debug goal (UNCHANGED - WORKING)"""
        if self.goal_state is not None:
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'
            goal_msg.pose.position.x = self.goal_state[0]
            goal_msg.pose.position.y = self.goal_state[1]
            goal_msg.pose.position.z = 0.0
            
            if len(self.goal_state) > 2:
                quat = euler2quat(0, 0, self.goal_state[2])
                goal_msg.pose.orientation.w = quat[0]
                goal_msg.pose.orientation.x = quat[1]
                goal_msg.pose.orientation.y = quat[2]
                goal_msg.pose.orientation.z = quat[3]
            
            self.pub_debug_goal.publish(goal_msg)
            
        if hasattr(self, 'turn_type'):
            turn_info_msg = String()
            turn_info_msg.data = f"{self.turn_type}|{self.turn_complexity:.2f}|{self.corner_mode}|{self.goal_set_method}"
            self.pub_debug_turn_info.publish(turn_info_msg)
        
        if hasattr(self, 'traffic_light_status'):
            traffic_msg = String()
            traffic_msg.data = f"{self.traffic_light_status}|{self.traffic_light_distance:.1f}|{self.traffic_light_state}"
            self.pub_traffic_light_status.publish(traffic_msg)
    
    def corner_goal_monitor(self):
        """Corner-aware goal monitoring (UNCHANGED - WORKING)"""
        if self.goal_state is None or self.current_state is None:
            return
        
        goal_x, goal_y = self.goal_state[0], self.goal_state[1]
        current_x, current_y = self.current_state[0], self.current_state[1]
        current_v = self.current_state[3]
        
        self.distance_to_goal = np.sqrt((goal_x - current_x)**2 + (goal_y - current_y)**2)
        
        self.analyze_corner_type()
        self.update_steering_commitment_tracking()
        
        goal_reached = False
        if self.distance_to_goal < self.goal_tolerance:
            if current_v < 0.5:
                goal_reached = True
                reason = "distance + stopped"
            elif self.distance_to_goal < 0.8:
                goal_reached = True
                reason = "very close distance"
        
        if goal_reached:
            if self.goal_start_time is not None:
                elapsed_time = time.time() - self.goal_start_time
                self.goal_reached_count += 1
                self.goal_stop_count += 1
                
                if self.turn_type == "SHARP_TURN" or self.turn_type == "U_TURN":
                    self.sharp_turn_count += 1
                elif self.turn_type != "STRAIGHT":
                    self.smooth_turn_count += 1
                else:
                    self.straight_count += 1
                
                if self.commitment_success_count > 0:
                    self.oscillation_prevention_count += 1
                
                self.get_logger().info(
                    f"🏁 ENHANCED STEERING COMMITMENT + TRAFFIC LIGHT + FIXED OBSTACLE AVOIDANCE GOAL REACHED! Type: {self.turn_type}, Reason: {reason}"
                )
                self.get_logger().info(
                    f"   Distance: {self.distance_to_goal:.2f}m, Speed: {current_v:.1f}m/s, Time: {elapsed_time:.1f}s"
                )
                self.get_logger().info(
                    f"   Total goals: {self.goal_reached_count}, Method: {self.goal_set_method}"
                )
                
                self.goal_state = None
                self.goal_start_time = None
                self.user_goal_locked = False
                self.goal_set_method = "none"
                
                self.reset_steering_commitment_state()
                
                self.get_logger().info("🔓 Goal completed - ready for new goal input")
    
    def disable_autopilot_once(self):
        """Disable autopilot once at startup (UNCHANGED - WORKING)"""
        self.autopilot_pub.publish(Bool(data=False))
        self.get_logger().info("🛑 Autopilot disabled by ENHANCED STEERING COMMITMENT + TRAFFIC LIGHT + FIXED OBSTACLE AVOIDANCE MPC node.")
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] (UNCHANGED - WORKING)"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    # 🔧 ADD NEW METHOD HERE:
    def determine_target_lane(self, heading_deg):
        """🛣️ Determine target lane based on heading direction"""
        heading_normalized = heading_deg % 360
        
        # For Town01 (assumption: 2-lane road, ~3.5m per lane)
        if 315 <= heading_normalized or heading_normalized < 45:
            # Heading East → Right lane
            target_lane = +1.75
            lane_desc = "RIGHT (East)"
        elif 45 <= heading_normalized < 135:
            # Heading North → Right lane (keep right rule)
            target_lane = +1.75  
            lane_desc = "RIGHT (North)"
        elif 135 <= heading_normalized < 225:
            # Heading West → Left lane
            target_lane = -1.75
            lane_desc = "LEFT (West)"
        elif 225 <= heading_normalized < 315:
            # Heading South → Left lane (keep right rule) 
            target_lane = -1.75
            lane_desc = "LEFT (South)"
        else:
            target_lane = 0.0  # Fallback to center
            lane_desc = "CENTER (Fallback)"
        
        self.get_logger().info(f"🛣️ LANE ASSIGNMENT: {lane_desc}, target_y={target_lane:.1f}m")
        return target_lane
    
    # === 🚀🚧🚦 ENHANCED MAIN CONTROL CALLBACK WITH SURGICAL OBSTACLE FIX ===
    
    def steering_commitment_odom_callback(self, msg):
        """🚀🚧 ENHANCED: Main control callback with FIXED obstacle avoidance"""
        # Extract state from odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        v = np.sqrt(vx**2 + vy**2)
        
        self.current_state = [x, y, yaw, v]
        
        self.debug_initial_position_once()
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_control_time < self.control_period:
            return
        
        self.last_control_time = current_time
        self.total_count += 1
        
        # === 🚧 OBSTACLE PRIORITY CHECK WITH TIMEOUT (HIGHEST PRIORITY) ===
        if self.obstacle_avoidance_active and self.avoidance_type == "emergency_brake":
            # 🔧 SURGICAL: Add timeout to prevent stuck emergency brake
            if not hasattr(self, '_emergency_start_time'):
                self._emergency_start_time = current_time
                self.get_logger().warn(f"🚨 EMERGENCY BRAKE START: {self.target_obstacle['type']} at {self.target_obstacle['distance']:.1f}m")
            
            emergency_duration = current_time - self._emergency_start_time
            
            # 🔧 SURGICAL: 3-second timeout to prevent infinite emergency brake
            if emergency_duration > 3.0:
                self.get_logger().warn(f"⏰ Emergency brake TIMEOUT after {emergency_duration:.1f}s - clearing obstacle state")
                self.obstacle_avoidance_active = False
                self.avoidance_type = "none"
                self.target_obstacle = None
                self._emergency_start_time = None
                # 🔒 PRESERVE: Continue with existing logic below
            else:
                # 🔒 PRESERVE: Keep existing emergency brake implementation
                control = CarlaEgoVehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                control.hand_brake = False
                control.reverse = False
                control.manual_gear_shift = False
                control.gear = 1
                
                obstacle_info = "unknown obstacle"
                if self.target_obstacle:
                    obstacle_info = f"{self.target_obstacle['type']} at {self.target_obstacle['distance']:.1f}m"
                
                # Reduced logging frequency
                if int(emergency_duration * 10) % 20 == 0:
                    self.get_logger().warn(f"🚨 EMERGENCY BRAKE: {obstacle_info} ({emergency_duration:.1f}s)")
                
                self.pub_control.publish(control)
                return  # 🔒 PRESERVE: Early return
        else:
            # Reset emergency timer when not in emergency
            if hasattr(self, '_emergency_start_time'):
                self._emergency_start_time = None
        
        # === 🚧 OBSTACLE BRAKE CHECK (SECOND HIGHEST PRIORITY) ===
        if self.obstacle_avoidance_active and self.avoidance_type == "brake":
            # 🔒 PRESERVE: Force controlled brake for obstacles
            delta = 0.0
            a = -2.5
            
            obstacle_info = "unknown obstacle"
            if self.target_obstacle:
                obstacle_info = f"{self.target_obstacle['type']} at {self.target_obstacle['distance']:.1f}m"
            
            brake_log_counter = getattr(self, '_obstacle_brake_log_counter', 0) + 1
            self._obstacle_brake_log_counter = brake_log_counter
            
            if brake_log_counter % 20 == 1:
                self.get_logger().info(f"🚧 OBSTACLE CONTROLLED BRAKE: {obstacle_info}")
            
            control = CarlaEgoVehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = min(-a / 4.0, 1.0)
            control.hand_brake = False
            control.reverse = False
            control.manual_gear_shift = False
            control.gear = 1
            
            self.last_delta = 0.0
            self.last_a = a
            
            self.pub_control.publish(control)
            return  # Exit early for obstacle brake
        
        # === 🚦 TRAFFIC LIGHT PRIORITY CHECK (THIRD PRIORITY - UNCHANGED - WORKING) ===
        if self.traffic_light_state == "APPROACHING_RED":
            self.get_logger().info(f"🔍 OVERRIDE CHECK: distance={self.traffic_light_distance:.1f}m, state={self.traffic_light_state}")
            
            if self.traffic_light_distance < 30.0:
                delta = 0.0
                a = -5.0
                self.get_logger().info(f"🚨 TRAFFIC LIGHT OVERRIDE ACTIVATED! FORCE BRAKE at {self.traffic_light_distance:.1f}m")
                
                control = CarlaEgoVehicleControl()
                control.steer = 0.0
                control.throttle = 0.0
                control.brake = 1.0
                control.hand_brake = False
                control.reverse = False
                control.manual_gear_shift = False
                control.gear = 1
                
                self.last_delta = delta
                self.last_a = a
                
                self.pub_control.publish(control)
                return
                
        elif self.traffic_light_state == "STOPPED_AT_RED":
            self.get_logger().info(f"🔴 STOPPED AT RED - FORCE MAXIMUM BRAKE")
            
            delta = 0.0  
            a = -8.0
            
            control = CarlaEgoVehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.reverse = False
            control.manual_gear_shift = False
            control.gear = 1
            
            self.last_delta = delta
            self.last_a = a
            
            self.pub_control.publish(control)
            return

        # Check if we have a goal
        if self.goal_state is None:
            self.get_logger().warn("⚠️ No enhanced steering commitment + traffic light + fixed obstacle avoidance goal set! Please publish a goal to /goal_pose")
            delta, a = self.basic_forward_controller(v)
        else:
            self.analyze_corner_type()
            goal_x, goal_y = self.goal_state[0], self.goal_state[1]
            dist_to_goal = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
            
            # 🔒 PRESERVE: CRITICAL goal stop logic (unchanged - proven)
            if dist_to_goal < 1.0:
                delta, a = 0.0, -4.0
                self.goal_stop_count += 1
                self.get_logger().info(f"🏁 ENHANCED STEERING COMMITMENT + TRAFFIC LIGHT + FIXED OBSTACLE AVOIDANCE FORCE STOP: Distance {dist_to_goal:.2f}m < 1.0m")
            elif dist_to_goal < 2.0:
                delta, a = 0.0, -2.0
                self.get_logger().info(f"🚶 ENHANCED STEERING COMMITMENT + TRAFFIC LIGHT + FIXED OBSTACLE AVOIDANCE GENTLE STOP: Distance {dist_to_goal:.2f}m < 2.0m")
            else:
                # 🚀🚦🚧 Use ENHANCED MPC with all features (UNCHANGED - WORKING)
                start_time = time.time()
                try:
                    # Before MPC call, add:
                    if hasattr(self, 'target_lane_center'):
                        self.get_logger().info(f"🔍 MPC CALL DEBUG: target_lane={self.target_lane_center:.1f}m")
                    else:
                        self.get_logger().info(f"🔍 MPC CALL DEBUG: NO target_lane_center set!")

                    delta, a = solve_corner_mpc_with_debug(
                        current_state=self.current_state,
                        goal_state=self.goal_state,
                        obstacles=self.obstacles,  # 🔧 FIXED: Now using reliable obstacle data
                        traffic_light_status=self.traffic_light_status,
                        traffic_light_distance=self.traffic_light_distance,
                        traffic_light_state=self.traffic_light_state,
                        target_lane_center=self.target_lane_center,  # 🔧 NEW parameter
                        turn_type=self.turn_type,
                        turn_complexity=self.turn_complexity,
                        corner_mode=self.corner_mode,
                        N=self.N,
                        Ts=self.Ts,
                        L=self.L
                    )
                    
                    solve_time = time.time() - start_time
                    self.solve_times.append(solve_time)
                    self.success_count += 1
                    
                    if self.turn_type != "STRAIGHT":
                        self.corner_count += 1
                    else:
                        self.straight_count += 1
                    
                    if abs(delta) > 0.25 or abs(a) > 1.8:
                        self.aggressive_control_count += 1
                    else:
                        self.smooth_control_count += 1
                    
                    self.update_steering_commitment_tracking()
                    
                    # 🚧 FIXED: Apply obstacle-aware modifications with reliable data
                    if self.nearby_obstacles:
                        if a > 0:
                            a = a * 0.8
                            
                        closest_nearby = min(self.nearby_obstacles, key=lambda x: x['distance'])
                        
                        mod_log_counter = getattr(self, '_obstacle_mod_log_counter', 0) + 1
                        self._obstacle_mod_log_counter = mod_log_counter
                        
                        if mod_log_counter % 50 == 1:
                            self.get_logger().info(
                                f"🚧 OBSTACLE AWARE: Reduced acceleration due to {closest_nearby['type']} "
                                f"at {closest_nearby['distance']:.1f}m"
                            )
                    
                    # Enhanced logging with all features
                    log_counter = getattr(self, '_steering_log_counter', 0) + 1
                    self._steering_log_counter = log_counter
                    
                    if log_counter % 10 == 1:
                        commitment_status = "ACTIVE" if self.steering_commitment_active else "INACTIVE"
                        commitment_info = ""
                        
                        if self.steering_commitment_active:
                            duration = time.time() - self.commitment_start_time if self.commitment_start_time else 0.0
                            commitment_info = f" [{self.commitment_direction} {duration:.1f}s]"
                        
                        traffic_light_info = f"🚦{self.traffic_light_status.upper()}"
                        if self.traffic_light_distance != float('inf'):
                            traffic_light_info += f"@{self.traffic_light_distance:.1f}m"
                        
                        obstacle_info = f"🚧{len(self.critical_obstacles)}critical/{len(self.nearby_obstacles)}nearby"
                        if self.obstacle_avoidance_active:
                            obstacle_info += f"[{self.avoidance_type}]"
                        
                        self.get_logger().info(
                            f'🚀🚦🚧 ENHANCED-ALL-FEATURES-MPC: δ={delta:.3f}rad({np.degrees(delta):.1f}°), a={a:.2f}m/s², '
                            f'v={v:.1f}m/s, goal_dist={dist_to_goal:.1f}m'
                        )
                        self.get_logger().info(
                            f'🔄 Turn: {self.turn_type} (complexity: {self.turn_complexity:.2f}), '
                            f'mode: {self.corner_mode}, commitment: {commitment_status}{commitment_info}'
                        )
                        self.get_logger().info(
                            f'🚦🚧 Status: {traffic_light_info}, {obstacle_info}, state: {self.traffic_light_state}, method: {self.goal_set_method}'
                        )
                        self.get_logger().info(f'⚡ Solve_time={solve_time*1000:.1f}ms')
                    
                    if len(self.solve_times) > 50:
                        self.solve_times.pop(0)
                        
                except Exception as e:
                    solve_time = time.time() - start_time
                    self.mpc_failure_count += 1
                    self.get_logger().error(f"❌ ENHANCED ALL FEATURES MPC failed: {e}")
                    
                    delta, a = self.enhanced_all_features_fallback_controller(x, y, yaw, v)
                    self.fallback_usage_count += 1
        
        # 🔒 PRESERVE: Gentle control smoothing for corners (unchanged)
        if abs(delta - self.last_delta) > 0.10:
            delta = self.last_delta + np.sign(delta - self.last_delta) * 0.10
            
        if abs(a - self.last_a) > 1.0:
            a = self.last_a + np.sign(a - self.last_a) * 1.0
        
        # 🚦🚧 ENHANCED: Safety limits with traffic light + obstacle awareness (UNCHANGED - WORKING)
        if self.traffic_light_state == "APPROACHING_RED":
            delta = np.clip(delta, -0.35, 0.35)
            a = np.clip(a, -5.0, 1.0)
        elif self.traffic_light_state == "STOPPED_AT_RED":
            delta = np.clip(delta, -0.20, 0.20)
            a = np.clip(a, -4.0, 0.5)
        elif self.obstacle_avoidance_active:
            delta = np.clip(delta, -0.40, 0.40)
            a = np.clip(a, -5.0, 1.5)
        elif self.turn_type in ["SHARP_TURN", "U_TURN"]:
            delta = np.clip(delta, -0.45, 0.45)
            a = np.clip(a, -4.5, 2.0)
        else:
            delta = np.clip(delta, -0.35, 0.35)
            a = np.clip(a, -4.0, 2.5)
        
        # Store control history
        self.last_delta = delta
        self.last_a = a
        
        # 🔒 PRESERVE: Create and publish control command (UNCHANGED - WORKING)
        control = CarlaEgoVehicleControl()
        if self.turn_type in ["SHARP_TURN", "U_TURN"]:
            control.steer = float(np.clip(delta / 0.45, -1.0, 1.0))
        else:
            control.steer = float(np.clip(delta / 0.35, -1.0, 1.0))
        
        if a >= 0:
            control.throttle = float(min(a / 2.5, 1.0))
            control.brake = 0.0
        else:
            control.throttle = 0.0
            if self.traffic_light_state == "APPROACHING_RED":
                control.brake = float(min(-a / 5.0, 1.0))
            elif self.traffic_light_state == "STOPPED_AT_RED":
                control.brake = float(min(-a / 4.0, 1.0))
            elif self.obstacle_avoidance_active:
                control.brake = float(min(-a / 5.0, 1.0))
            elif self.turn_type in ["SHARP_TURN", "U_TURN"]:
                control.brake = float(min(-a / 4.5, 1.0))
            else:
                control.brake = float(min(-a / 4.0, 1.0))
        
        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False
        control.gear = 1
        
        self.pub_control.publish(control)
    
    def basic_forward_controller(self, v):
        """Basic forward motion when no goal (UNCHANGED - WORKING)"""
        target_speed = 2.0
        speed_error = target_speed - v
        a = np.clip(speed_error * 1.0, -2.0, 1.5)
        delta = 0.0
        return delta, a
    
    def enhanced_all_features_fallback_controller(self, x, y, yaw, v):
        """🚀🚦🚧 ENHANCED: All features aware fallback controller (UNCHANGED - WORKING)"""
        if self.goal_state is None:
            return self.basic_forward_controller(v)
        
        goal_x, goal_y = self.goal_state[0], self.goal_state[1]
        theta_goal = self.goal_state[2] if len(self.goal_state) > 2 else None
        
        dx = goal_x - x
        dy = goal_y - y
        dist_to_goal = np.sqrt(dx**2 + dy**2)
        
        if self.obstacle_avoidance_active and self.target_obstacle:
            if self.target_obstacle['distance'] <= self.emergency_distance:
                self.get_logger().info(f"🚨 ENHANCED FALLBACK OBSTACLE EMERGENCY: {self.target_obstacle['type']} at {self.target_obstacle['distance']:.1f}m")
                return 0.0, -6.0
        
        if self.traffic_light_state == "STOPPED_AT_RED":
            self.get_logger().info(f"🔴 ENHANCED FALLBACK STOPPED AT RED: Maintaining complete stop")
            return 0.0, -6.0
        
        if self.traffic_light_status == "red" and self.traffic_light_distance < 15.0:
            self.get_logger().info(f"🚦 ENHANCED FALLBACK RED LIGHT: Emergency stop at {self.traffic_light_distance:.1f}m")
            return 0.0, -5.0
        
        if dist_to_goal < 1.2:
            self.get_logger().info(f"🏁 ENHANCED ALL FEATURES FALLBACK GOAL REACHED: Distance {dist_to_goal:.2f}m - STOPPING")
            return 0.0, -3.5
        elif dist_to_goal < 2.5:
            self.get_logger().info(f"🚶 ENHANCED ALL FEATURES FALLBACK APPROACHING: Distance {dist_to_goal:.2f}m - GENTLE BRAKE")
            return 0.0, -1.8
        
        desired_heading = np.arctan2(dy, dx)
        heading_error = self.normalize_angle(desired_heading - yaw)
        heading_error_deg = np.degrees(heading_error)
        
        if abs(heading_error_deg) > 25.0 and self.turn_type in ["NORMAL_TURN", "SHARP_TURN", "U_TURN"]:
            if heading_error > 0:
                if self.turn_type in ["SHARP_TURN", "U_TURN"]:
                    delta_fb = 0.50
                else:
                    delta_fb = 0.40
            else:
                if self.turn_type in ["SHARP_TURN", "U_TURN"]:
                    delta_fb = -0.50
                else:
                    delta_fb = -0.40
        else:
            if self.turn_type in ["SHARP_TURN", "U_TURN"]:
                delta_fb = np.clip(heading_error * 0.8, -0.50, 0.50)
            else:
                delta_fb = np.clip(heading_error * 0.5, -0.35, 0.35)
        
        if self.traffic_light_state == "APPROACHING_RED":
            if self.traffic_light_distance < 10.0:
                target_speed = 0.5
            else:
                target_speed = min(2.0, self.traffic_light_distance * 0.15)
        elif self.obstacle_avoidance_active and self.target_obstacle:
            obs_distance = self.target_obstacle['distance']
            if obs_distance < 8.0:
                target_speed = max(0.5, obs_distance * 0.2)
            else:
                target_speed = 2.0
        elif self.turn_type in ["SHARP_TURN", "U_TURN"]:
            if dist_to_goal < 5.0:
                target_speed = 1.0
            else:
                target_speed = 2.0
        else:
            target_speed = 3.0
        
        complexity_reduction = 1.0 - (self.turn_complexity * 0.4)
        target_speed *= complexity_reduction
        
        speed_error = target_speed - v
        
        if self.traffic_light_state == "APPROACHING_RED":
            a_fb = np.clip(speed_error * 1.5, -4.5, 0.8)
        elif self.obstacle_avoidance_active:
            a_fb = np.clip(speed_error * 1.2, -4.0, 0.6)
        else:
            a_fb = np.clip(speed_error * 1.0, -3.0, 1.2)
        
        self.get_logger().info(
            f"🎯 ENHANCED ALL FEATURES Fallback: δ={np.degrees(delta_fb):.1f}°, a={a_fb:.2f}, "
            f"target_v={target_speed:.1f}, turn={self.turn_type}"
        )
        
        return delta_fb, a_fb
    
    # === 🚀🚦🚧 ENHANCED STATUS MONITORING (UNCHANGED - WORKING) ===
    
    def print_enhanced_status(self):
        """🚀🚦🚧 ENHANCED: Print status with all features (UNCHANGED - WORKING)"""
        if self.total_count > 0:
            success_rate = (self.success_count / self.total_count) * 100
            avg_solve_time = np.mean(self.solve_times) if self.solve_times else 0
            
            total_controls = self.smooth_control_count + self.aggressive_control_count
            if total_controls > 0:
                smooth_percentage = (self.smooth_control_count / total_controls) * 100
            else:
                smooth_percentage = 0
            
            if self.commitment_activation_count > 0:
                commitment_success_rate = (self.commitment_success_count / self.commitment_activation_count) * 100
                avg_commitment_time = self.total_commitment_time / self.commitment_success_count if self.commitment_success_count > 0 else 0
            else:
                commitment_success_rate = 0
                avg_commitment_time = 0
            
            goal_status = "NO_GOAL"
            if self.goal_state is not None:
                goal_x, goal_y = self.goal_state[0], self.goal_state[1]
                lock_status = "🔒LOCKED" if self.user_goal_locked else "🔓UNLOCKED"
                goal_status = f"Goal({goal_x:.1f},{goal_y:.1f}) dist={self.distance_to_goal:.1f}m {lock_status} [{self.goal_set_method}]"
            
            self.get_logger().info(
                f"📊 ENHANCED ALL FEATURES MPC Status: {success_rate:.1f}% success, "
                f"avg_solve: {avg_solve_time*1000:.1f}ms, "
                f"goals_reached: {self.goal_reached_count}, "
                f"smooth_control: {smooth_percentage:.1f}%"
            )
            
            self.get_logger().info(
                f"🚀 COMMITMENT Stats: activations={self.commitment_activation_count}, "
                f"success_rate={commitment_success_rate:.1f}%, "
                f"avg_duration={avg_commitment_time:.1f}s, "
                f"oscillations_prevented={self.oscillation_prevention_count}"
            )
            
            yellow_proceed_rate = 0
            yellow_stop_rate = 0
            if self.yellow_light_decisions > 0:
                yellow_proceed_rate = (self.yellow_proceed_count / self.yellow_light_decisions) * 100
                yellow_stop_rate = (self.yellow_stop_count / self.yellow_light_decisions) * 100
            
            self.get_logger().info(
                f"🚦 TRAFFIC LIGHT Stats: red_stops={self.red_light_stops}, "
                f"yellow_decisions={self.yellow_light_decisions} (proceed={yellow_proceed_rate:.1f}%, stop={yellow_stop_rate:.1f}%), "
                f"emergency_brakes={self.emergency_brakes}, "
                f"smooth_stops={self.smooth_traffic_stops}"
            )
            
            self.get_logger().info(
                f"🚧 FIXED OBSTACLE Stats: detected={self.obstacles_detected_count}, "
                f"emergency_brakes={self.obstacle_emergency_brakes}, "
                f"avoidance_maneuvers={self.obstacle_avoidance_maneuvers}, "
                f"data_source={self.obstacle_data_source}"
            )
            
            total_turns = self.sharp_turn_count + self.smooth_turn_count
            if total_turns > 0:
                sharp_ratio = (self.sharp_turn_count / total_turns) * 100
                self.get_logger().info(
                    f"🔄 Corner Stats: total_turns={total_turns}, sharp={self.sharp_turn_count} ({sharp_ratio:.1f}%), "
                    f"smooth={self.smooth_turn_count}, straight={self.straight_count}"
                )
            
            self.get_logger().info(f"🎯 Navigation: {goal_status}")
            
            if self.traffic_light_status != "unknown":
                data_source = "POSITION" if self.traffic_light_position else "ENHANCED_FALLBACK"
                self.get_logger().info(
                    f"🚦 Current Traffic Light: {self.traffic_light_status.upper()} "
                    f"at {self.traffic_light_distance:.1f}m [{data_source}], "
                    f"State: {self.traffic_light_state}, ID: {self.nearest_traffic_light_id}"
                )
            
            if self.obstacle_avoidance_active and self.target_obstacle:
                self.get_logger().info(
                    f"🚧 Current Obstacle: {self.target_obstacle['type']} "
                    f"at {self.target_obstacle['distance']:.1f}m, "
                    f"Avoidance: {self.avoidance_type}, "
                    f"Critical: {len(self.critical_obstacles)}, Nearby: {len(self.nearby_obstacles)}"
                )
            elif self.obstacles_detected_count > 0:
                self.get_logger().info(
                    f"🚧 Obstacles Detected: {self.obstacles_detected_count} total, "
                    f"{len(self.critical_obstacles)} critical, {len(self.nearby_obstacles)} nearby "
                    f"[{self.obstacle_data_source}]"
                )
            
            position_data_available = len(self.traffic_light_positions) > 0
            state_data_available = len(self.traffic_light_states) > 0
            
            self.get_logger().info(
                f"📡 DATA SOURCES: TL_positions={len(self.traffic_light_positions)}, "
                f"TL_states={len(self.traffic_light_states)}, "
                f"obstacles={self.obstacle_data_source}, "
                f"reliability={'HIGH' if position_data_available else 'ENHANCED_FALLBACK' if state_data_available else 'NONE'}"
            )
            
            if hasattr(self, 'turn_type'):
                commitment_status = ""
                if self.steering_commitment_active:
                    duration = time.time() - self.commitment_start_time if self.commitment_start_time else 0.0
                    commitment_status = f" [COMMITTED {self.commitment_direction} {duration:.1f}s]"
                
                obstacle_status = ""
                if self.obstacle_avoidance_active:
                    obstacle_status = f" [AVOIDING {self.avoidance_type}]"
                
                self.get_logger().info(
                    f"🔄 Current Turn: {self.turn_type} (complexity: {self.turn_complexity:.2f}), "
                    f"mode: {self.corner_mode}{commitment_status}{obstacle_status}"
                )
            
            if self.current_state:
                x, y, yaw, v = self.current_state
                self.get_logger().info(
                    f"🚗 Vehicle State: pos=({x:.1f},{y:.1f}), heading={np.degrees(yaw):.1f}°, speed={v:.1f}m/s"
                )


def main(args=None):
  rclpy.init(args=args)
  node = CornerCapableMPCNode()
  
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      node.get_logger().info("📊 Keyboard interrupt - Enhanced All Features CSV logs saved")
      pass
  finally:
      node.destroy_node()
      rclpy.shutdown()

if __name__ == '__main__':
  main()