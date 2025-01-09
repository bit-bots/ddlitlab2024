import time

import cv2
import numpy as np
import rclpy
import torch
import torch.nn.functional as F  # noqa
from bitbots_msgs.msg import JointCommand
from bitbots_tf_buffer import Buffer
from cv_bridge import CvBridge
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from ema_pytorch import EMA
from game_controller_hl_interfaces.msg import GameState
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState

from ddlitlab2024.dataset.pytorch import Normalizer
from ddlitlab2024.ml.model import End2EndDiffusionTransformer
from ddlitlab2024.ml.model.encoder.image import ImageEncoderType, SequenceEncoderType
from ddlitlab2024.ml.model.encoder.imu import IMUEncoder
from ddlitlab2024.utils.utils import JOINT_NAMES_ORDER

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inference(Node):
    def __init__(self, node_name, context):
        super().__init__(node_name, context=context)
        # Activate sim time
        self.get_logger().info("Activate sim time")
        self.set_parameters(
            [rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)],
        )

        # Params
        self.sample_rate = 50  # Hz
        hidden_dim = 256
        self.action_context_length = 100
        self.trajectory_prediction_length = 10
        train_denoising_timesteps = 1000
        self.inference_denosing_timesteps = 30
        self.image_context_length = 10
        self.imu_context_length = 100
        self.joint_state_context_length = 100
        self.num_joints = 20
        checkpoint = "/homes/17vahl/ddlitlab2024/ddlitlab2024/ml/training/trajectory_transformer_model_fixed_norm.pth"

        # Subscribe to all the input topics
        self.joint_state_sub = self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.img_sub = self.create_subscription(Image, "/camera/image_proc", self.img_callback, 10)
        self.gamestate_sub = self.create_subscription(GameState, "/gamestate", self.gamestate_callback, 10)

        # Publisher for the output topic
        self.joint_state_pub = self.create_publisher(JointCommand, "/DynamixelController/command", 10)

        # Image embedding buffer
        self.latest_image = None
        self.image_embeddings = []

        # IMU buffer
        self.imu_data = []

        # Joint state buffer
        self.latest_joint_state = None
        self.joint_state_data = []

        # Joint command buffer
        self.joint_command_data = []

        # Gamestate
        self.latest_game_state = None

        # Add default values to the buffers
        self.image_embeddings = [torch.zeros(3, 480, 480)] * self.image_context_length
        self.imu_data = [torch.tensor([0.0, 0.0, 0.0, 1.0])] * self.imu_context_length
        self.joint_state_data = [torch.zeros(len(JOINT_NAMES_ORDER))] * self.joint_state_context_length
        self.joint_command_data = [torch.zeros(self.num_joints)] * self.action_context_length

        # TF buffer to estimate imu similarly to the way we fixed the dataset
        self.tf_buffer = Buffer(self, Duration(seconds=10))
        self.cv_bridge = CvBridge()
        self.rate = self.create_rate(self.sample_rate)

        # Load model
        self.get_logger().info("Load model")
        self.model = End2EndDiffusionTransformer(
            num_joints=self.num_joints,
            hidden_dim=hidden_dim,
            use_action_history=True,
            num_action_history_encoder_layers=2,
            max_action_context_length=self.action_context_length,
            use_imu=True,
            imu_orientation_embedding_method=IMUEncoder.OrientationEmbeddingMethod.QUATERNION,
            num_imu_encoder_layers=2,
            max_imu_context_length=self.imu_context_length,
            use_joint_states=True,
            joint_state_encoder_layers=2,
            max_joint_state_context_length=self.joint_state_context_length,
            use_images=True,
            image_sequence_encoder_type=SequenceEncoderType.TRANSFORMER,
            image_encoder_type=ImageEncoderType.RESNET18,
            num_image_sequence_encoder_layers=1,
            max_image_context_length=self.image_context_length,
            num_decoder_layers=4,
            trajectory_prediction_length=self.trajectory_prediction_length,
        ).to(device)

        self.og_model = self.model

        self.normalizer = Normalizer(self.model.mean, self.model.std)
        self.model = EMA(self.model)
        self.model.load_state_dict(torch.load(checkpoint, weights_only=True))
        self.model.eval()
        print(self.normalizer.mean)

        # Create diffusion noise scheduler
        self.get_logger().info("Create diffusion noise scheduler")
        self.scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2", clip_sample=False)
        self.scheduler.config["num_train_timesteps"] = train_denoising_timesteps
        self.scheduler.set_timesteps(self.inference_denosing_timesteps)

        # Create control timer to run inference at a fixed rate
        interval = 1 / self.sample_rate  # * self.trajectory_prediction_length
        self.create_timer(interval, self.step)

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def img_callback(self, msg: Image):
        self.latest_image = msg

    def gamestate_callback(self, msg: GameState):
        self.latest_game_state = msg

    def step(self):
        self.get_logger().info("Step")

        # First we want to fill the buffers
        if self.latest_joint_state is not None:
            # Joint names are not in the correct order, so we need to reorder them
            joint_state = torch.zeros(len(JOINT_NAMES_ORDER))
            for i, joint_name in enumerate(JOINT_NAMES_ORDER):
                idx = self.latest_joint_state.name.index(joint_name)
                joint_state[i] = self.latest_joint_state.position[idx]
            self.get_logger().info("Storing joint state")
            self.joint_state_data.append(joint_state)

        self.get_logger().info("Calculating image embeddings")
        if self.latest_image is not None:
            # Here we don't just want to put the image in the buffer, but calculate the embedding first
            # But for now the model dos not support the direct use of embeddings so we
            # calculate them every timestep for the whole sequence.
            # This is not efficient and should be changed in the future TODO

            # Deserialize the image
            img = self.cv_bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="rgb8")

            # Resize the image
            img = cv2.resize(img, (480, 480))

            # Make chw from hwc
            img = np.moveaxis(img, -1, 0)

            # Convert the image to a tensor
            img = torch.tensor(img, dtype=torch.float32)

            self.image_embeddings.append(img)

        self.get_logger().info("Calculating IMU data")
        # Due to a bug in the recordings of the bit-bots we can not use the imu data directly,
        # but instead need to derive it from the tf tree
        imu_transform = self.tf_buffer.lookup_transform("base_footprint", "base_link", Time())

        self.get_logger().info("Storing IMU data")

        # Store imu data as np array in the form wxyz
        self.imu_data.append(
            torch.tensor(
                [
                    imu_transform.transform.rotation.x,
                    imu_transform.transform.rotation.y,
                    imu_transform.transform.rotation.z,
                    imu_transform.transform.rotation.w,
                ]
            )
        )

        print(self.imu_data[-1])
        print(len(self.imu_data))
        print(self.joint_state_data[-1])

        # Remove the oldest data from the buffers
        self.joint_state_data = self.joint_state_data[-self.joint_state_context_length :]
        self.image_embeddings = self.image_embeddings[-self.image_context_length :]
        self.imu_data = self.imu_data[-self.imu_context_length :]
        self.joint_command_data = self.joint_command_data[-self.action_context_length :]

        # Prepare the data for inference
        batch = {
            "joint_state": (torch.stack(list(self.joint_state_data), dim=0).unsqueeze(0).to(device) + 3 * np.pi)
            % (2 * np.pi),
            "image_data": torch.stack(list(self.image_embeddings), dim=0).unsqueeze(0).to(device),
            "rotation": torch.stack(list(self.imu_data), dim=0).unsqueeze(0).to(device),
            "joint_command_history": torch.stack(list(self.joint_command_data), dim=0).unsqueeze(0).to(device),
        }

        # Perform the denoising process
        trajectory = torch.randn(1, self.trajectory_prediction_length, self.num_joints).to(device)

        self.get_logger().info("Performing denoising process")

        ## Perform the embedding of the conditioning
        embedded_input = self.og_model.encode_input_data(batch)

        self.scheduler.set_timesteps(self.inference_denosing_timesteps)
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                # Predict the noise residual
                noise_pred = self.og_model.forward_with_context(
                    embedded_input, trajectory, torch.tensor([t], device=device)
                )

                # Update the trajectory based on the predicted noise and the current step of the denoising process
                trajectory = self.scheduler.step(noise_pred, t, trajectory).prev_sample

        # Undo the normalization
        trajectory = self.normalizer.denormalize(trajectory)

        self.get_logger().info("Publishing trajectory")

        # Store the trajectory in the joint command buffer (action history)
        # self.joint_command_data.append(trajectory[0, 0].cpu())

        # Publish the trajectory
        # self.joint_state_pub.publish(
        #    JointCommand(
        #        joint_names=JOINT_NAMES_ORDER,
        #        velocities=[-1.0] * len(JOINT_NAMES_ORDER),
        #        accelerations=[-1.0] * len(JOINT_NAMES_ORDER),
        #        max_currents=[-1.0] * len(JOINT_NAMES_ORDER),
        #        positions=trajectory[0, 0].cpu().numpy() - np.pi,
        #    )
        # )

        # Store the trajectory in the joint command buffer (action history)
        for i in range(self.trajectory_prediction_length):
            self.joint_command_data.append(trajectory[0, i].cpu())

        # Publish the trajectory one by one
        for i in range(self.trajectory_prediction_length):
            time.sleep(1 / self.sample_rate)
            self.joint_state_pub.publish(
                JointCommand(
                    joint_names=JOINT_NAMES_ORDER,
                    velocities=[-1.0] * len(JOINT_NAMES_ORDER),
                    accelerations=[-1.0] * len(JOINT_NAMES_ORDER),
                    max_currents=[-1.0] * len(JOINT_NAMES_ORDER),
                    positions=trajectory[0, i].cpu().numpy() - np.pi,
                )
            )


def main(args=None):
    rclpy.init(args=args)
    node = Inference("inference", None)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
