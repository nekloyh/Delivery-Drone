"""ORB-SLAM3 to ZMQ bridge for RL environment.

Forwards pose and map points from SLAM to training via ZMQ.
RPE = ||p_t - p_{t-1}|| measures localization quality.
"""

import argparse
import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud
import zmq

LOGGER = logging.getLogger(__name__)


class FeatureBridge:
    """Bridge between ROS2 (ORB-SLAM3) and ZMQ (RL environment).
    
    Subscribes to:
    - /orb_slam3/pose: Current estimated pose
    - /orb_slam3/map_points: Sparse map points
    
    Publishes via ZMQ:
    - JSON with {p: position, map_points: list, rpe: float}
    """

    def __init__(
        self, 
        pub_address: str = 'tcp://127.0.0.1:5557', 
        rate_hz: float = 20.0
    ) -> None:
        """Initialize feature bridge.
        
        Args:
            pub_address: ZMQ PUB socket address to bind
            rate_hz: Publishing rate in Hz
        """
        self.pub_address = pub_address
        self.rate = rate_hz
        self.pose: Optional[Tuple[float, float, float]] = None
        self.prev_pose: Optional[Tuple[float, float, float]] = None
        self.map_pts: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.rpe: float = 0.0

        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(self.pub_address)
        LOGGER.info("ZMQ publisher bound to %s", self.pub_address)

        rclpy.init()
        self.node = rclpy.create_node('feature_bridge')
        self.node.create_subscription(
            PoseStamped, '/orb_slam3/pose', self._pose_cb, 10
        )
        self.node.create_subscription(
            PointCloud, '/orb_slam3/map_points', self._map_cb, 10
        )
        LOGGER.info("Subscribed to ORB-SLAM3 topics")

        self._stop = threading.Event()
        self.pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.pub_thread.start()
        LOGGER.info("Publishing thread started at %.1f Hz", rate_hz)

    def _pose_cb(self, msg: PoseStamped) -> None:
        """Handle pose updates and compute RPE."""
        p = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

        if self.prev_pose is not None:
            dp = np.array(p) - np.array(self.prev_pose)
            self.rpe = float(np.linalg.norm(dp))
        
        self.prev_pose = p
        self.pose = p

    def _map_cb(self, msg: PointCloud) -> None:
        """Handle map point updates."""
        pts = [(pt.x, pt.y, pt.z) for pt in msg.points]
        self.map_pts = (
            np.array(pts, dtype=np.float32) 
            if pts else np.empty((0, 3), dtype=np.float32)
        )

    def _publish_loop(self) -> None:
        """Publish aggregated features at fixed rate."""
        period = 1.0 / self.rate
        msg_count = 0
        
        while not self._stop.is_set():
            start = time.time()
            
            if self.pose is not None:
                payload = {
                    'p': self.pose,
                    'map_points': self.map_pts.tolist() if self.map_pts.size > 0 else [],
                    'rpe': float(self.rpe),
                }
                
                try:
                    self.sock.send_json(payload)
                    msg_count += 1
                    
                    if msg_count % 100 == 0:
                        LOGGER.debug(
                            "Published %d messages, %d map points", 
                            msg_count, len(self.map_pts)
                        )
                except zmq.ZMQError as exc:
                    LOGGER.warning("ZMQ send error: %s", exc)
            
            dt = time.time() - start
            time.sleep(max(0.0, period - dt))

    def spin(self) -> None:
        """Spin ROS2 node (blocking)."""
        try:
            LOGGER.info("Feature bridge running...")
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            LOGGER.info("Keyboard interrupt received")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown of all resources."""
        LOGGER.info("Shutting down feature bridge...")
        self._stop.set()
        self.pub_thread.join(timeout=2.0)
        self.node.destroy_node()
        rclpy.shutdown()
        self.sock.close()
        LOGGER.info("Feature bridge shut down")


def main() -> None:
    """CLI entry point for feature bridge."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description='Bridge ORB-SLAM3 features to ZMQ for RL environment'
    )
    parser.add_argument(
        '--pub', 
        default='tcp://127.0.0.1:5557', 
        help='ZMQ PUB address to bind'
    )
    parser.add_argument(
        '--rate', 
        type=float, 
        default=20.0, 
        help='Publish rate in Hz'
    )
    args = parser.parse_args()

    bridge = FeatureBridge(pub_address=args.pub, rate_hz=args.rate)
    try:
        bridge.spin()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
