
import argparse
import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud
import zmq

class FeatureBridge:

    def __init__(self, pub_address: str = 'tcp://127.0.0.1:5557', rate_hz: float = 20.0) -> None:
        self.pub_address = pub_address
        self.rate = rate_hz
        self.pose: Optional[Tuple[float, float, float]] = None
        self.prev_pose: Optional[Tuple[float, float, float]] = None
        self.map_pts: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.rpe: float = 0.0

        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(self.pub_address)

        rclpy.init()
        self.node = rclpy.create_node('feature_bridge')
        self.node.create_subscription(PoseStamped, '/orb_slam3/pose', self._pose_cb, 10)
        self.node.create_subscription(PointCloud, '/orb_slam3/map_points', self._map_cb, 10)

        self._stop = threading.Event()
        self.pub_thread = threading.Thread(target=self._publish_loop)
        self.pub_thread.start()

    def _pose_cb(self, msg: PoseStamped) -> None:
        p = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

        if self.prev_pose is not None:
            dp = np.array(p) - np.array(self.prev_pose)
            self.rpe = float(np.linalg.norm(dp))
        self.prev_pose = p
        self.pose = p

    def _map_cb(self, msg: PointCloud) -> None:
        pts = [(pt.x, pt.y, pt.z) for pt in msg.points]
        self.map_pts = np.array(pts, dtype=np.float32) if pts else np.empty((0, 3), dtype=np.float32)

    def _publish_loop(self) -> None:
        period = 1.0 / self.rate
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
                except zmq.ZMQError:
                    pass
            dt = time.time() - start
            time.sleep(max(0.0, period - dt))

    def spin(self) -> None:
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop.set()
        self.pub_thread.join()
        self.node.destroy_node()
        rclpy.shutdown()
        self.sock.close()

def main() -> None:
    parser = argparse.ArgumentParser(description='Bridge ORB‑SLAM3 features to ZMQ')
    parser.add_argument('--pub', default='tcp://127.0.0.1:5557', help='ZMQ PUB address to bind')
    parser.add_argument('--rate', type=float, default=20.0, help='Publish rate in Hz')
    args = parser.parse_args()

    bridge = FeatureBridge(pub_address=args.pub, rate_hz=args.rate)
    try:
        bridge.spin()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
