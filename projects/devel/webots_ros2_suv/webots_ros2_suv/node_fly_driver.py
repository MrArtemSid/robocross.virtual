import json
import math
import os
import socket

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float32

from .lib.orientation import euler_from_quaternion

# Узел fly_driver: машиной управляет мозг мухи (сервер flyd на хосте, ML/labs/project/flyd.py).
# Здесь нет решений - только перевод датчиков в лучи, как в 2D-мире, и команды в cmd_ackermann:
#   камера глубины 840x480 -> 64 дальности по лучам слева направо (поле 1 рад)
#   лидар 64x1024 -> 128 дальностей по кругу (без земли и своего кузова)
#   камера цвета -> доля запрещающего (красного или жёлтого) огня светофора в верхней половине кадра
# Сервер: FLY_HOST:FLY_PORT (по умолчанию host.docker.internal:5005).

CAM_FOV, CAM_RAYS, CAM_MIN, CAM_MAX = 1.0, 64, 0.5, 40.0
CAM_ROWS = 30                    # строк над горизонтом: ниже камера видит землю
LIDAR_RAYS, LIDAR_MIN, LIDAR_MAX = 128, 1.5, 40.0
LIDAR_H = 1.37                   # высота лидара над землёй, м
GROUND = 0.15                    # м: точки ниже - земля
BODY_X = (-6.0, 0.6)             # кузов в системе лидара: от задней кромки до капота
BODY_Y = 1.05
GPS_X = 2.4                      # GPS (верхний слот + 0.4) впереди начала координат машины
RED_MIN, RED_G, RED_B = 200, 180, 80   # запрещающий огонь (красный или жёлтый): R > 200, G < 180, B < 80
RED_PIX = 40                     # столько пикселей огня - сигнал насыщен
COLOR_TOPIC = '/vehicle/camera/image_color'   # config/simulator/devices_topics.yaml: front_image_topicname


# Доля красного огня в кадре 0..1: верхняя половина (светофор выше горизонта)
def red_signal(img, encoding):
    k = (0, 1, 2) if encoding.startswith('rgb') else (2, 1, 0)   # R, G, B по раскладке кадра
    top = img[:img.shape[0] // 2]
    r, g, b = top[:, :, k[0]], top[:, :, k[1]], top[:, :, k[2]]
    return min(float(((r > RED_MIN) & (g < RED_G) & (b < RED_B)).sum()) / RED_PIX, 1.0)


class FlyDriver(Node):
    def __init__(self):
        super().__init__('node_fly_driver')
        qos = qos_profile_sensor_data
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        host = os.environ.get('FLY_HOST', 'host.docker.internal')
        port = int(os.environ.get('FLY_PORT', '5005'))
        self.sock = socket.create_connection((host, port))
        self.io = self.sock.makefile('rw')
        self.get_logger().info(f'подключено к мозгу мухи {host}:{port}')
        self.cmd = self.create_publisher(AckermannDrive, 'cmd_ackermann', 1)
        self.create_subscription(Image, '/vehicle/range_finder/image', self.on_range, qos)
        self.create_subscription(PointCloud2, '/vehicle/lidar_base/point_cloud', self.on_lidar, qos)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos)
        self.create_subscription(Float32, '/vehicle/gps/speed', self.on_speed, qos)
        self.create_subscription(Image, COLOR_TOPIC, self.on_color, qos)
        self.lid, self.pose, self.speed, self.red = None, None, 0.0, 0.0
        w = 840
        cols = (np.arange(w) - w / 2 + 0.5) / (w / 2) * math.tan(CAM_FOV / 2)
        self.col_ang = -np.arctan(cols)   # > 0 - слева; столбец 0 - левый край кадра
        self.groups = np.array_split(np.arange(w), CAM_RAYS)

    def on_speed(self, m):
        self.speed = float(m.data)

    # Кадр цвета -> доля красного огня в верхней половине
    def on_color(self, m):
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        self.red = red_signal(img, m.encoding)

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.pose = (p.x - GPS_X * math.cos(yaw), p.y - GPS_X * math.sin(yaw), yaw)

    # Облако лидара -> 128 дальностей по кругу: без земли, без своего кузова, ближе LIDAR_MIN - нет отражения
    def on_lidar(self, m):
        pts = np.frombuffer(m.data, dtype=np.float32).reshape(-1, m.point_step // 4)[:, :3]
        pts = pts[np.isfinite(pts).all(1)]
        x, y, z = pts.T
        r = np.hypot(x, y)
        keep = (z + LIDAR_H > GROUND) & (r >= LIDAR_MIN) & (r < LIDAR_MAX)
        keep &= ~((x > BODY_X[0]) & (x < BODY_X[1]) & (np.abs(y) < BODY_Y))
        k = np.floor((np.arctan2(y[keep], x[keep]) + math.pi) / (2 * math.pi) * LIDAR_RAYS + 0.5).astype(int) % LIDAR_RAYS
        res = np.full(LIDAR_RAYS, LIDAR_MAX)
        np.minimum.at(res, k, r[keep])
        self.lid = res

    # Кадр глубины -> 64 луча; глубина вдоль оптической оси переводится в дальность по лучу
    def on_range(self, m):
        if self.lid is None or self.pose is None:
            return
        img = np.frombuffer(m.data, dtype=np.float32).reshape(m.height, m.width)
        band = img[m.height // 2 - CAM_ROWS:m.height // 2]
        depth = np.nan_to_num(band.min(0), nan=CAM_MAX, posinf=CAM_MAX) / np.cos(self.col_ang[:m.width])
        cam = np.clip([depth[g].min() for g in self.groups], CAM_MIN, CAM_MAX)
        t = self.get_clock().now().nanoseconds * 1e-9   # время симуляции (/clock, use_sim_time из launch)
        req = {'t': t, 'x': self.pose[0], 'y': self.pose[1], 'th': self.pose[2], 'v': self.speed,
               'cam': [round(float(v), 3) for v in cam], 'lid': [round(float(v), 3) for v in self.lid],
               'red': round(self.red, 3)}
        self.io.write(json.dumps(req) + '\n')
        self.io.flush()
        res = json.loads(self.io.readline())
        cmd = AckermannDrive()
        cmd.steering_angle = -float(res['steer'])   # у мухи > 0 - влево, у Webots Driver > 0 - вправо
        cmd.speed = float(res['speed']) * 3.6   # suv_driver -> setCruisingSpeed: км/ч
        self.cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = FlyDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():   # Webots закрыт - контекст уже недействителен, стоп-команду слать некуда
            node.cmd.publish(AckermannDrive())
            rclpy.shutdown()
        node.sock.close()


if __name__ == '__main__':
    main()
