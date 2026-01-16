import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor
from rosgraph_msgs.msg import Clock

from sensor_msgs.msg import Image, Imu, CameraInfo
from cv_bridge import CvBridge

import cv2
import csv
import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path



class TUMVIDatasetPlayer(Node):
    def __init__(self, base_path, dataset_name):
        super().__init__("tumvi_dataset_player")
        

        self.pub_clock = self.create_publisher(Clock, "/clock", 10)

        # PARAMETERS
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.declare_parameter('image_width', 512)
        self.declare_parameter('image_height', 512)

        # Obtener los valores de los parámetros
        self.width = self.get_parameter('image_width').value
        self.height = self.get_parameter('image_height').value

        if not self.base_path or self.base_path == "":
            self.get_logger().error("Debe especificar base_path")
            raise RuntimeError("Missing base_path")

        # Ruta completa del dataset elegido
        self.dataset_path = os.path.join(self.base_path, self.dataset_name, "mav0")

        if not os.path.exists(self.dataset_path):
            self.get_logger().error(f"Dataset path no existe: {self.dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        # Publicar a frecuencia X, pero usamos timestamps reales
        self.publish_rate = self.declare_parameter("publish_rate", 100.0).value
        self.speed_factor = self.declare_parameter("speed_factor", 1.0).value

        self.bridge = CvBridge()

        #Perfil QoS
        image_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        imu_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # PUBLISHERS (topics del dataset)
        self.pub_image = self.create_publisher(Image, "/cam0/image_raw", image_qos)
        self.pub_imu = self.create_publisher(Imu, "/imu0", imu_qos)
        self.pub_camera_info = self.create_publisher(CameraInfo, "/cam0/camera_info", image_qos)

        # LOAD DATASET FILES
        self.load_image_data()
        self.load_imu_data()

        # índices
        self.img_idx = 0
        self.imu_idx = 0

        # Tiempo de inicio del dataset
        self.dataset_start_time = (self.image_data[0][0]-0.2) if self.image_data else 0
        self.wall_start_time = time.time()

        #Condicion para que el dataset espere un poco
        self.startup_delay = 2.0 
        self.first_publish = True

        # timer principal
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.get_logger().info(
            f"Dataset player iniciado: {self.dataset_name} "
            f"({len(self.image_data)} imágenes, {len(self.imu_data)} IMU lecturas)"
        )

    # LOAD IMAGE DATA
    def load_image_data(self):
        cam_folder = os.path.join(self.dataset_path, "cam0")
        csv_path = os.path.join(cam_folder, "data.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        self.image_data = []

        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header

            for row in reader:
                timestamp = float(row[0]) * 1e-9  # nanoseconds → seconds
                filename = row[1]
                filepath = os.path.join(cam_folder, "data", filename)

                if os.path.exists(filepath):
                    self.image_data.append((timestamp, filepath))
                else:
                    self.get_logger().warn(f"Archivo no encontrado: {filepath}")

        if not self.image_data:
            raise RuntimeError("No se cargaron imágenes del dataset")
        
        self.get_logger().info(f"{len(self.image_data)} imágenes cargadas")

    # LOAD IMU DATA
    def load_imu_data(self):
        imu_folder = os.path.join(self.dataset_path, "imu0")
        csv_path = os.path.join(imu_folder, "data.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        self.imu_data = []

        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # header

            for row in reader:
                # timestamp, wx, wy, wz, ax, ay, az
                t = float(row[0]) * 1e-9
                wx, wy, wz = float(row[1]), float(row[2]), float(row[3])
                ax, ay, az = float(row[4]), float(row[5]), float(row[6])
                self.imu_data.append((t, wx, wy, wz, ax, ay, az))

        if not self.imu_data:
            raise RuntimeError("No se cargaron datos IMU del dataset")
        
        self.get_logger().info(f"{len(self.imu_data)} IMU lecturas cargadas")

    #tiempo actual del dataset
    def get_dataset_time(self):
        elapsed = (time.time() - self.wall_start_time) * self.speed_factor
        return self.dataset_start_time + elapsed

    #convierte el tipo de dato a ROS2
    def timestamp_to_ros(self, timestamp):
        seconds = int(timestamp)
        nanoseconds = int((timestamp - seconds) * 1e9)
        return rclpy.time.Time(seconds=seconds, nanoseconds=nanoseconds).to_msg()

        if not hasattr(self, '_logged_timestamp'):
            self._logged_timestamp = True
            self.get_logger().info(f"Primer timestamp dataset: {timestamp:.6f} sec")
            self.get_logger().info(f"   -> ROS time: {seconds}s + {nanoseconds}ns")

    #obtiene la informacion basica de la camara
    def create_camera_info(self, timestamp):
        info = CameraInfo()
        info.header.stamp = self.timestamp_to_ros(timestamp)
        info.header.frame_id = "cam0"
        info.width = self.width
        info.height = self.height
        return info

    #TIMER
    def timer_callback(self):
        
        current_dataset_time = self.get_dataset_time()

        print(f"DEBUG: Reloj={current_dataset_time:.3f} | Img_Idx={self.img_idx}/{len(self.image_data)}", end='\r')

        msg_clock = Clock()
        msg_clock.clock = self.timestamp_to_ros(current_dataset_time)
        self.pub_clock.publish(msg_clock)

        #retraso inicial
        if self.first_publish:
            elapsed_startup = time.time() - self.wall_start_time
            if elapsed_startup < self.startup_delay:
                return  # Esperar hasta completar el delay
            else:
                self.first_publish = False
                self.wall_start_time = time.time()  # Reiniciar el contador
                self.get_logger().info("Iniciando publicación de datos...")
                return

        # Publicar IMU
        imu_look_ahead = 0.05
        while (self.imu_idx < len(self.imu_data) and 
               self.imu_data[self.imu_idx][0] <= (current_dataset_time + imu_look_ahead)):
            t, wx, wy, wz, ax, ay, az = self.imu_data[self.imu_idx]

            msg = Imu()
            msg.header.stamp = self.timestamp_to_ros(t)
            msg.header.frame_id = "imu0"

            msg.angular_velocity.x = wx
            msg.angular_velocity.y = wy
            msg.angular_velocity.z = wz

            msg.linear_acceleration.x = ax
            msg.linear_acceleration.y = ay
            msg.linear_acceleration.z = az

            # Covarianzas
            msg.angular_velocity_covariance = [0.0] * 9
            msg.linear_acceleration_covariance = [0.0] * 9

            self.pub_imu.publish(msg)
            self.imu_idx += 1

        # Publicar IMAGEN
        if (self.img_idx < len(self.image_data) and 
            self.image_data[self.img_idx][0] <= current_dataset_time):
            
            timestamp, filepath = self.image_data[self.img_idx]

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.get_logger().error(f"Error leyendo imagen: {filepath}")
                self.img_idx += 1
                return

            # Crear mensaje de imagen
            msg = self.bridge.cv2_to_imgmsg(img, encoding="mono8")
            msg.header.stamp = self.timestamp_to_ros(timestamp)
            msg.header.frame_id = "cam0"

            self.pub_image.publish(msg)

            # Publicar CameraInfo básico
            info = self.create_camera_info(timestamp)
            self.pub_camera_info.publish(info)

            self.img_idx += 1

        # FIN DEL DATASET
        if self.img_idx >= len(self.image_data) and self.imu_idx >= len(self.imu_data):
            self.get_logger().info("✓ Dataset completamente publicado. Terminando...")
            rclpy.shutdown()


def main(args=None):
    
    # Parser de argumentos CLI
    parser = argparse.ArgumentParser(description='TUM-VI Dataset Player for ROS2')
    parser.add_argument('base_path', help='Base path to datasets folder')
    parser.add_argument('dataset', help='Dataset name (e.g., outdoors4)')
    
    # Filtrar argumentos de ROS2 (que empiezan con __)
    try: 
        parsed, unknown = parser.parse_known_args()
    except Exception as e:
        print(f"\nERROR al parsear argumentos: {e}")
        return

    # Inicializar ROS2
    rclpy.init(args=args)
    node = None 
    try:
        node = TUMVIDatasetPlayer(parsed.base_path, parsed.dataset)
        rclpy.spin(node)
    except Exception as e:
        print(f"Error al iniciar el nodo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
