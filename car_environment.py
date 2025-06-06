import glob
import os
import sys
import random
import numpy as np
import cv2
import time

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class CarEnv:
    SHOW_CAM = False
    im_width = 640
    im_height = 480
    STEER_AMT = 1.0

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]
        self.front_camera = None
        self.collision_hist = []
        self.actor_list = []
        self.vehicle = None
        self.collision_sensor = None
        self.camera_sensor = None

        # Define discrete actions
        self.actions = [
            carla.VehicleControl(throttle=1.0, steer=-self.STEER_AMT),
            carla.VehicleControl(throttle=1.0, steer=0),
            carla.VehicleControl(throttle=1.0, steer=self.STEER_AMT)
        ]

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i2)
            cv2.waitKey(1)
        i3 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        i3 = cv2.resize(i3, (160, 120))
        i3 = np.expand_dims(i3, axis=-1)
        self.front_camera = i3

    def reset(self, spawn_index=None):
        self.collision_hist = []

        if self.vehicle:
            for actor in self.actor_list:
                actor.destroy()
            self.actor_list = []

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[spawn_index] if spawn_index is not None else random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)
        self.actor_list.append(self.vehicle)

        # Attach camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{self.im_width}')
        camera_bp.set_attribute('image_size_y', f'{self.im_height}')
        camera_bp.set_attribute('fov', '110')
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera_sensor = self.world.spawn_actor(camera_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: self.process_img(data))

        # Attach collision sensor
        col_sensor_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_sensor_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_hist.append(event))

        while self.front_camera is None:
            time.sleep(0.01)

        return self.front_camera

    def step(self, action):
        self.vehicle.apply_control(self.actions[action])
        v = self.vehicle.get_velocity()
        kmh = (3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            return self.front_camera, -10, True, {}

        if kmh < 50:
            return self.front_camera, -1, False, {}
        elif kmh >= 80:
            return self.front_camera, 2, False, {}
        else:
            return self.front_camera, 1, False, {}
