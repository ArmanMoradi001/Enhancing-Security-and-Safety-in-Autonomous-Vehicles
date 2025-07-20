import math
import numpy as np
import weakref
import pygame
from simulation.connection import carla
from simulation.settings import RGB_CAMERA, SSC_CAMERA

# ---------------------------------------------------------------------
# ------------------------------- CAMERA SENSOR -----------------------
# ---------------------------------------------------------------------

class CameraSensor:
    """Primary front camera sensor for capturing visual observations for the network."""

    def __init__(self, vehicle):
        self.sensor_name = SSC_CAMERA
        self.parent = vehicle
        self.front_camera = []
        self.last_valid_sensors = None  # Store last valid sensor data
        self.sensor = self._setup_sensor()
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: self._process_image(weak_self, image))

    def _setup_sensor(self):
        """Sets up the front-facing camera sensor."""
        world = self.parent.get_world()
        camera_bp = world.get_blueprint_library().find(self.sensor_name)
        camera_bp.set_attribute('image_size_x', '160')
        camera_bp.set_attribute('image_size_y', '80')
        camera_bp.set_attribute('fov', '125')

        transform = carla.Transform(carla.Location(x=2.4, z=1.5), carla.Rotation(pitch=-10))
        return world.spawn_actor(camera_bp, transform, attach_to=self.parent)

    @staticmethod
    def _process_image(weak_self, image):
        """Processes incoming camera images and appends to the front_camera buffer."""
        self = weak_self()
        if not self:
            return

        image.convert(carla.ColorConverter.CityScapesPalette)
        raw_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        reshaped_array = raw_array.reshape((image.height, image.width, 4))[:, :, :3]  # Extract RGB channels
        self.front_camera.append(reshaped_array)
        self.last_valid_sensors = reshaped_array  # Always update last valid data

# ---------------------------------------------------------------------
# --------------------------- ENVIRONMENT CAMERA ----------------------
# ---------------------------------------------------------------------

class CameraSensorEnv:
    """Environment camera for third-person visualization using pygame."""

    def __init__(self, vehicle):
        pygame.init()
        self.display = pygame.display.set_mode((720, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface = None
        self.running = True
        self.sensor = self._setup_sensor()
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: self._process_image(weak_self, image))

    def _setup_sensor(self):
        """Sets up the third-person environment camera."""
        world = self.parent.get_world()
        camera_bp = world.get_blueprint_library().find(self.sensor_name)
        camera_bp.set_attribute('image_size_x', '720')
        camera_bp.set_attribute('image_size_y', '720')

        transform = carla.Transform(carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=-12.0))
        return world.spawn_actor(camera_bp, transform, attach_to=self.parent)

    @staticmethod
    def _process_image(weak_self, image):
        """Processes third-person camera images and displays using pygame."""
        self = weak_self()
        if not self or not self.running:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return

        try:
            raw_array = np.frombuffer(image.raw_data, dtype=np.uint8)
            reshaped_array = raw_array.reshape((image.height, image.width, 4))[:, :, :3]
            reshaped_array = reshaped_array[:, :, ::-1]  # Convert BGR to RGB

            self.surface = pygame.surfarray.make_surface(reshaped_array.swapaxes(0, 1))
            if pygame.get_init():
                self.display.blit(self.surface, (0, 0))
                pygame.display.flip()
        except pygame.error:
            print("Warning: Pygame display has quit")
            self.running = False

    def destroy(self):
        """Ensures proper cleanup of Pygame resources."""
        if self.running and pygame.get_init():
            pygame.quit()
        self.running = False

# ---------------------------------------------------------------------
# ---------------------------- COLLISION SENSOR -----------------------
# ---------------------------------------------------------------------

class CollisionSensor:
    """Collision sensor to detect and track collision events."""

    def __init__(self, vehicle):
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = []
        self.sensor = self._setup_sensor()
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: self._on_collision(weak_self, event))

    def _setup_sensor(self):
        """Sets up the collision sensor for the vehicle."""
        world = self.parent.get_world()
        sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        transform = carla.Transform(carla.Location(x=1.3, z=0.5))
        return world.spawn_actor(sensor_bp, transform, attach_to=self.parent)

    @staticmethod
    def _on_collision(weak_self, event):
        """Handles collision events and records impact intensity."""
        self = weak_self()
        if not self:
            return

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_data.append(intensity)