import threading

import rclpy
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Wall
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale


def build_apartment_walls(world: World):
    root = world.root

    slam_map_transformation = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0, y=0, yaw=0
    )

    with world.modify_world():
        south_wall1 = Wall.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("south_wall1"),
            world_root_T_self= slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0, y=-1.375, yaw=0
            ),
            scale=Scale(0.05, 1.85, 3.00),
        )

        east_wall = Wall.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("east_wall"),
            world_root_T_self= slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0, y=-1.15, yaw=0
            ),
            scale=Scale(4.40, 0.05, 3.00),
        )

        return world


class Publisher:
    def __init__(self, name):
        self.context = rclpy.init()
        self.node = rclpy.create_node(name)
        self.thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True
        )
        self.thread.start()

    def publish(self, world):
        viz = VizMarkerPublisher(_world=world, node=self.node)
        viz.with_tf_publisher()


publisher = Publisher("semantic_digital_twin")
publisher.publish(build_apartment_walls())