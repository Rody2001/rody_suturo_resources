import threading

import numpy as np
import rclpy
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Wall, Table, Door, Sofa
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale, Color
from semantic_digital_twin.world_description.world_entity import Body


def build_apartment_map():
    world = World()
    root = Body(name=PrefixedName("map"))

    with world.modify_world():
        world.add_body(root)

    build_apartment_walls(world)
    build_apartment_furniture(world)

    return world

def build_apartment_walls(world: World):
    root = world.root

    slam_map_transformation = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.5, y=0, yaw=0
    )

    with world.modify_world():
        world.add_body(root)
        south_wall1 = Wall.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("south_wall1"),
            world_root_T_self= slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-0.025, y=-1.375, yaw=0
            ),
            scale=Scale(0.05, 1.85, 3.00),
        )
        south_wall2 = Wall.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("south_wall2"),
            world_root_T_self= slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-0.025, y=4.365, yaw=0
            ),
            scale=Scale(0.05, 7.83, 3.00),
        )

        east_wall = Wall.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("east_wall"),
            world_root_T_self= slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=2.485, y=-2.325, yaw=np.pi/2
            ),
            scale=Scale(0.05, 4.97, 3.00),
        )

        north_wall = Wall.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("north_wall"),
            world_root_T_self= slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=4.995, y=2.99
            ),
            scale=Scale(0.05, 10.58, 3.00),
        )

        west_wall = Wall.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("west_wall"),
            world_root_T_self= slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=2.485, y=8.305, yaw=np.pi/2
            ),
            scale=Scale(0.05, 4.97, 3.00),
        )

    return world


def build_apartment_furniture(world: World):
    root = world.root

    slam_map_transformation = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.5, y=0, yaw=0
    )

    with world.modify_world():
        food_table = Table.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("food_table"),
            world_root_T_self=slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=3.905, y=6.185, z=0.375, yaw=np.pi/2
            ),
            scale=Scale(0.90, 1.40, 0.75),
        )
        for color in food_table.bodies[0].visual.shapes:
            color.color = Color.BEIGE()

        desk = Table.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("desk"),
            world_root_T_self=slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.40, y=4.55, z=0.43
            ),
            scale=Scale(0.80, 1.40, 0.86),
        )

        high_table = Table.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("high_table"),
            world_root_T_self=slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=4.525, y=2.955, z=0.527
            ),
            scale=Scale(0.70, 0.70, 1.054),
        )


        door = Door.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("door"),
            world_root_T_self=slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.48, y=0.4725, z=1.10, yaw=np.pi/2
            ),
            scale=Scale(0.045, 0.96, 2.20),
        )


        big_sofa = Sofa.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("big_sofa"),
            world_root_T_self=slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=4.53, y=-0.185, z=0.3675
            ),
            scale=Scale(0.88, 2.10, 0.735),
        )
        for color in big_sofa.bodies[0].visual.shapes:
            color.color = Color.RED()


        small_sofa = Sofa.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("small_sofa"),
            world_root_T_self=slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=3.54, y=1.555, z=0.3675
            ),
            scale=Scale(0.81, 0.90, 0.735),
        )
        for color in small_sofa.bodies[0].visual.shapes:
            color.color = Color.RED()

        sofa_table = Table.create_with_new_body_in_world(
            world=world,
            name=PrefixedName("sofa_table"),
            world_root_T_self=slam_map_transformation @ HomogeneousTransformationMatrix.from_xyz_rpy(
                x=3.2, y=0.08, z=0.201
            ),
            scale=Scale(0.697, 1.197, 0.402),
        )
        for color in sofa_table.bodies[0].visual.shapes:
            color.color = Color.BEIGE()

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


# publisher = Publisher("semantic_digital_twin")
# publisher.publish(build_apartment_map())