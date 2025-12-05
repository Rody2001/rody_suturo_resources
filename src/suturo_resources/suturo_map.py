import numpy as np
from PIL.ImageOps import scale
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import Box, Scale, Sphere, Cylinder, FileMesh, Color
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
import threading
import rclpy
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import TransformationMatrix, Point3
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import Connection6DoF, FixedConnection, RevoluteConnection, \
    ActiveConnection
from semantic_digital_twin.world_description.geometry import Box, Scale, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.spatial_types.spatial_types import Vector3

from semantic_digital_twin.semantic_annotations.factories import (
    DrawerFactory,
    ContainerFactory,
    HandleFactory,
    Direction,
    SemanticPositionDescription,
    HorizontalSemanticDirection,
    VerticalSemanticDirection, DoorFactory, DresserFactory, FridgeFactory, DishwasherFactory, CounterTopFactory,
    RoomFactory,
)


white = Color(1, 1, 1)
red = Color(1, 0, 0)
black = Color(0, 0, 0)
gray = Color(0.45, 0.49, 0.53)
wall = Color(0.74, 0.74, 0.74)
wood = Color(1, 0.827, 0.6078)
silver = Color(0.59, 0.45, 0.59)

def load_environment():
    world = World()
    root = Body(name=PrefixedName("root"))

    with world.modify_world():
        world.add_body(root)

    build_environment_walls(world)
    build_environment_furniture(world)

    return world

def build_environment_walls(world: World):
    all_wall_bodies = []
    all_wall_connections = []
    root = world.root

    south_wall1 = Box(scale=Scale(0.05, 1.00, 3.00), color=wall)
    shape_geometry = ShapeCollection([south_wall1])
    south_wall1_body = Body(name=PrefixedName("south_wall1_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(south_wall1_body)

    root_C_south_wall1 = FixedConnection(parent=root, child=south_wall1_body,
                                    parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(y=-2.01, z=1.50))
    all_wall_connections.append(root_C_south_wall1)

    south_wall2 = Box(scale=Scale(0.29, 0.05, 3.00), color=wall)
    shape_geometry = ShapeCollection([south_wall2])
    south_wall2_body = Body(name=PrefixedName("south_wall2_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(south_wall2_body)

    root_C_south_wall2 = FixedConnection(parent=root, child=south_wall2_body,
                                    parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=-0.145, y=-1.45, z=1.50))
    all_wall_connections.append(root_C_south_wall2)

    south_wall3 = Box(scale=Scale(0.05, 1.085, 1.00), color=wall)
    shape_geometry = ShapeCollection([south_wall3])
    south_wall3_body = Body(name=PrefixedName("south_wall3_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(south_wall3_body)

    root_C_south_wall3 = FixedConnection(parent=root, child=south_wall3_body,
                                    parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=-0.29, y=-0.9925, z=0.5))
    all_wall_connections.append(root_C_south_wall3)

    south_wall4 = Box(scale=Scale(0.29, 0.05, 1.00), color=wall)
    shape_geometry = ShapeCollection([south_wall4])
    south_wall4_body = Body(name=PrefixedName("south_wall4_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(south_wall4_body)

    root_C_south_wall4 = FixedConnection(parent=root, child=south_wall4_body,
                                    parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=-0.145, y=-0.45, z=0.5))
    all_wall_connections.append(root_C_south_wall4)

    south_wall5 = Box(scale=Scale(0.29, 0.05, 1.00), color=wall)
    shape_geometry = ShapeCollection([south_wall5])
    south_wall5_body = Body(name=PrefixedName("south_wall5_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(south_wall5_body)

    root_C_south_wall5 = FixedConnection(parent=root, child=south_wall5_body,
                                    parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=-0.145, y=0.45, z=0.5))
    all_wall_connections.append(root_C_south_wall5)

    south_wall6 = Box(scale=Scale(0.05, 2.75, 1.00), color=wall)
    shape_geometry = ShapeCollection([south_wall6])
    south_wall6_body = Body(name=PrefixedName("south_wall6_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(south_wall6_body)

    root_C_south_wall6 = FixedConnection(parent=root, child=south_wall6_body,
                                    parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=-0.29025, y=1.80, z=0.5))
    all_wall_connections.append(root_C_south_wall6)

    south_wall7 = Box(scale=Scale(0.05, 2.27, 1.00), color=wall)
    shape_geometry = ShapeCollection([south_wall7])
    south_wall7_body = Body(name=PrefixedName("south_wall7_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(south_wall7_body)

    root_C_south_wall7 = FixedConnection(parent=root, child=south_wall7_body,
                                    parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=-0.29025, y=5.16, z=0.5))
    all_wall_connections.append(root_C_south_wall7)

    east_wall = Box(scale=Scale(4.924, 0.05, 3.00), color=wall)
    shape_geometry = ShapeCollection([east_wall])
    east_wall_body = Body(name=PrefixedName("east_wall_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(east_wall_body)

    root_C_east_wall = FixedConnection(parent=root, child=east_wall_body,
                                   parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=2.462, y=-2.535, z=1.50))
    all_wall_connections.append(root_C_east_wall)

    middle_wall = Box(scale=Scale(0.05, 2.67, 1.00), color=wall)
    shape_geometry = ShapeCollection([middle_wall])
    middle_wall_body = Body(name=PrefixedName("middle_wall_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(middle_wall_body)

    root_C_middle_wall = FixedConnection(parent=root, child=middle_wall_body,
                                   parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=2.20975, y=5.00, z=0.50))
    all_wall_connections.append(root_C_middle_wall)

    west_wall = Box(scale=Scale(4.449, 0.05, 3.00), color=wall)
    shape_geometry = ShapeCollection([west_wall])
    west_wall_body = Body(name=PrefixedName("west_wall_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(west_wall_body)

    root_C_west_wall = FixedConnection(parent=root, child=west_wall_body,
                                   parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=1.9345, y=6.32, z=1.50))
    all_wall_connections.append(root_C_west_wall)

    north_wall = Box(scale=Scale(0.05, 8.04, 3.00), color=wall)
    shape_geometry = ShapeCollection([north_wall])
    north_wall_body = Body(name=PrefixedName("north_wall_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(north_wall_body)

    root_C_north_wall = FixedConnection(parent=root, child=north_wall_body,
                                   parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=4.949, y=1.51, z=1.50))
    all_wall_connections.append(root_C_north_wall)

    north_west_wall = Cylinder(width=1.53, height=3.00, color=wall)
    shape_geometry = ShapeCollection([north_west_wall])
    north_west_wall_body = Body(name=PrefixedName("north_west_wall_body"), collision=shape_geometry, visual=shape_geometry)
    all_wall_bodies.append(north_west_wall_body)

    root_C_north_west_wall = FixedConnection(parent=root, child=north_west_wall_body,
                                    parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=4.924, y=6.295, z=1.50))
    all_wall_connections.append(root_C_north_west_wall)

    with world.modify_world():
        for body in all_wall_bodies:
            world.add_body(body)

        for conn in all_wall_connections:
            world.add_connection(conn)
        return world


def build_environment_furniture(world: World):
    all_elements_bodies = []
    all_elements_connections = []
    root = world.root

    # refrigerator = Box(scale=Scale(0.60, 0.658, 1.49), color=white)
    # shape_geometry = ShapeCollection([refrigerator])
    # refrigerator_body = Body(name=PrefixedName("refrigerator_body"), collision=shape_geometry, visual=shape_geometry)
    # all_elements_bodies.append(refrigerator_body)
    #
    # root_C_fridge = FixedConnection(parent=root, child=refrigerator_body,
    #                                 parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=0.537, y=-2.181, z=0.745))
    # all_elements_connections.append(root_C_fridge)


    refrigerator_container = ContainerFactory(name=PrefixedName("refrigerator_container"),
                                                    scale=Scale(x=0.60, y=0.60, z=1.49))
    refrigerator_drawer = DrawerFactory(name=PrefixedName("refrigerator_drawer"), container_factory=ContainerFactory(name=PrefixedName("container"), direction=Direction.Z, scale=Scale(0.58, 0.58, 0.49),),
                                        handle_factory=HandleFactory(name=PrefixedName("refrigerator_drawer_handle"), scale=Scale(0.07, 0.505, 0.02)),
                                        parent_T_handle = TransformationMatrix.from_xyz_rpy(x=0.29, y=0, z=0.195))

    refrigerator_door = DoorFactory(name=PrefixedName("refrigerator_door"), scale=Scale(x=0.02, y=0.6, z=1.00),
                                    handle_factory=HandleFactory(name=PrefixedName("refrigerator_door_handle"), scale=Scale(0.07, 0.505, 0.02)),
                                    parent_T_handle=TransformationMatrix.from_xyz_rpy(x=0, y=0.1, z=0, roll=np.pi/2))


    refrigerator_world = FridgeFactory(name=PrefixedName("refrigerator"),
                                                container_factory=refrigerator_container,
                                                door_factories=[refrigerator_door],
                                                door_transforms= [TransformationMatrix.from_xyz_rpy(x=0.30, y=0, z=0.25)],
                                                drawers_factories=[refrigerator_drawer],
                                                parent_T_drawers = [TransformationMatrix.from_xyz_rpy(x=0,z=-0.5)]).create()

    root_C_refrigerator = FixedConnection(parent=root, child=refrigerator_world.root,
                                          parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=0.537, y=-2.181, z=0.745, yaw=np.pi/2))



    with world.modify_world():
        world.merge_world(refrigerator_world, root_C_refrigerator)




    # counterTop = Box(scale=Scale(2.044, 0.58, 0.845), color=wood)
    # shape_geometry = ShapeCollection([counterTop])
    # counterTop_body = Body(name=PrefixedName("counterTop_body"), collision=shape_geometry, visual=shape_geometry)
    # all_elements_bodies.append(counterTop_body)
    #
    # root_C_counterTop = FixedConnection(parent=root, child=counterTop_body,
    #                                     parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=1.859,y=-2.181, z=0.4225))
    # all_elements_connections.append(root_C_counterTop)


    counterTop_main_container = ContainerFactory(name=PrefixedName("counterTop_container"),
                                            scale=Scale(x=0.58, y=2.044, z=0.845))
    counterTop_sink_drawer = DrawerFactory(name=PrefixedName("counterTop_sink_drawer"), container_factory=ContainerFactory(name=PrefixedName("counterTop_sink_drawer_container"), direction=Direction.Z, scale=Scale(0.58, 0.593, 0.575),),
                                            handle_factory=HandleFactory(name=PrefixedName("counterTop_sink_drawer_handle"), scale=Scale(0.07, 0.505, 0.02)),
                                            parent_T_handle = TransformationMatrix.from_xyz_rpy(x=0.2725, y=0, z=0.2475))
    counterTop_bottom_drawer = DrawerFactory(name=PrefixedName("counterTop_bottom_drawer"), container_factory=ContainerFactory(name=PrefixedName("counterTop_bottom_drawer_container"), direction=Direction.Z, scale=Scale(0.58, 0.794, 0.288),),
                                            handle_factory=HandleFactory(name=PrefixedName("counterTop_bottom_drawer_handle"), scale=Scale(0.07, 0.705, 0.02)),
                                            parent_T_handle = TransformationMatrix.from_xyz_rpy(x=0.2725, y=0, z=0.0925))
    counterTop_middle_drawer = DrawerFactory(name=PrefixedName("counterTop_middle_drawer"), container_factory=ContainerFactory(name=PrefixedName("counterTop_middle_drawer_container"), direction=Direction.Z, scale=Scale(0.58, 0.794, 0.287),),
                                            handle_factory=HandleFactory(name=PrefixedName("counterTop_middle_drawer_handle"), scale=Scale(0.07, 0.705, 0.02)),
                                            parent_T_handle = TransformationMatrix.from_xyz_rpy(x=0.2725, y=0, z=0.0925))
    counterTop_top_drawer = DrawerFactory(name=PrefixedName("counterTop_top_drawer"), container_factory=ContainerFactory(name=PrefixedName("counterTop_top_drawer_container"), direction=Direction.Z, scale=Scale(0.58, 0.794, 0.14),),
                                            handle_factory=HandleFactory(name=PrefixedName("counterTop_top_drawer_handle"), scale=Scale(0.07, 0.705, 0.02)),
                                            parent_T_handle = TransformationMatrix.from_xyz_rpy(x=0.2725, y=0, z=0.02))


    dishwasher_container = ContainerFactory(name=PrefixedName("dishwasher_container"),scale=Scale(0.5, 0.595, 0.72))
    dishwasher_door = DoorFactory(name=PrefixedName("dishwasher_door"), scale=Scale(x=0.08, y=0.595, z=0.72),
                                  handle_factory=HandleFactory(name=PrefixedName("dishwasher_door_handle"), scale=Scale(0.07, 0.505, 0.02)),
                                  parent_T_handle=TransformationMatrix.from_xyz_rpy(z=0.32))
    dishwasher_bottom_drawer = DrawerFactory(name=PrefixedName("dishwasher_bottom_drawer"), container_factory=ContainerFactory(name=PrefixedName("dishwasher_bottom_drawer_container"), direction=Direction.Z, scale=Scale(0.5, 0.595, 0.17),),
                                            handle_factory=HandleFactory(name=PrefixedName("dishwasher_bottom_drawer_handle"), scale=Scale(0.06, 0.37, 0.01)),
                                            parent_T_handle = TransformationMatrix.from_xyz_rpy(x=0.25, y=0, z=0.085, pitch=-np.pi/2))
    dishwasher_top_drawer = DrawerFactory(name=PrefixedName("dishwasher_top_drawer"), container_factory=ContainerFactory(name=PrefixedName("dishwasher_top_drawer_container"), direction=Direction.Z, scale=Scale(0.5, 0.595, 0.17)),
                                          handle_factory=HandleFactory(name=PrefixedName("dishwasher_top_drawer_handle"), scale=Scale(0.06, 0.37, 0.01)),
                                          parent_T_handle = TransformationMatrix.from_xyz_rpy(x=0.25, y=0, z=0.085, pitch=-np.pi/2))

    dishwasher_world = DishwasherFactory(name=PrefixedName("dishwasher_world"),
                                         container_factory=dishwasher_container,
                                         door_factories=[dishwasher_door],
                                         door_transforms= [TransformationMatrix.from_xyz_rpy(x=0.25, y=0, z=0)],
                                         drawers_factories=[dishwasher_bottom_drawer, dishwasher_top_drawer],
                                         parent_T_drawers = [TransformationMatrix.from_xyz_rpy(x=0,z=-0.25),
                                                             TransformationMatrix.from_xyz_rpy(x=0, z=0.15)])


    counterTop_world = CounterTopFactory(name=PrefixedName("counterTop"),
                                         container_factory=counterTop_main_container,
                                         doors_factories=[],
                                         drawers_factories=[counterTop_sink_drawer, counterTop_bottom_drawer, counterTop_middle_drawer, counterTop_top_drawer],
                                         parent_T_drawers=[TransformationMatrix.from_xyz_rpy(x=0, y=0.6995, z=-0.035),
                                                           TransformationMatrix.from_xyz_rpy(x=0, y=-0.6, z=-0.1785),
                                                           TransformationMatrix.from_xyz_rpy(x=0.35, y=-0.6, z=0.1095),
                                                           TransformationMatrix.from_xyz_rpy(x=0.25, y=-0.6, z=0.3235)],
                                         dishwashers_factories=[dishwasher_world],
                                         parent_T_dishwashers=[TransformationMatrix.from_xyz_rpy(x=0, y=0.0945, z=0.0375)]).create()
    ##########################################################################################################
    root_C_counterTop = FixedConnection(parent=root, child=counterTop_world.root,
                                        parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=1.859,y=-2.181, z=0.4225, yaw=np.pi/2))


    with world.modify_world():
        world.merge_world(counterTop_world, root_C_counterTop)

    def coloring():
        for container in world.get_body_by_name("refrigerator_container").visual.shapes:
            container.color = gray
        for handle in world.get_body_by_name("refrigerator_door_handle").visual.shapes:
            handle.color = silver
        for handle in world.get_body_by_name("refrigerator_drawer_handle").visual.shapes:
            handle.color = silver
        for handle in world.get_body_by_name("counterTop_sink_drawer_handle").visual.shapes:
            handle.color = silver
        for handle in world.get_body_by_name("counterTop_bottom_drawer_handle").visual.shapes:
            handle.color = silver
        for handle in world.get_body_by_name("counterTop_middle_drawer_handle").visual.shapes:
            handle.color = silver
        for handle in world.get_body_by_name("counterTop_top_drawer_handle").visual.shapes:
            handle.color = silver
        for container in world.get_body_by_name("counterTop_container").visual.shapes:
            container.color = gray
        #world.get_body_by_name("counterTop_container").visual.shapes[5].color = wood
        #world.get_body_by_name("counterTop_container").visual.shapes[8].color = wood
        #world.get_body_by_name("counterTop_container").visual.shapes[10].color = wood
    coloring()
    #world.get_body_by_name("refrigerator_door_handle").visual.shapes[2].color = silver

    # open/close refrigerator door
    that_connection = world.get_body_by_name("refrigerator_door").parent_connection.parent.parent_connection
    that_connection.position = np.pi / 4

    # open/close drawers from CounterTop
    bottom_drawer_connection = world.get_body_by_name("counterTop_bottom_drawer_container").parent_connection
    bottom_drawer_connection.position = 0

    # open/close dishwasher door
    that_connection = world.get_body_by_name("dishwasher_door").parent_connection.parent.parent_connection
    that_connection.position = np.pi / 4
    #print(world.connections)
    #print(that_connection.name, that_connection.child.name)
    #print(world.get_body_by_name("dishwasher_door").parent_connection.parent)

    ovenArea = Box(scale=Scale(1.20, 0.658, 1.49), color=white)
    shape_geometry = ShapeCollection([ovenArea])
    ovenArea_body = Body(name=PrefixedName("ovenArea_body"), collision=shape_geometry, visual=shape_geometry)
    all_elements_bodies.append(ovenArea_body)

    root_C_ovenArea = FixedConnection(parent=root, child=ovenArea_body,
                                      parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=3.481,y=-2.181, z=0.745))
    all_elements_connections.append(root_C_ovenArea)

    table = Box(scale=Scale(2.45, 0.796, 0.845), color=white)
    shape_geometry = ShapeCollection([table])
    table_body = Body(name=PrefixedName("table_body"), collision=shape_geometry, visual=shape_geometry)
    all_elements_bodies.append(table_body)

    root_C_table = FixedConnection(parent=root, child=table_body,
                                   parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=3.545, y=0.426, z=0.4225))
    all_elements_connections.append(root_C_table)

    sofa = Box(scale=Scale(1.68, 0.94, 0.68), color=wood)
    shape_geometry = ShapeCollection([sofa])
    sofa_body = Body(name=PrefixedName("sofa_body"), collision=shape_geometry, visual=shape_geometry)
    all_elements_bodies.append(sofa_body)

    root_C_sofa = FixedConnection(parent=root, child=sofa_body,
                                  parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=3.60, y=1.20, z=0.34))
    all_elements_connections.append(root_C_sofa)

    lowerTable = Box(scale=Scale(0.37, 0.91, 0.44), color=white)
    shape_geometry = ShapeCollection([lowerTable])
    lowerTable_body = Body(name=PrefixedName("lowerTable_body"), collision=shape_geometry, visual=shape_geometry)
    all_elements_bodies.append(lowerTable_body)

    root_C_lowerTable = FixedConnection(parent=root, child=lowerTable_body,
                                        parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=4.22, y=2.22, z=0.22))
    all_elements_connections.append(root_C_lowerTable)

    # cabinet = Box(scale=Scale(0.43, 0.80, 2.02), color=white)
    # shape_geometry = ShapeCollection([cabinet])
    # cabinet_body = Body(name=PrefixedName("cabinet_body"), collision=shape_geometry, visual=shape_geometry)
    # all_elements_bodies.append(cabinet_body)
    #
    # root_C_cabinet = FixedConnection(parent=root, child=cabinet_body,
    #                                  parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=4.65, y=4.72, z=1.01))
    # all_elements_connections.append(root_C_cabinet)

    container_world = ContainerFactory(name=PrefixedName("drawer_container"),
                                       scale=Scale(x=0.43, y=0.8, z=2.02)).create()

    root_C_cabinet = FixedConnection(parent=world.root, child=container_world.root,
                                     parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=4.65, y=4.72,
                                                                                                      z=1.01,
                                                                                                      yaw=np.pi))
    with world.modify_world():
        world.merge_world(container_world, root_C_cabinet)

    desk = Box(scale=Scale(0.60, 1.20, 0.75), color=white)
    shape_geometry = ShapeCollection([desk])
    desk_body = Body(name=PrefixedName("desk_body"), collision=shape_geometry, visual=shape_geometry)
    all_elements_bodies.append(desk_body)

    root_C_desk = FixedConnection(parent=root, child=desk_body,
                                  parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=0.05, y=1.48, z=0.375))
    all_elements_connections.append(root_C_desk)

    cookingTable = Box(scale=Scale(1.75, 0.64, 0.71),color=wood)
    shape_geometry = ShapeCollection([cookingTable])
    cookingTable_body = Body(name=PrefixedName("cookingTable_body"), collision=shape_geometry, visual=shape_geometry)
    all_elements_bodies.append(cookingTable_body)

    root_C_cookingTable = FixedConnection(parent=root,child=cookingTable_body,
                                  parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=1.325, y=5.675, z=0.355))
    all_elements_connections.append(root_C_cookingTable)


    diningTable = Box(scale=Scale(0.73, 1.18, 0.73),color=wood)
    shape_geometry = ShapeCollection([diningTable])
    diningTable_body = Body(name=PrefixedName("diningTable_body"), collision=shape_geometry, visual=shape_geometry)
    all_elements_bodies.append(diningTable_body)

    root_C_diningTable = FixedConnection(parent=root,child=diningTable_body,
                                         parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=2.59975, y=5.705, z=0.365))
    all_elements_connections.append(root_C_diningTable)

    # define floor polygon (z = 0 for all points)
    kitchen_floor = [
        Point3(-0.29, -2.51, 0.0),
        Point3(4.924, -2.51, 0.0),
        Point3(4.924, 0.477, 0.0),
        Point3(-0.29, 0.477, 0.0),
    ]

    living_room_floor = [
        Point3(-0.29, 0.477, 0.0),
        Point3(4.924, 0.477, 0.0),
        Point3(4.949, 4.8665, 0.0),
        Point3(-0.29, 4.8665, 0.0),
    ]

    # TODo : magerment bedroom and office floor polygons
    bed_room_floor = [
        Point3(-0.29, 4.8665, 0.0),
        Point3(4.949, 4.8665, 0.0),
        Point3(4.949, 6.32, 0.0),
        Point3(-0.29, 6.32, 0.0),
    ]

    office_floor = [
        Point3(2.20975, 5.00, 0.0),
        Point3(4.949, 5.00, 0.0),
        Point3(4.949, 6.32, 0.0),
        Point3(2.20975, 6.32, 0.0),
    ]

    # create factory and world
    kitchen_world = RoomFactory(name=PrefixedName("kitchen_room"), floor_polytope=kitchen_floor).create()

    root_C_room = FixedConnection(parent=root, child=kitchen_world.root,
                                        parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=1.859, y=-2.181, z=0.4225, yaw=np.pi / 2))

    with world.modify_world():
        world.merge_world(kitchen_world, root_C_room)


    with world.modify_world():
        for body in all_elements_bodies:
            world.add_body(body)

        for conn in all_elements_connections:
            world.add_connection(conn)
        return world

class Publisher:
    def __init__(self, name):
        self.context = rclpy.init()
        self.node = rclpy.create_node(name)
        self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()

    def publish(self, world):
        viz = VizMarkerPublisher(world=world, node=self.node)


def published(world: World):
    # container_world = ContainerFactory(name=PrefixedName("drawer_container"),
    #                                    scale=Scale(x=0.43, y=0.8, z=2.02)).create()
    #
    # root_C_cabinet = FixedConnection(parent=world.root, child=container_world.root,
    #                                  parent_T_connection_expression=TransformationMatrix.from_xyz_rpy(x=4.65, y=4.72,
    #                                                                                                   z=1.01,
    #                                                                                                   yaw=np.pi))
    # with world.modify_world():
    #     world.merge_world(container_world, root_C_cabinet)

    rclpy.init()
    node = rclpy.create_node("semantic_digital_twin")
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

publisher = Publisher("semantic_digital_twin")
publisher.publish(load_environment())
