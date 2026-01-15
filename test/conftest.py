from suturo_resources.suturo_map import load_environment

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale, Sphere
from semantic_digital_twin.world_description.shape_collection import ShapeCollection



def test_load_world():
    world = load_environment()
    all_elements_connections = world.connections
    root = world.root
    milk = Box(scale=Scale(0.10, 0.10, 0.20))
    shape_geometry = ShapeCollection([milk])
    milk_body = Body(name=PrefixedName("milk_body"), collision=shape_geometry, visual=shape_geometry)

    root_C_milk = FixedConnection(parent=root, child=milk_body,
                                  parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(x=3.545,
                                                                                                              y=0.626,
                                                                                                              z=0.9225))
    all_elements_connections.append(root_C_milk)

    apple = Sphere(radius=0.10)
    shape_geometry = ShapeCollection([apple])
    apple_body = Body(name=PrefixedName("apple_body"), collision=shape_geometry, visual=shape_geometry)

    root_C_apple = FixedConnection(parent=root, child=apple_body,
                                   parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(x=3.545,
                                                                                                               y=0.426,
                                                                                                               z=0.9225))
    all_elements_connections.append(root_C_apple)

    banana = Box(scale=Scale(0.20, 0.05, 0.05))
    shape_geometry = ShapeCollection([banana])
    banana_body = Body(name=PrefixedName("banana_body"), collision=shape_geometry, visual=shape_geometry)
    root_C_banana = FixedConnection(parent=root, child=banana_body,
                                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(x=3.245,
                                                                                                                y=0.426,
                                                                                                                z=0.8225))
    all_elements_connections.append(root_C_banana)



    return world