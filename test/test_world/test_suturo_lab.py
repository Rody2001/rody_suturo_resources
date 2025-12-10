from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

from suturo_resources.queries import query_kitchen_area, query_living_room_area, query_bed_room_area, query_office_area
from suturo_resources.suturo_map import load_environment

def test_load_environment_returns_world():
    world = load_environment()
    assert isinstance(world, World)
    assert world.root.name == PrefixedName("root_slam")


def test_areas():   #TODO needs to be fixed
    """
    Checks that key room areas can be queried and have valid center and pose.
    """
    world = load_environment()

    # List of areas and their query functions
    area_queries = [
        ("kitchen", query_kitchen_area),
        ("living_room", query_living_room_area),
        ("bedroom", query_bed_room_area),
        ("office", query_office_area),
    ]

    for area_name, query_func in area_queries:
        center, pose = query_func(world)
        assert center is not None, f"{area_name} center should not be None"
        assert pose is not None, f"{area_name} pose should not be None"