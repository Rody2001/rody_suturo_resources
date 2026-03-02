from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color

from conftest import test_load_world
from suturo_resources.queries import (
    query_surface_of_most_similar_obj,
    query_semantic_annotations_on_surfaces,
    query_get_next_object_euclidean_x_y, query_annotations_by_color,
)
from suturo_resources.suturo_map import load_environment


def test_load_environment_returns_world():
    """
    Tests that loading the environment returns a World object with the correct root name.
    """
    world = load_environment()
    assert isinstance(world, World)
    assert world.root.name == PrefixedName("root_slam")


def test_query_semantic_annotations_on_surfaces():
    """
    Tests that giving Table annotations gives a list of the correct annotation on top.
    """
    world = test_load_world()
    table1 = world.get_semantic_annotation_by_name("fruit_table")
    table2 = world.get_semantic_annotation_by_name("vegetable_table")
    table3 = world.get_semantic_annotation_by_name("empty_table")
    apple = world.get_semantic_annotation_by_name("apple")
    carrot = world.get_semantic_annotation_by_name("carrot")
    orange = world.get_semantic_annotation_by_name("orange")
    lettuce = world.get_semantic_annotation_by_name("lettuce")
    assert query_semantic_annotations_on_surfaces([table1, table2], world).tolist() == [
        apple,
        orange,
        carrot,
        lettuce,
    ]
    assert query_semantic_annotations_on_surfaces([table3], world).tolist() == []
    assert query_semantic_annotations_on_surfaces([], world).tolist() == []


def test_query_get_next_object_euclidean_x_y():
    """
    Tests the functionality of the `query_get_next_object_euclidean_x_y` function to verify that it accurately identifies
    the next objects based on their Euclidean proximity within a simulation world. The test involves setting up a virtual
    world, retrieving specific objects and annotations, and validating the results returned by the function against
    predetermined expectations.

    :raises AssertionError: If any of the function assertions fail during testing.
    """
    world = test_load_world()
    toya = world.get_body_by_name("base_link_body")
    table1 = world.get_semantic_annotation_by_name("fruit_table")
    table2 = world.get_semantic_annotation_by_name("vegetable_table")
    table3 = world.get_semantic_annotation_by_name("empty_table")
    apple = world.get_semantic_annotation_by_name("apple")
    carrot = world.get_semantic_annotation_by_name("carrot")
    orange = world.get_semantic_annotation_by_name("orange")
    lettuce = world.get_semantic_annotation_by_name("lettuce")

    assert query_get_next_object_euclidean_x_y(toya, table1).tolist() == [orange, apple]
    assert query_get_next_object_euclidean_x_y(toya, table2).tolist() == [
        carrot,
        lettuce,
    ]
    assert query_get_next_object_euclidean_x_y(toya, table3).tolist() == []


def test_query_surface_of_most_similar_obj():
    """
    Tests the `query_surface_of_most_similar_obj` function for determining the surface of the most suitable
    semantic annotation object from a set of candidates based on similarity and
    other constraints. The function is evaluated under multiple scenarios to verify
    its logic in choosing the correct table, handling empty tables, and cases with
    no valid candidates.
    """
    world = test_load_world()
    table1 = world.get_semantic_annotation_by_name("fruit_table")
    table2 = world.get_semantic_annotation_by_name("vegetable_table")
    table3 = world.get_semantic_annotation_by_name("empty_table")
    table4 = world.get_semantic_annotation_by_name("empty_table2")

    banana = world.get_semantic_annotation_by_name("banana")
    apple = world.get_semantic_annotation_by_name("apple")
    carrot = world.get_semantic_annotation_by_name("carrot")
    orange = world.get_semantic_annotation_by_name("orange")
    lettuce = world.get_semantic_annotation_by_name("lettuce")

    # choosing the correct table
    assert query_surface_of_most_similar_obj(banana, [table1, table2, table3]) == table1
    assert query_surface_of_most_similar_obj(carrot, [table1, table2, table3]) == table2
    # choosing the empty table
    assert query_surface_of_most_similar_obj(lettuce, [table1, table3]) == table3
    assert query_surface_of_most_similar_obj(table1, [table1, table2, table3]) == table3
    # trying with a new threshold
    assert query_surface_of_most_similar_obj(orange, [table2, table3], 2) == table2
    # returning None if there is no empty table or no tables
    assert query_surface_of_most_similar_obj(apple, [table2]) == None
    assert query_surface_of_most_similar_obj(orange, []) == None
    # trying with 2 empty tables
    assert query_surface_of_most_similar_obj(apple, [table2, table3, table4]) == table3
    assert query_surface_of_most_similar_obj(apple, [table2, table4, table3]) == table4

def test_query_body_by_color():
    """
    Tests the query_annotations_by_color function by verifying the retrieval of semantic
    annotations by their associated colors.

    The function validates that calling query_bodies_by_color with different color
    parameters returns the expected list of annotations corresponding to that color
    within the test world.
    """
    world1 = test_load_world()
    apple = world1.get_semantic_annotation_by_name("apple")
    orange = world1.get_semantic_annotation_by_name("orange")
    carrot = world1.get_semantic_annotation_by_name("carrot")

    assert query_annotations_by_color(Color.RED(), world1) == [apple]
    assert query_annotations_by_color(Color.ORANGE(), world1) == [orange, carrot]
    assert query_annotations_by_color(Color.BLUE(), world1) == []
