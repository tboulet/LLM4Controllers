import pytest
from agent.agentic.items import Controller, Knowledge
from agent.agentic.codebase_manager import CodebaseManager  # Adjust the import as needed

@pytest.fixture
def codebase():
    return CodebaseManager()

def test_create_constant(codebase):
    codebase.edit_code("PI = 3.14")
    assert "PI" in codebase.items
    assert codebase.items["PI"].pure_code == "PI = 3.14"

def test_create_function_using_constant(codebase):
    codebase.edit_code("PI = 3.14")
    codebase.edit_code("""
def area(radius):
    return PI * radius ** 2
""")
    assert "area" in codebase.items
    assert "PI" in codebase.items["area"].dependencies

def test_modify_constant(codebase):
    codebase.edit_code("PI = 3.14")
    codebase.edit_code("PI = 3.14159")
    assert codebase.items["PI"].pure_code == "PI = 3.14159"

def test_add_function_with_dependency(codebase):
    codebase.edit_code("PI = 3.14")
    codebase.edit_code("""
def area(radius):
    return PI * radius ** 2
""")
    codebase.edit_code("""
def circle_stats(radius):
    return {'radius': radius, 'area': area(radius)}
""")
    assert "circle_stats" in codebase.items
    assert "area" in codebase.items["circle_stats"].dependencies

def test_rollback_on_invalid_edit(codebase):
    codebase.edit_code("PI = 3.14")
    codebase.edit_code("""
def area(radius):
    return PI * radius ** 2
""")

    original_code = codebase.items["area"].pure_code

    with pytest.raises(Exception):
        codebase.edit_code("""
def area(radius):
    return PI * radius *** 2  # invalid syntax
""")
    
    assert codebase.items["area"].pure_code == original_code

def test_detect_circular_dependency(codebase):
    codebase.edit_code("""
def dummy():
    return 42
""")
    codebase.edit_code("""
def circle_stats(radius):
    return dummy()
""")

    # Now attempt to create a circular dependency
    with pytest.raises(ValueError, match="Circular dependency detected"):
        codebase.edit_code("""
def dummy():
    return circle_stats(5)
""")

    # Ensure original dummy remains
    assert "dummy" in codebase.items
    assert "circle_stats" in codebase.items

def test_delete_item_safe():
    manager = CodebaseManager()
    manager.edit_code("a = 1")
    assert "a" in manager.items

    manager.edit_code("del a")
    assert "a" not in manager.items


def test_delete_item_with_dependents_raises():
    manager = CodebaseManager()
    manager.edit_code("a = 1")
    manager.edit_code("b = a + 1")

    with pytest.raises(ValueError, match="used by: b"):
        manager.edit_code("del a")