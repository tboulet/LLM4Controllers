import ast
import re
from typing import Dict, List, Any, Set
from agent.agentic.items import Controller, Knowledge
from enum import Enum

from core.types import CodeExecutionError, CodeExtractionError


class ItemType(Enum):
    BASIC_OBJECT = "basic_object"
    CONTROLLER = "controller"
    KNOWLEDGE = "knowledge"


class Item:
    def __init__(
        self,
        name: str,
        pure_code: str,
        item_type: ItemType,
        imports: List[str] = [],
        dependencies: List[str] = [],
        docstring: str = "",
        view_code: bool=True,
    ):
        self.name = name
        self.pure_code = pure_code
        self.item_type = item_type
        self.imports = imports
        self.dependencies = dependencies
        self.docstring = docstring
        self.view_code = view_code
        
    def __repr__(self):
        parts = [
            f"Item(name={self.name!r}, type={self.item_type.value})",
            f"  Imports     : {self.imports or '[]'}",
            f"  Dependencies: {self.dependencies or '[]'}",
            f"  Docstring : {self.docstring!r}" if self.docstring else "",
            "  Code:\n"
            + "\n".join(f"    {line}" for line in self.pure_code.splitlines()),
        ]
        return "\n".join(part for part in parts if part)


class CodebaseManager:
    def __init__(self, namespace: Dict[str, Any] = None):
        self.items: Dict[str, Item] = {}
        self.namespace = namespace or {}

    def edit_code(self, code: str, allow_several_top_defs: bool = False):
        # Step 1: Parse and validate
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise CodeExecutionError(f"Invalid Python code: {e}")

        # Step 2: Detect top-level definitions and deletions
        top_defs = []
        deletions = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
                top_defs.append(node)
            elif isinstance(node, ast.Delete):
                deletions.append(node)

        # Step 3: Delete items if any deletions are present
        if deletions and top_defs:
            raise CodeExtractionError(f"Cannot mix deletions with definitions in a single edit. {len(deletions)} deletions and {len(top_defs)} definitions found.")

        if deletions:
            for del_node in deletions:
                for target in del_node.targets:
                    if not isinstance(target, ast.Name):
                        raise CodeExtractionError("Only simple variable names can be deleted.")
                    self._delete_item(target.id)
            return  # Done

        if len(top_defs) == 0:
            raise CodeExtractionError(
                "Code must contain at least one top-level definition, found none."
            )

        # Pre-step before step 4: if there are several top-level definitions, deal with them one by one
        if len(top_defs) > 1:
            if allow_several_top_defs:
                import_lines = [
                    line
                    for line in code.splitlines()
                    if re.match(r"^\s*(import|from)\s", line)
                ]
                for node in top_defs:
                    single_def_code = ast.get_source_segment(code, node)
                    if single_def_code is None:
                        raise CodeExtractionError(
                            "Could not extract source code for a top-level item."
                        )
                    code_block = "\n".join(import_lines + ["", single_def_code])
                    self.edit_code(code_block, allow_several_top_defs=False)
                return
            else:
                raise CodeExtractionError(
                    f"Code must contain exactly one top-level definition, but found {len(top_defs)}."
                )

        # Step 4: Extract name, imports, docstring, and pure code
        top_def = top_defs[0]
        if isinstance(top_def, ast.FunctionDef):
            name = top_def.name
        elif isinstance(top_def, ast.ClassDef):
            name = top_def.name
        elif isinstance(top_def, ast.Assign):
            if len(top_def.targets) != 1 or not isinstance(
                top_def.targets[0], ast.Name
            ):
                raise CodeExtractionError("Assignment must be to a single variable name.")
            name = top_def.targets[0].id

        import_lines = []
        definition_lines = []
        docstring = ""

        lines = code.splitlines()
        for line in lines:
            if re.match(r"^\s*(import|from)\s", line):
                import_lines.append(line)
            else:
                definition_lines.append(line)

        docstring_match = re.match(
            r'^\s*[\'"]{3}(.*?)[\'"]{3}', "\n".join(definition_lines), re.DOTALL
        )
        if docstring_match:
            docstring = docstring_match.group(1).strip()
            code_wo_docstring = re.sub(
                r'^\s*[\'"]{3}(.*?)[\'"]{3}',
                "",
                "\n".join(definition_lines),
                flags=re.DOTALL,
            )
        else:
            code_wo_docstring = "\n".join(definition_lines)

        pure_code = code_wo_docstring.strip()
                
        # Step 5: Infer dependencies from AST
        used_names = set()
        try:
            node = ast.parse(pure_code)
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    used_names.add(n.id)
        except Exception:
            # ignore, fallback to no dependencies
            print(
                f"WARNING: Could not parse code for dependencies: {pure_code}. Still proceeding without dependencies."
            )

        dependencies = [dep for dep in used_names if dep in self.items and dep != name]

        # Step 6: Create Item (temporary or final)
        new_item = Item(
            name=name,
            pure_code=pure_code,
            item_type="pending",  # will determine after exec
            imports=import_lines,
            dependencies=dependencies,
            docstring=docstring,
        )

        is_new = name not in self.items
        old_item = self.items.get(name)

        # Step 7: Temporarily insert item, check circular deps
        self.items[name] = new_item
        try:
            if self._has_circular_dependency():
                raise CodeExecutionError(
                    f"Circular dependency detected when adding item '{name}'."
                )

            # Step 8: Get topologically sorted dependencies
            sorted_deps = self._collect_dependencies_topo_sorted(name)
            all_imports = set()
            code_blocks = []

            for dep_name in sorted_deps:
                item = self.items[dep_name]
                all_imports.update(item.imports)
                code_blocks.append(item.pure_code)

            all_imports.update(import_lines)
            code_blocks.append(pure_code)

            code_to_exec = (
                "\n".join(sorted(all_imports)) + "\n\n" + "\n\n".join(code_blocks)
            )

            # Step 9: Execute code
            temp_namespace = self.namespace.copy()
            try:
                exec(code_to_exec, temp_namespace)
            except Exception as e:
                raise CodeExecutionError(
                    f"Error executing code for item '{name}': {e}"
                )
                
            obj = temp_namespace[name]
            if isinstance(obj, type) and issubclass(obj, Controller):
                item_type = ItemType.CONTROLLER
            elif isinstance(obj, Knowledge):
                item_type = ItemType.KNOWLEDGE
            else:
                item_type = ItemType.BASIC_OBJECT

            new_item.item_type = item_type

            # Step 10: Save to namespace and finalize
            self.namespace.update(temp_namespace)
            self.items[name] = new_item

        except Exception as e:
            # Rollback
            if is_new:
                del self.items[name]
            else:
                self.items[name] = old_item
            raise e

    def execute_code(self, code: str, variables: Dict[str, Any] = {}):

        # Step 1: Find all variable names used in code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise CodeExtractionError(f"Invalid Python code: {e}")

        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)

        # Step 2: Collect codebase items that are required
        required_items: Set[str] = set()
        for name in used_names:
            if name in self.items:
                required_items.add(name)

        # Recursively collect dependencies
        all_deps = set()
        for name in required_items:
            all_deps.update(self._collect_dependencies_topo_sorted(name))
        all_deps.update(required_items)

        # Step 3: Build the execution code
        all_imports = set()
        code_blocks = []

        for dep in all_deps:
            item = self.items[dep]
            all_imports.update(item.imports)
            code_blocks.append(item.pure_code)

        # Add the user-provided code last
        code_to_exec = (
            "\n".join(sorted(all_imports))
            + "\n\n"
            + "\n\n".join(code_blocks)
            + "\n\n"
            + code
        )

        # Step 4: Execute
        exec_namespace = {}
        exec_namespace.update(variables)
        try:
            exec(code_to_exec, exec_namespace)
        except Exception as e:
            raise CodeExecutionError(f"Error executing code: {e}")
        
    def _has_circular_dependency(self) -> bool:
        visited = set()
        rec_stack = set()

        def visit(node: str) -> bool:
            if node in rec_stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)
            for dep in self.items.get(node, Item("", "", "", [])).dependencies:
                if visit(dep):
                    return True
            rec_stack.remove(node)
            return False

        for item_name in self.items:
            if visit(item_name):
                return True
        return False

    def _collect_dependencies_topo_sorted(self, name: str) -> List[str]:
        visited = set()
        result = []

        def dfs(n: str):
            if n in visited:
                return
            visited.add(n)
            for dep in self.items[n].dependencies:
                dfs(dep)
            result.append(n)

        dfs(name)
        return result[:-1]  # exclude self (will be appended later)

    def _collect_dependencies_topo_sorted_for_all(self):
        visited = set()
        result = []

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in self.items[name].dependencies:
                if dep in self.items:
                    visit(dep)
            result.append(name)

        for name in self.items:
            visit(name)

        return result


    def _delete_item(self, item_name: str):
        if item_name not in self.items:
            raise CodeExecutionError(f"Cannot delete '{item_name}': item does not exist.")

        # Check if any item depends on it
        dependents = [
            other.name
            for other in self.items.values()
            if item_name in other.dependencies
        ]
        if dependents:
            raise CodeExecutionError(
                f"Cannot delete '{item_name}' because it is used by: {', '.join(dependents)}"
            )

        del self.items[item_name]

    def __repr__(self):
        result = []
        for item_type in ItemType:
            if item_type != ItemType.KNOWLEDGE:
                continue  # only print knowledge items for now
            result.append(f"** {item_type.value.capitalize()}s Items **")
            items_of_type = [
                item for item in self.items.values() if item.item_type == item_type
            ]
            if len(items_of_type) == 0:
                result.append("  No items of this type.")
            else:
                for item in items_of_type:
                    result.append(f"{item}")
        return "\n\n".join(result) if result else "No items in codebase."
    
    def __repr__(self):
        result = []
        sorted_names = self._collect_dependencies_topo_sorted_for_all()
        sorted_items = [self.items[name] for name in sorted_names if name]

        for item_type in ItemType:
            result.append(f"** {item_type.value.capitalize()}s Items **")
            items_of_type = [item for item in sorted_items if item.item_type == item_type]

            if not items_of_type:
                result.append("  No items of this type.")
            else:
                for item in items_of_type:
                    result.append(item.pure_code)

        return "\n\n".join(result) if result else "No items in codebase."
