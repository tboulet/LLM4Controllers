import ast
import re
from typing import Dict, List, Any, Set
from agent.agentic.items import Controller, Knowledge

ITEM_TYPE_BASIC_OBJECT = "basic_object"
ITEM_TYPE_CONTROLLER = "controller"
ITEM_TYPE_KNOWLEDGE = "knowledge"


class Item:
    def __init__(
        self,
        name: str,
        pure_code: str,
        item_type: str,
        imports: List[str] = [],
        dependencies: List[str] = [],
        description: str = "",
    ):
        self.name = name
        self.pure_code = pure_code
        self.item_type = item_type
        self.imports = imports
        self.dependencies = dependencies
        self.description = description

    def __repr__(self):
        parts = [
            f"Item(name={self.name!r}, type={self.item_type!r})",
            f"  Imports     : {self.imports or '[]'}",
            f"  Dependencies: {self.dependencies or '[]'}",
            f"  Description : {self.description!r}" if self.description else "",
            "  Code:\n" + "\n".join(f"    {line}" for line in self.pure_code.splitlines())
        ]
        return "\n".join(part for part in parts if part)


class CodebaseManager:
    def __init__(self, namespace: Dict[str, Any] = None):
        self.items = {}  # name -> Item
        self.namespace = namespace or {}

    def edit_code(self, code: str, allow_several_top_defs: bool = False):
        # Step 1: Parse and validate
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {e}")

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
            raise ValueError("Cannot mix deletions with definitions in a single edit.")

        if deletions:
            for del_node in deletions:
                for target in del_node.targets:
                    if not isinstance(target, ast.Name):
                        raise ValueError("Only simple variable names can be deleted.")
                    self._delete_item(target.id)
            return  # Done

        if len(top_defs) == 0:
            raise ValueError("Code must contain at least one top-level definition, found none.")
        
        # Pre-step before step 4: if there are several top-level definitions, deal with them one by one
        if len(top_defs) > 1:
            if allow_several_top_defs:
                import_lines = [line for line in code.splitlines() if re.match(r"^\s*(import|from)\s", line)]
                for node in top_defs:
                    single_def_code = ast.get_source_segment(code, node)
                    if single_def_code is None:
                        raise ValueError("Could not extract source code for a top-level item.")
                    code_block = "\n".join(import_lines + ["", single_def_code])
                    self.edit_code(code_block, allow_several_top_defs=False)
                return
            else:
                raise ValueError(f"Code must contain exactly one top-level definition, but found {len(top_defs)}.")

        # Step 4: Extract name, imports, docstring, and pure code
        top_def = top_defs[0]
        if isinstance(top_def, ast.FunctionDef):
            name = top_def.name
        elif isinstance(top_def, ast.ClassDef):
            name = top_def.name
        elif isinstance(top_def, ast.Assign):
            if len(top_def.targets) != 1 or not isinstance(top_def.targets[0], ast.Name):
                raise ValueError("Assignment must be to a single variable name.")
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

        docstring_match = re.match(r'^\s*[\'"]{3}(.*?)[\'"]{3}', "\n".join(definition_lines), re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1).strip()
            code_wo_docstring = re.sub(r'^\s*[\'"]{3}(.*?)[\'"]{3}', '', "\n".join(definition_lines), flags=re.DOTALL)
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
            print(f"Warning: Could not parse code for dependencies: {pure_code}. Still proceeding without dependencies.")

        dependencies = [dep for dep in used_names if dep in self.items and dep != name]

        # Step 6: Create Item (temporary or final)
        new_item = Item(
            name=name,
            pure_code=pure_code,
            item_type="pending",  # will determine after exec
            imports=import_lines,
            dependencies=dependencies,
            description=docstring
        )

        is_new = name not in self.items
        old_item = self.items.get(name)

        # Step 7: Temporarily insert item, check circular deps
        self.items[name] = new_item
        try:
            if self._has_circular_dependency():
                raise ValueError(f"Circular dependency detected when adding item '{name}'.")

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

            code_to_exec = "\n".join(sorted(all_imports)) + "\n\n" + "\n\n".join(code_blocks)

            # Step 9: Execute code
            temp_namespace = self.namespace.copy()
            exec(code_to_exec, temp_namespace)

            obj = temp_namespace[name]
            if isinstance(obj, type) and issubclass(obj, Controller):
                item_type = ITEM_TYPE_CONTROLLER
            elif isinstance(obj, Knowledge):
                item_type = ITEM_TYPE_KNOWLEDGE
            else:
                item_type = ITEM_TYPE_BASIC_OBJECT

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

    def _delete_item(self, item_name: str):
        if item_name not in self.items:
            raise ValueError(f"Cannot delete '{item_name}': item does not exist.")

        # Check if any item depends on it
        dependents = [
            other.name for other in self.items.values()
            if item_name in other.dependencies
        ]
        if dependents:
            raise ValueError(
                f"Cannot delete '{item_name}' because it is used by: {', '.join(dependents)}"
            )

        del self.items[item_name]