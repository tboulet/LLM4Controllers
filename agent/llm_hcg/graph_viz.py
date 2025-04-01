import os
from typing import Dict, List, Tuple
import webbrowser
import networkx as nx
from pyvis.network import Network
import time


class ControllerVisualizer:
    def __init__(
        self,
        agent: "LLMBasedHCG",
        log_dirs: List[str] = ["logs"],
        auto_open_browser: bool = False,
    ):
        # Get parameters
        self.agent = agent
        self.log_dirs = log_dirs if isinstance(log_dirs, list) else [log_dirs]
        for log_dir in self.log_dirs:
            os.makedirs(log_dir, exist_ok=True)
        self.auto_open_browser = auto_open_browser
        # Initialize the graph
        self.G = nx.DiGraph()
        self.PCs: Dict[str, str] = {}
        self.positions: Dict[str, Tuple[int, int]] = {}
        self.time_step = 0
        self.x_current_step = 0
        self.is_update_vis_called = False

    def add_PCs(self, PCs: Dict[str, str]):
        self.PCs.update(PCs)
        for controller_name, pc_code in PCs.items():
            node = f"{controller_name}_{self.time_step}"
            self.G.add_node(node, label=controller_name, title=pc_code)
            n_nodes_in_this_step = self.get_first_available_y_position()
            x_position = self.x_current_step + n_nodes_in_this_step
            y_position = n_nodes_in_this_step
            self.positions[node] = (x_position, y_position)

    def remove_controllers(self, pc_names: List[str]):
        for pc_name in pc_names:
            if pc_name in self.PCs:
                del self.PCs[pc_name]

    def new_step(self):
        self.time_step += 1
        self.x_current_step = max([x for x, _ in self.positions.values()]) + 3
        new_positions = {}
        max_y = 0

        for name, pc_code in self.PCs.items():
            old_node = f"{name}_{self.time_step - 1}"
            new_node = f"{name}_{self.time_step}"

            self.G.add_node(new_node, label=name, title=pc_code)
            self.G.add_edge(old_node, new_node)
            x_last, y_last = self.positions[old_node]
            x_position = self.x_current_step + y_last
            y_position = y_last
            new_positions[new_node] = (x_position, y_position)
            max_y = max(max_y, y_position)

        self.positions.update(new_positions)

        # Add vertical bar to indicate time step
        bar_x = self.x_current_step - 1
        bar_top = f"bar_top_{self.time_step}"
        bar_bottom = f"bar_bottom_{self.time_step}"

        self.G.add_node(bar_top, label="", shape="square", color="black")
        self.G.add_node(bar_bottom, label="", shape="square", color="black")
        self.G.add_edge(bar_top, bar_bottom, color="black", width=2)

        self.positions[bar_top] = (bar_x, max_y + 1)
        self.positions[bar_bottom] = (bar_x, -1)

    def get_first_available_y_position(self):
        y_positions_taken = [
            y for x, y in self.positions.values() if x >= self.x_current_step
        ]
        y_available = 0
        while y_available in y_positions_taken:
            y_available += 1
        return y_available

    def update_vis(self):
        # Create the network
        net = Network(notebook=True, directed=True)
        net.from_nx(self.G)
        for node, (x, y) in self.positions.items():
            net.get_node(node)["x"] = x * 100
            net.get_node(node)["y"] = -y * 100
            net.get_node(node)["physics"] = False
            net.get_node(node)["label"] = (
                ""
                if "bar_top" in node or "bar_bottom" in node
                else net.get_node(node)["label"]
            )
        # Save the graph
        for log_dir in self.log_dirs:
            file_path = os.path.abspath(rf"{log_dir}/graph.html")
            net.show(file_path)
        # Notify the user that the graph is ready, and eventually open it in the browser
        if not self.is_update_vis_called:
            self.is_update_vis_called = True
            print(
                (
                    "[VISUALIZER] : Visualizer initialized. "
                    f"\033]8;;{file_path}\033\\Click here to open the graph in your browser\033]8;;\033\\"
                )
            )
            if self.auto_open_browser:
                webbrowser.open(file_path)  # open in browser the last graph


if __name__ == "__main__":
    viz = ControllerVisualizer(log_dirs="logs/_last")

    viz.update_vis()

    viz.add_PCs({"PC1": "code1", "PC2": "code2", "PC3": "code3"})
    viz.update_vis()

    viz.new_step()
    viz.update_vis()
    viz.add_PCs({"PC4": "code4", "PC5": "code5"})
    viz.update_vis()

    viz.new_step()
    viz.update_vis()

    viz.remove_controllers(["PC1", "PC5"])
    viz.update_vis()

    viz.new_step()
    viz.update_vis()

    viz.add_PCs({"PC6": "code6"})
    viz.update_vis()

    viz.new_step()
    viz.update_vis()
