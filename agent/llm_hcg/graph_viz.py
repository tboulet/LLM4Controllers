import os
from typing import Dict, List, Tuple
import networkx as nx
from pyvis.network import Network
import time


class ControllerVisualizer:
    def __init__(self, log_dirs: List[str] = ["logs"]):
        self.G = nx.DiGraph()
        self.PCs : Dict[str, str] = {}  # Stores active controllers and their codes
        self.positions : Dict[str, Tuple[int, int]] = {}  # Stores positions for visualization
        self.time_step = 0  # time step
        self.x_current_step = 0 # X-axis position of the current time step

        # Initialize log directories
        self.log_dirs = log_dirs
        if isinstance(log_dirs, str):
            self.log_dirs = [log_dirs]
        for log_dir in self.log_dirs:
            os.makedirs(log_dir, exist_ok=True)
            
    def add_PCs(self, PCs : Dict[str, str]):
        """Add new controllers at the current time step."""
        self.PCs.update(PCs)
        for controller_name, pc_code in PCs.items():
            node = f"{controller_name}_{self.time_step}"
            self.G.add_node(node, label=controller_name, title=pc_code)
            n_nodes_in_this_step = self.get_first_available_y_position()
            x_position = self.x_current_step + n_nodes_in_this_step
            y_position = n_nodes_in_this_step
            self.positions[node] = (x_position, y_position)
        
    def remove_controllers(self, pc_names : List[str]):
        """Remove controllers from the internal state (they will not move forward)."""
        for pc_name in pc_names:
            if pc_name in self.PCs:
                del self.PCs[pc_name]

    def new_step(self):
        """Move time forward by one step, keeping track of active controllers."""
        self.time_step += 1
        self.x_current_step = max([x for x, _ in self.positions.values()], default=0) + 3  
        new_positions = {}
        max_y = 0  # Track maximum y-position for the vertical bar

        for name, pc_code in self.PCs.items():
            old_node = f"{name}_{self.time_step - 1}"
            new_node = f"{name}_{self.time_step}"
            
            self.G.add_node(new_node, label=name, title=pc_code)
            self.G.add_edge(old_node, new_node)
            x_last, y_last = self.positions[old_node]
            x_position = self.x_current_step + y_last
            y_position = y_last
            new_positions[new_node] = (x_position, y_position)
            max_y = max(max_y, y_position)  # Update max_y

        # Add vertical bar node (special node that will be rendered as a line)
        bar_node = f"time_bar_{self.time_step}"
        self.G.add_node(bar_node, 
                       label=f"Step {self.time_step}",
                       shape="line",
                       color="#888888",
                       width=2,
                       level=-1)  # Ensure it's drawn behind other nodes
        # Position the bar at x=self.x_current_step + 2, centered vertically
        self.positions[bar_node] = (self.x_current_step + 2, max_y / 2 if max_y > 0 else 0)

        self.positions.update(new_positions)

    def get_first_available_y_position(self):
        """Return first available y position in the current time step."""
        y_positions_taken = [y for x, y in self.positions.values() if x >= self.x_current_step]
        y_availabe = 0
        while y_availabe in y_positions_taken:
            y_availabe += 1 
        return y_availabe
    
    def generate_html(self):
        """Generate the visualization as an HTML file and display it."""
        net = Network(notebook=True, directed=True, height="800px", width="100%")
        net.from_nx(self.G)
        
        for node, (x, y) in self.positions.items():
            net.get_node(node)["x"] = x * 100
            net.get_node(node)["y"] = -y * 100  # Invert y-axis for display
            net.get_node(node)["physics"] = False
            
            # Customize the vertical bar appearance
            if "time_bar" in node:
                net.get_node(node)["shape"] = "line"
                net.get_node(node)["color"] = "#888888"
                net.get_node(node)["width"] = 2
                net.get_node(node)["level"] = -1
                # Make the line span vertically by adjusting size
                net.get_node(node)["size"] = 50  # This affects the line length in pyvis
        
        for log_dir in self.log_dirs:
            net.show(f"{log_dir}/graph.html")  # Save the visualization as an HTML file

if __name__ == "__main__":
    viz = ControllerVisualizer(log_dirs="logs/_last")
    
    viz.generate_html()
     
    viz.add_PCs({"PC1": "code1", "PC2": "code2", "PC3": "code3"})
    viz.generate_html()
     
    viz.new_step()
    viz.generate_html()
    viz.add_PCs({"PC4": "code4", "PC5": "code5"})
    viz.generate_html()
     
    viz.new_step()
    viz.generate_html()
     
    viz.remove_controllers(["PC1", "PC5"])
    viz.generate_html()
     
    viz.new_step()
    viz.generate_html()
     
    viz.add_PCs({"PC6": "code6"})
    viz.generate_html()
     
    viz.new_step()
    viz.generate_html()