import networkx as nx
from pyvis.network import Network
import time


class VisualizerHCG:
    """Visualizer for the Hierarchical Controller Generation agent controllers.
    Based on NetworkX and Pyvis for visualization.
    Open file <log_dir>/viz.html in your browser to see the visualization.
    """

    def __init__(self, agent, config: dict):
        self.G = nx.DiGraph()
        self.agent = agent
        self.config = config
        self.controllers = {}  # Stores active controllers and their codes
        self.positions = {}  # Stores positions for visualization
        self.time_step = 0  # X-axis position in time
        self.n_cum_controllers = 0  # Number of controllers added so far
        
    def initialize(self, controllers):
        """Initialize the visualization with a given set of controllers."""
        self.controllers = controllers
        for idx, name in enumerate(controllers.keys()):
            node_id = f"{name}_0"
            self.G.add_node(node_id, label=name, title=controllers[name])
            self.positions[node_id] = (0, -idx)  # x=0, y varies

    def new_step(self):
        """Move time forward by one step, keeping track of active controllers."""
        self.time_step += 1
        new_positions = {}
        for idx, name in enumerate(list(self.controllers.keys())):
            old_node = f"{name}_{self.time_step - 1}"
            new_node = f"{name}_{self.time_step}"

            self.G.add_node(new_node, label=name, title=self.controllers[name])
            self.G.add_edge(old_node, new_node)
            new_positions[new_node] = (
                self.time_step,
                self.positions[old_node][1],
            )  # Keep y position same

        self.positions.update(new_positions)

    def add_pc(self, name : str, code : str):
        """Add a new controller"""
        self.controllers[name] = code
        new_node = f"{name}_{self.time_step}"
        self.G.add_node(new_node, label=name, title=code)
        self.positions[new_node] = (self.time_step, self.n_cum_controllers)
        self.n_cum_controllers += 1

    def remove_controllers(self, name : str):
        """Remove controller from the internal state (they will not move forward)."""
        if name in self.controllers:
            del self.controllers[name]

    def generate_html(self):
        """Generate the visualization as an HTML file and display it."""
        net = Network(notebook=True, directed=True)
        net.from_nx(self.G)
        for node, (x, y) in self.positions.items():
            # Explicitly set positions to avoid internal adjustments by pyvis
            net.get_node(node)["x"] = x * 100  # x position scaling
            net.get_node(node)["y"] = y * 100  # y position scaling
            net.get_node(node)[
                "physics"
            ] = False  # Disable physics so positions stay fixed
        net.show(
            "temp/controllers.html"
        )  # Open the visualization directly in your browser


if __name__ == "__main__":
    viz = VisualizerHCG()
    viz.initialize({"A": "def foo(): pass", "B": "def bar(): pass"})

    # Main loop to update the visualization every 2 seconds
    print("Visualization started. Display <log_dir>/viz.vtml in your browser at ")

    try:
        while True:
            viz.new_step()  # Update the controllers' positions
            viz.generate_html()  # Generate and update the HTML file
            time.sleep(2)  # Wait for 2 seconds before the next update
    except KeyboardInterrupt:
        print("Exiting...")
