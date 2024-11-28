import tkinter as tk
from tkinter import filedialog, messagebox
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


class NetworkAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Social Network Analysis")
        self.root.configure(bg="#000000")  
        self.network = None

        # GUI Layout
        self.create_widgets()

    def create_widgets(self):
    
        container = tk.Frame(self.root, bg="#2b2b2b")
        container.pack(padx=150)
        tk.Label(
            self.root,
            text="Social Network Analysis",
            font=("Century Gothic", 18, "bold"),
            bg="#000000",
            fg="#00FF7F", 
        ).pack(pady=15)

        self.upload_button = tk.Button(
            self.root,
            text="Upload Adjacency Matrix (CSV)",
            command=self.upload_file,
            bg="#1a1a1a",  # Dark button
            fg="#FFFFFF",
            font=("Century Gothic", 12),
            bd=0,  # Sharp edges
            activebackground="#333333",
            activeforeground="#00FF7F",
        )
        self.upload_button.pack(pady=10)

        self.features_label = tk.Label(
            self.root,
            text="Select Feature to Perform:",
            font=("Century Gothic", 14),
            bg="#000000",
            fg="#FFFFFF",
        )
        self.features_label.pack(pady=10)

        self.features_frame = tk.Frame(self.root, bg="#000000")
        self.features_frame.pack(pady=10)

        # Add feature buttons
        features = [
            ("Network Summary", self.network_summary),
            ("Link Prediction", self.link_prediction),
            ("Community Detection", self.community_detection),
            ("Degree Centrality", self.centrality_measures),
            ("Network Modularity", self.network_modularity),
            ("Betweenness", self.calculate_betweenness),
            ("Articulation Points", self.articulation_points),
            ("Shortest Path", self.shortest_path),
            ("Network Diameter", self.network_diameter),
            ("Clustering Coefficient", self.clustering_coefficient),
            ("Degree Distribution", self.degree_distribution),
            ("Plot Network", self.plot_network),
            ("Save Network Image", self.save_network_image),
        ]

        for text, command in features:
            tk.Button(
                self.features_frame,
                text=text,
                command=command,
                width=30,
                bg="#1a1a1a",
                fg="#FFFFFF",
                font=("Century Gothic", 10),
                bd=0,
                activebackground="#333333",
                activeforeground="#00FF7F",
            ).pack(pady=5)

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path, index_col=0)
                self.network = nx.from_pandas_adjacency(df)
                messagebox.showinfo("Success", "Adjacency matrix successfully loaded!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load the adjacency matrix: {e}")

    def link_prediction(self):
        if self.network is None:
            self.show_error()
            return
        preds = list(nx.common_neighbor_centrality(self.network))
        result = "Link Prediction:\n" + "\n".join([f"{u}-{v}: {score:.2f}" for u, v, score in preds])
        self.show_result(result)

    def community_detection(self):
        if self.network is None:
            self.show_error()
            return
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(self.network))
        result = f"Detected {len(communities)} communities:\n"
        result += "\n".join([f"Community {i+1}: {list(community)}" for i, community in enumerate(communities)])
        self.show_result(result)

    def centrality_measures(self):
        if self.network is None:
            self.show_error()
            return
        degree_centrality = nx.degree_centrality(self.network)
        result = "Degree Centrality:\n" + "\n".join([f"{node}: {centrality:.2f}" for node, centrality in degree_centrality.items()])
        self.show_result(result)

    def network_modularity(self):
        if self.network is None:
            self.show_error()
            return
        from networkx.algorithms.community import modularity
        communities = list(nx.community.greedy_modularity_communities(self.network))
        mod_value = modularity(self.network, communities)
        self.show_result(f"Network Modularity: {mod_value:.2f}")

    def calculate_betweenness(self):
        if self.network is None:
            self.show_error()
            return
        betweenness = nx.betweenness_centrality(self.network)
        result = "Betweenness Centrality:\n" + "\n".join([f"{node}: {score:.2f}" for node, score in betweenness.items()])
        self.show_result(result)

    def articulation_points(self):
        if self.network is None:
            self.show_error()
            return
        points = list(nx.articulation_points(self.network))
        result = "Articulation Points:\n" + ", ".join(map(str, points))
        self.show_result(result)

    def plot_network(self):
        if self.network is None:
            self.show_error()
            return
        # Create a plot for the network
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.network)  # Use the spring layout for better visualization
        nx.draw(
            self.network, pos, with_labels=True, node_color='skyblue',
            edge_color='gray', node_size=500, font_size=6
        )
        plt.title("Network Visualization")
        plt.show()

    def shortest_path(self):
        if self.network is None:
            self.show_error()
            return
       
        path_window = tk.Toplevel(self.root)
        path_window.title("Find Shortest Path")
        
        tk.Label(path_window, text="Source Node:").pack(pady=5)
        source_entry = tk.Entry(path_window)
        source_entry.pack(pady=5)

        tk.Label(path_window, text="Target Node:").pack(pady=5)
        target_entry = tk.Entry(path_window)
        target_entry.pack(pady=5)

        def calculate_path():
            source = source_entry.get()
            target = target_entry.get()
            if source in self.network and target in self.network:
                try:
                    path = nx.shortest_path(self.network, source=source, target=target)
                    result = f"Shortest Path from {source} to {target}: {path}"
                except nx.NetworkXNoPath:
                    result = f"No path exists between {source} and {target}"
            else:
                result = "Invalid nodes."
            self.show_result(result)

        tk.Button(path_window, text="Find Path", command=calculate_path).pack(pady=10)

        
        
    def clustering_coefficient(self):
        if self.network is None:
            self.show_error()
            return
        clustering = nx.clustering(self.network)
        result = "Clustering Coefficient:\n" + "\n".join([f"{node}: {value:.2f}" for node, value in clustering.items()])
        self.show_result(result)

    def network_diameter(self):
        if self.network is None:
            self.show_error()
            return
        try:
            diameter = nx.diameter(self.network)
            self.show_result(f"Network Diameter: {diameter}")
        except nx.NetworkXError:
            self.show_result("Network is not connected. Diameter cannot be calculated.")
    
    def network_summary(self):
        if self.network is None:
            self.show_error()
            return
        num_nodes = self.network.number_of_nodes()
        num_edges = self.network.number_of_edges()
        result = f"Network Summary:\nNodes: {num_nodes}\nEdges: {num_edges}"
        self.show_result(result)

    def save_network_image(self):
        if self.network is None:
            self.show_error()
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(self.network)
            nx.draw(
                self.network, pos, with_labels=True, node_color='skyblue',
                edge_color='gray', node_size=500, font_size=10
            )
            plt.title("Network Visualization")
            plt.savefig(file_path)
            messagebox.showinfo("Success", f"Network image saved as {file_path}")
    
    def degree_distribution(self):
        if self.network is None:
            self.show_error()
            return
        degrees = [degree for _, degree in self.network.degree()]
        plt.figure(figsize=(8, 6))
        plt.hist(degrees, bins=range(1, max(degrees) + 2), align="left", rwidth=0.8)
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()

    def show_result(self, result):
        def search_keyword():
            keyword = search_entry.get()
            text.tag_remove("highlight", "1.0", tk.END)  
            if keyword:
                start_pos = "1.0"
                found = False 
                while True:
                    start_pos = text.search(keyword, start_pos, stopindex=tk.END, nocase=1)
                    if not start_pos:
                        break
                    end_pos = f"{start_pos}+{len(keyword)}c"
                    text.tag_add("highlight", start_pos, end_pos)
                    if not found:  # Scroll to the first match only
                        text.see(start_pos)
                        found = True
                    start_pos = end_pos
                text.tag_config("highlight", background="#00FF7F", foreground="#000000")  # Highlight style

        result_window = tk.Toplevel(self.root)
        result_window.title("Result")
        result_window.configure(bg="#000000")

        
        search_frame = tk.Frame(result_window, bg="#000000")
        search_frame.pack(pady=5, padx=5, fill="x")

        search_label = tk.Label(
            search_frame,
            text="Search:",
            font=("Century Gothic", 10),
            bg="#000000",
            fg="#FFFFFF",
        )
        search_label.pack(side="left", padx=5)

        search_entry = tk.Entry(
            search_frame,
            font=("Century Gothic", 10),
            bg="#1a1a1a",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",  # Cursor color
            width=30,
        )
        search_entry.pack(side="left", padx=5)

        search_button = tk.Button(
            search_frame,
            text="Find",
            command=search_keyword,
            bg="#1a1a1a",
            fg="#FFFFFF",
            font=("Century Gothic", 10),
            bd=0,
            activebackground="#333333",
            activeforeground="#00FF7F",
        )
        search_button.pack(side="left", padx=5)

        
        text = tk.Text(
            result_window,
            wrap="word",
            width=60,
            height=20,
            bg="#1a1a1a",
            font=("Century Gothic", 10),
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
        )
        text.insert("1.0", result)
        text.configure(state="normal") 
        text.pack(padx=10, pady=10)

       
        text.tag_configure("highlight", background="#00FF7F", foreground="#000000")

    def show_error(self):
        messagebox.showerror("Error", "No network loaded. Please upload an adjacency matrix first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkAnalysisApp(root)
    root.mainloop()
