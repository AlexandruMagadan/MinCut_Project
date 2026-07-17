import customtkinter as ctk
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import time
import threading

from core.karger import load_adjacency_list,karger_iteration, karger_iteraded

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class MinCutApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Min-Cut Simulator (Karger's Algorithm)")
        self.geometry("1050x800")

        self.adj_list = None
        self.data_dir = Path(__file__).resolve().parent.parent / "data"

        self.adj_viz = None
        self.viz_current_step = 0
        self.node_contents = {}

        self.title_label = ctk.CTkLabel(self, text="Analysis and Visualization of the Min-Cut Algorithm",
                                        font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=15)

        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.pack(pady=5, padx=20, fill="x")
        self.control_frame.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="col")

        files = self.get_graph_files()

        self.file_combo = ctk.CTkComboBox(self.control_frame, values=files)
        self.file_combo.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.load_button = ctk.CTkButton(self.control_frame, text="1. Load Graph", command=self.load_selected_graph)
        self.load_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.calc_actual_btn = ctk.CTkButton(self.control_frame, text="2. Calculate Actual Min-Cut",
                                             command=self.calculate_actual_mincut, fg_color="#2980b9",
                                             hover_color="#3498db", state="disabled")
        self.calc_actual_btn.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        self.bulk_button = ctk.CTkButton(self.control_frame, text="4. Complexity Study (Folder)",
                                         command=self.process_collection, fg_color="#27ae60", hover_color="#2ecc71")
        self.bulk_button.grid(row=0, column=3, padx=10, pady=10, sticky="ew")

        self.viz_start_btn = ctk.CTkButton(self.control_frame, text="3. Visualization: Start",
                                           command=self.start_visualization, fg_color="#8e44ad", hover_color="#9b59b6",
                                           state="disabled")
        self.viz_start_btn.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.viz_next_btn = ctk.CTkButton(self.control_frame, text="Next Step", command=self.execute_visualization_step,
                                          state="disabled", fg_color="#f39c12", hover_color="#d68910")
        self.viz_next_btn.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.viz_skip_btn = ctk.CTkButton(self.control_frame, text="Skip to End",
                                          command=self.skip_to_end_visualization, state="disabled", fg_color="#e74c3c",
                                          hover_color="#c0392b")
        self.viz_skip_btn.grid(row=1, column=2, padx=10, pady=10, sticky="ew")

        self.info_label = ctk.CTkLabel(self, text="Select a graph from the list to begin.", font=ctk.CTkFont(size=14),
                                       height=30)
        self.info_label.pack(pady=5)

        self.main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content_frame.pack(pady=10, fill="both", expand=True, padx=20)

        self.plot_frame = ctk.CTkFrame(self.main_content_frame)
        self.plot_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.canvas_widget = None

        self.log_box = ctk.CTkTextbox(self.main_content_frame, width=320, font=ctk.CTkFont(size=13))
        self.log_box.pack(side="right", fill="y")
        self.log_box.insert("0.0", "--- Algorithm Log ---\n\n")

    def get_graph_files(self):
        if not self.data_dir.exists():
            return ["Folder 'data' does not exist"]
        files = [f.name for f in self.data_dir.glob("*.*") if f.suffix in ['.mtx', '.edges']]
        return files if files else ["No valid files found"]

    def load_selected_graph(self):
        filename = self.file_combo.get()
        if not filename or filename in ["Folder 'data' does not exist", "No valid files found"]:
            self.info_label.configure(text=f"Error: Check the 'data' folder!", text_color="#e74c3c")
            return

        filepath = self.data_dir / filename
        try:

            self.adj_list = load_adjacency_list(filepath)

            n = len(self.adj_list)
            m = sum(len(neighbors) for neighbors in self.adj_list.values()) // 2

            self.info_label.configure(text=f"Loaded: {filename} (Nodes: {n}, Edges: {m})", text_color="#2ecc71")
            self.log_box.delete("0.0", "end")
            self.log_box.insert("end", "--- Algorithm Log ---\n\n")

            self.calc_actual_btn.configure(state="normal")
            self.viz_start_btn.configure(state="normal")
            self.viz_next_btn.configure(state="disabled")
            self.viz_skip_btn.configure(state="disabled")

            G_initial = self._adj_to_networkx(self.adj_list)
            self.draw_graph_ui(G_initial, title=f"Initial Topology: {filename}")

        except Exception as e:
            self.info_label.configure(text=f"Read Error: {e}", text_color="#e74c3c")

    def _adj_to_networkx(self, adj):
        G = nx.MultiGraph()
        for node in adj.keys():
            G.add_node(node)
        for u, neighbors in adj.items():
            for v in neighbors:
                if u <= v:
                    G.add_edge(u, v)
        return G

    def calculate_actual_mincut(self):
        if self.adj_list is None or len(self.adj_list) < 2: return
        n = len(self.adj_list)
        h = (n ** 2) // 2

        self.calc_actual_btn.configure(state="disabled")
        self.viz_start_btn.configure(state="disabled")
        self.bulk_button.configure(state="disabled")

        self.info_label.configure(text=f"Computing guaranteed Min-Cut... (running {h} iterations)",
                                  text_color="#f39c12")
        self.log_box.delete("0.0", "end")
        self.log_box.insert("end",
                            f"--- Min-Cut Search ---\nGraph has {n} nodes.\nRunning h = {h} Monte Carlo iterations.\n\nComputing...\n")
        self.update()

        threading.Thread(target=self._worker_actual_mincut, args=(self.adj_list,), daemon=True).start()

    def _worker_actual_mincut(self, adj_list):
        start_time = time.time()

        min_cut_found = karger_iteraded(adj_list)

        total_time = time.time() - start_time
        self.after(0, lambda: self._finalize_actual_mincut(min_cut_found, total_time))

    def _finalize_actual_mincut(self, min_cut_found, total_time):
        self.info_label.configure(text=f"Calculation finished in {total_time:.2f} seconds! Min-Cut: {min_cut_found}",
                                  text_color="#2ecc71")
        self.log_box.insert("end", f"\n--- FINAL RESULT ---\nGuaranteed Min-Cut Value: {min_cut_found}\n")
        self.log_box.see("end")

        self.calc_actual_btn.configure(state="normal")
        self.viz_start_btn.configure(state="normal")
        self.bulk_button.configure(state="normal")

    def start_visualization(self):
        if self.adj_list is None: return
        if len(self.adj_list) > 100:
            self.info_label.configure(text="Error: Graph is too dense for clear visualization.", text_color="#e74c3c")
            return

        self.adj_viz = {node: neighbors.copy() for node, neighbors in self.adj_list.items()}
        self.viz_current_step = 1
        self.node_contents = {n: [n] for n in self.adj_viz.keys()}

        self.viz_next_btn.configure(state="normal")
        self.viz_skip_btn.configure(state="normal")
        self.calc_actual_btn.configure(state="disabled")

        self.log_box.delete("0.0", "end")
        self.log_box.insert("end", "--- Single Run Visualization ---\nNOTE: This run uses Graph Minor explicitly.\n\n")
        self.info_label.configure(text="Visualization Mode Activated. Press 'Next Step'.", text_color="white")

        G_draw = self._adj_to_networkx(self.adj_viz)
        self.draw_graph_ui(G_draw, title="Initial State")

    def execute_visualization_step(self):
        if len(self.adj_viz) <= 2:
            self.finish_visualization()
            return


        self.adj_viz, u, v = karger_iteration(self.adj_viz)


        self.node_contents[u].extend(self.node_contents[v])
        del self.node_contents[v]

        self.log_box.insert("end",
                            f"STEP {self.viz_current_step}: Node {v} merges into {u}.\nRemaining Nodes: {len(self.adj_viz)}\n\n")
        self.log_box.see("end")

        self.info_label.configure(
            text=f"Step {self.viz_current_step}: {v} merged into {u}. Remaining: {len(self.adj_viz)}",
            text_color="#f1c40f")

        G_draw = self._adj_to_networkx(self.adj_viz)
        self.draw_graph_ui(G_draw, title=f"Contraction Step {self.viz_current_step}", highlight_node=u)

        self.viz_current_step += 1
        if len(self.adj_viz) <= 2:
            self.finish_visualization()

    def skip_to_end_visualization(self):
        if self.adj_viz is None or len(self.adj_viz) <= 2: return
        self.info_label.configure(text="Calculating final cut and log history...", text_color="white")
        self.update()

        while len(self.adj_viz) > 2:
            self.adj_viz, u, v = karger_iteration(self.adj_viz)

            self.node_contents[u].extend(self.node_contents[v])
            del self.node_contents[v]

            self.viz_current_step += 1

        self.log_box.insert("end", "Skipped to end.\n")
        self.log_box.see("end")
        self.finish_visualization()

    def finish_visualization(self):
        self.viz_next_btn.configure(state="disabled")
        self.viz_skip_btn.configure(state="disabled")
        self.calc_actual_btn.configure(state="normal")

        noduri_ramase = list(self.adj_viz.keys())
        cut_size = len(self.adj_viz[noduri_ramase[0]])

        self.info_label.configure(text=f"Visual Run Completed. Cut Found = {cut_size}", text_color="#2ecc71")
        G_draw = self._adj_to_networkx(self.adj_viz)
        self.draw_graph_ui(G_draw, title=f"Final Partition (Cut = {cut_size})")

    def process_collection(self):
        valid_files = [f for f in self.data_dir.glob("*.*") if f.suffix in ['.mtx', '.edges']]
        if not valid_files: return

        self.bulk_button.configure(state="disabled")
        self.calc_actual_btn.configure(state="disabled")
        self.viz_start_btn.configure(state="disabled")
        self.info_label.configure(text=f"Starting complexity projection for {len(valid_files)} files...",
                                  text_color="white")

        threading.Thread(target=self._worker_analysis, args=(valid_files,), daemon=True).start()

    def _worker_analysis(self, valid_files):
        results = []
        for file in valid_files:
            try:
                adj = load_adjacency_list(file)
                n = len(adj)
                if n > 150 or n < 2: continue
                ops = (n ** 3) // 2
                results.append((n, ops))

            except Exception as e:
                print(f"Internal error on {file.name}: {e}")

        if results:
            results.sort(key=lambda x: x[0])
            self.after(0, lambda: self._finalize_analysis(results))
        else:
            self.after(0, self._analysis_error)

    def _finalize_analysis(self, results):
        self.draw_complexity_plot(results)
        self.info_label.configure(text="Complexity study completed.", text_color="#2ecc71")
        self.bulk_button.configure(state="normal")
        if self.adj_list:
            self.calc_actual_btn.configure(state="normal")
            self.viz_start_btn.configure(state="normal")

    def _analysis_error(self):
        self.info_label.configure(text="Error: No graph < 150 nodes processed.", text_color="#e74c3c")
        self.bulk_button.configure(state="normal")
        if self.adj_list:
            self.calc_actual_btn.configure(state="normal")
            self.viz_start_btn.configure(state="normal")

    def draw_graph_ui(self, G_to_draw, title="", highlight_node=None):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
            plt.close('all')

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')

        pos = nx.spring_layout(G_to_draw, seed=42)
        color_map = ["#e74c3c" if node == highlight_node else "#3498db" for node in G_to_draw.nodes()]

        nx.draw_networkx_nodes(G_to_draw, pos, ax=ax, node_size=350, node_color=color_map)
        nx.draw_networkx_labels(G_to_draw, pos, ax=ax, font_color="white", font_size=10, font_weight="bold")

        for u, v, key in G_to_draw.edges(keys=True):
            rad = 0.0 if key == 0 else 0.15 * ((key + 1) // 2) * (1 if key % 2 != 0 else -1)
            nx.draw_networkx_edges(G_to_draw, pos, ax=ax, edgelist=[(u, v)], edge_color="#7f8c8d", width=1.0,
                                   connectionstyle=f'arc3, rad={rad}')

        ax.set_title(title, color="white", pad=10)
        ax.axis('off')

        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas_ui = self.canvas_widget.get_tk_widget()
        canvas_ui.pack(fill="both", expand=True)
        self.plot_frame.update_idletasks()
        self.canvas_widget.draw()

    def draw_complexity_plot(self, results):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
            plt.close('all')

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')

        nodes = [r[0] for r in results]
        operations = [r[1] for r in results]

        ax.plot(nodes, operations, marker='o', linestyle='-', color='#e74c3c', linewidth=2, markersize=6)
        ax.set_title("Theoretical Operations Complexity", color='white', pad=10)
        ax.set_xlabel("Number of Nodes (n)", color='white')
        ax.set_ylabel("Total Mathematical Operations", color='white')

        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax.grid(True, color='#555555', linestyle='--', alpha=0.7)

        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas_ui = self.canvas_widget.get_tk_widget()
        canvas_ui.pack(fill="both", expand=True)
        self.plot_frame.update_idletasks()
        self.canvas_widget.draw()


if __name__ == "__main__":
    app = MinCutApp()
    app.mainloop()