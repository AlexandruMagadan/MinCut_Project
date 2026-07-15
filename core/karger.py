import random
import networkx as nx
import scipy.io as sio


def load_raw_graph_data(filepath):
    """Loads raw graph data from a file, bypassing NetworkX for maximum speed."""
    edges_list = []
    nodes_set = set()

    if filepath.suffix == ".mtx":
        sparse_matrix = sio.mmread(filepath, spmatrix=False)
        G_temp = nx.MultiGraph(sparse_matrix)
        edges_list = list(G_temp.edges(keys=False))
        nodes_set = set(G_temp.nodes())
    elif filepath.suffix == ".edges":
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('%') or line.startswith('#'): continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    u, v = int(float(parts[0])), int(float(parts[1]))
                    edges_list.append((u, v))
                    nodes_set.add(u)
                    nodes_set.add(v)
    return list(nodes_set), edges_list


def compute_actual_mincut_core(nodes_list, edges, n, h):
    """Runs the Union-Find algorithm h times to find the guaranteed minimum cut."""
    min_cut_found = float('inf')
    best_partition = {}

    for _ in range(h):
        edges_iter = edges.copy()
        random.shuffle(edges_iter)
        parent = {node: node for node in nodes_list}

        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]

        num_components = n
        for u, v in edges_iter:
            if num_components <= 2: break
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                parent[root_v] = root_u
                num_components -= 1

        cut_size = 0
        for u, v in edges_iter:
            if find(u) != find(v): cut_size += 1

        if cut_size < min_cut_found:
            min_cut_found = cut_size
            temp_partition = {}
            for node in nodes_list:
                root = find(node)
                if root not in temp_partition:
                    temp_partition[root] = []
                temp_partition[root].append(node)
            best_partition = temp_partition

    return min_cut_found, best_partition


def simulate_complexity_core(nodes_list, edges, n, h):
    """Runs Karger and counts fundamental operations for the complexity study."""
    min_cut_found = float('inf')
    fundamental_operations = 0

    for _ in range(h):
        edges_iter = edges.copy()
        random.shuffle(edges_iter)
        parent = {node: node for node in nodes_list}

        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]

        num_components = n
        for u, v in edges_iter:
            fundamental_operations += 1
            if num_components <= 2: break
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                parent[root_v] = root_u
                num_components -= 1

        cut_size = 0
        for u, v in edges_iter:
            fundamental_operations += 1
            if find(u) != find(v): cut_size += 1

        if cut_size < min_cut_found:
            min_cut_found = cut_size

    return fundamental_operations


def skip_to_end_core(nodes_list, edges, node_contents, current_step):
    """Executes contractions rapidly and generates the log for UI visualization."""
    random.shuffle(edges)
    parent = {n: n for n in nodes_list}

    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    num_components = len(nodes_list)
    log_messages = []
    step = current_step

    for u, v in edges:
        if num_components <= 2: break
        root_u, root_v = find(u), find(v)

        if root_u != root_v:
            node_contents[root_u].extend(node_contents[root_v])
            del node_contents[root_v]

            msg = f"STEP {step}: Node {root_v} merges into {root_u}.\nSuper-node [{root_u}] now contains:\n{node_contents[root_u]}\n\n"
            log_messages.append(msg)

            parent[root_v] = root_u
            num_components -= 1
            step += 1

    G_final = nx.MultiGraph()
    G_final.add_nodes_from(node_contents.keys())

    for u, v in edges:
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            G_final.add_edge(root_u, root_v)

    return G_final, log_messages, step, node_contents