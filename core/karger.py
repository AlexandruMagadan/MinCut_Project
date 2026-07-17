import random

import networkx as nx
import numpy as np
import scipy.io as sio

"""
Test
"""
def print_adj_list(l):
    print("--- Generated Adjacency List (Preview) ---")
    print(f"Total nodes in graph: {len(l)}")
    count = 0
    for node, neighbors in sorted(l.items()):
        print(f"Node {node}: {neighbors}")
        count += 1
        if count >= 100:
            print("... (output limited to first 100 nodes for readability)")
            break

def load_adjacency_list(filepath):
    """
    Read the graph from data and build the Adjacency List asociated
    """
    adj = {}

    if filepath.suffix == ".mtx":
        sparse_matrix = sio.mmread(filepath, spmatrix=False)
        G_temp = nx.MultiGraph(sparse_matrix)

        for node in G_temp.nodes():
            adj[node] = []

        # Populăm listele cu muchiile (inclusiv muchiile multiple)
        for u, v in G_temp.edges(keys=False):
            adj[u].append(v)
            adj[v].append(u)

    elif filepath.suffix == ".edges":
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('%') or line.startswith('#'):
                    continue

                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    u = int(float(parts[0]))
                    v = int(float(parts[1]))

                    # Dacă nodurile nu există în dicționar, le creăm
                    if u not in adj:
                        adj[u] = []
                    if v not in adj:
                        adj[v] = []

                    # Adăugăm vecinii (graf neorientat)
                    adj[u].append(v)
                    adj[v].append(u)

    return adj

def rand_edge_choise(l):
    active_vertices=list(l.keys())
    degrees = [len(l[node]) for node in active_vertices]
    u_rand = random.choices(active_vertices,weights=degrees)[0]
    v_rand = random.choice(l[u_rand])

    return (u_rand,v_rand)

def contract(l,u,v):
    if u == v:
        l[u] = [neighbor for neighbor in l[u] if neighbor != u]
        return l

    for neighbor in l[v]:
        for i in range(len(l[neighbor])):
            if l[neighbor][i] == v:
                l[neighbor][i] = u
    l[u].extend(l[v])
    del l[v]
    l[u] = [neighbor for neighbor in l[u] if neighbor !=u]
    return l

def karger_iteration(l):
    e=rand_edge_choise(l)
    l=contract(l,e[0],e[1])
    return l,e[0], e[1]

def karger(l):
    while len(l) > 2:
        l,u,v=karger_iteration(l)

    noduri_ramase = list(l.keys())
    print(noduri_ramase)
    dimensiune_taietura = len(l[noduri_ramase[0]])
    return dimensiune_taietura
def karger_iteraded(l):
    n=len(l)
    h = (n ** 2) // 2
    min=float('inf')
    for i in range (h):
        l_copy = {node: neighbors.copy() for node, neighbors in l.items()}
        k=karger(l_copy)
        if k < min:
            min=k
    return min

if __name__ == "__main__":
    from pathlib import Path
    import sys

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    if not data_dir.exists():
        print(f"Error: Folder '{data_dir}' not found.")
        sys.exit()


    valid_files = [f for f in data_dir.glob("*.*") if f.suffix in ['.mtx', '.edges']]

    if not valid_files:
        print("Error: No .mtx or .edges files found in the 'data' folder.")
        sys.exit()

    # Alegem automat primul fisier gasit in folder
    test_file = random.choice(valid_files)
    print(f"Loading real graph from: {test_file.name}...\n")

    # Apelam functia ta
    adj_list = load_adjacency_list(test_file)
    print(karger_iteraded(adj_list))
