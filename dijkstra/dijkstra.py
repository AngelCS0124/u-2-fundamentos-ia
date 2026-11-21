import heapq
import matplotlib

# Selección automática de backend: si Tk no está bien instalado (como en el
# Python de Laragon), se prueba con Qt. Instalar con:  pip install PyQt6
def _elegir_backend():
    import importlib
    for backend, modulo in (("TkAgg", "tkinter"), ("QtAgg", "PyQt6"),
                            ("QtAgg", "PyQt5"), ("QtAgg", "PySide6")):
        try:
            if modulo == "tkinter":
                import tkinter
                tkinter.Tk().destroy()      # falla si falta init.tcl
            else:
                importlib.import_module(modulo)
            matplotlib.use(backend, force=True)
            return backend
        except Exception:
            continue
    raise SystemExit("No hay backend gráfico disponible. Ejecuta: pip install PyQt6")

_elegir_backend()

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button

# ============================================================
# 1. ESTRUCTURA DEL GRAFO (transcrito de la imagen)
# ============================================================
EDGES = [
    (29, 7, 5), (29, 21, 3), (21, 11, 2), (21, 24, 2), (7, 11, 5), (7, 14, 3),
    (7, 6, 10), (7, 1, 20), (6, 15, 3), (6, 14, 5), (6, 9, 7), (6, 8, 4),
    (15, 8, 2), (15, 1, 8), (15, 0, 8), (0, 44, 70), (0, 1, 2), (44, 3, 80),
    (44, 5, 69), (5, 3, 2), (5, 20, 2), (5, 69, 3), (3, 2, 9), (3, 69, 6),
    (20, 69, 3), (20, 25, 7), (11, 14, 9), (11, 10, 7), (11, 24, 7),
    (14, 9, 5), (14, 10, 9), (14, 4, 8), (9, 10, 3), (8, 1, 7), (8, 12, 9),
    (8, 4, 15), (10, 1, 5), (1, 2, 3), (1, 23, 7), (1, 12, 7), (2, 69, 11),
    (2, 23, 3), (2, 17, 5), (69, 25, 6), (69, 17, 4), (69, 70, 8),
    (25, 70, 17), (23, 16, 11), (12, 16, 16), (12, 16, 3),  # arista doble 12-16
    (12, 18, 4), (12, 66, 7), (12, 13, 9), (17, 16, 7), (17, 70, 6),
    (16, 13, 3), (10, 4, 6), (10, 27, 9), (27, 4, 3), (27, 66, 1),
    (4, 18, 5), (18, 66, 6), (66, 13, 777),
]

# Posiciones aproximadas (x, y) tomadas de la imagen (y invertida)
_PIX = {
    29: (52, 52), 7: (247, 52), 21: (52, 199), 6: (472, 199), 15: (697, 199),
    0: (872, 97), 44: (1013, 47), 5: (1180, 159), 3: (1014, 199),
    20: (1331, 243), 11: (194, 311), 14: (316, 318), 9: (472, 316),
    8: (598, 351), 1: (872, 351), 2: (1014, 351), 69: (1180, 351),
    25: (1331, 428), 24: (52, 428), 10: (321, 492), 23: (921, 463),
    12: (721, 513), 17: (1053, 531), 70: (1262, 584), 4: (482, 624),
    18: (574, 624), 16: (926, 643), 27: (296, 648), 66: (672, 761),
    13: (833, 755),
}
POS = {n: (x, -y) for n, (x, y) in _PIX.items()}

G = nx.MultiGraph()
G.add_nodes_from(POS)
for i, (u, v, w) in enumerate(EDGES):
    G.add_edge(u, v, key=i, weight=w)

START, TARGET = 14, 20

# ============================================================
# 2. DIJKSTRA: genera una "foto" del estado por cada paso
# ============================================================
def dijkstra_steps(graph, start, target):
    dist = {n: float("inf") for n in graph}
    prev = {n: None for n in graph}
    prev_edge = {}            # vertice -> id de arista usada para llegar
    visited, examined = set(), set()
    dist[start] = 0
    pq = [(0, start)]
    steps = []

    def snap(msg, current=None):
        steps.append(dict(dist=dict(dist), prev=dict(prev), visited=set(visited),
                          prev_edge=dict(prev_edge), examined=set(examined),
                          current=current, msg=msg, path=None))

    snap(f"Inicio: costo({start}) = 0, resto = ∞")
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            path, n = [], target
            while n is not None:
                path.append(n)
                n = prev[n]
            snap(f"Se alcanzó {target} con costo {d}. Algoritmo detenido.", u)
            steps[-1]["path"] = path[::-1]
            return steps
        for _, v, k, data in graph.edges(u, keys=True, data=True):
            if v in visited:
                continue
            examined.add(k)
            nd = d + data["weight"]
            if nd < dist[v]:
                dist[v], prev[v], prev_edge[v] = nd, u, k
                heapq.heappush(pq, (nd, v))
        snap(f"Se visita {u} (costo {d}) y se relajan sus aristas", u)
    snap("No existe ruta al destino")
    return steps

STEPS = dijkstra_steps(G, START, TARGET)

# ============================================================
# 3. INTERFAZ INTERACTIVA
# ============================================================
C_BASE, C_VISIT, C_CUR, C_FRONT, C_PATH = "#1f6f8b", "#2e9e5b", "#e07b39", "#f2c14e", "#d62828"

fig, ax = plt.subplots(figsize=(16, 9))
plt.subplots_adjust(bottom=0.12)
state = {"i": 0}


def edge_curve(u, v, rad):
    (x1, y1), (x2, y2) = POS[u], POS[v]
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    cx, cy = mx - (y2 - y1) * rad, my + (x2 - x1) * rad
    ts = [t / 20 for t in range(21)]
    xs = [(1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2 for t in ts]
    ys = [(1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2 for t in ts]
    return xs, ys, xs[10], ys[10]


def draw():
    s = STEPS[state["i"]]
    ax.clear()
    ax.axis("off")
    path_edges = set()
    if s["path"]:
        path_edges = {s["prev_edge"][v] for v in s["path"][1:]}
    tree_edges = set(s["prev_edge"].values())

    # aristas
    for k, (u, v, w) in enumerate(EDGES):
        rad = 0
        if (u, v) == (12, 16):
            rad = 0.12 if w == 16 else -0.12
        xs, ys, lx, ly = edge_curve(u, v, rad)
        if k in path_edges:
            col, lw, z = C_PATH, 4, 3
        elif k in tree_edges:
            col, lw, z = C_CUR, 3, 2
        elif k in s["examined"]:
            col, lw, z = C_FRONT, 1.8, 1
        else:
            col, lw, z = "#7a8a8f", 1.2, 0
        ax.plot(xs, ys, color=col, lw=lw, zorder=z)
        ax.text(lx, ly, str(w), fontsize=8, fontweight="bold", ha="center", va="center",
                zorder=4, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # vertices
    for n, (x, y) in POS.items():
        if n in (s["path"] or []):
            col = C_PATH
        elif n == s["current"]:
            col = C_CUR
        elif n in s["visited"]:
            col = C_VISIT
        elif s["dist"][n] < float("inf"):
            col = C_FRONT
        else:
            col = C_BASE
        ax.scatter(x, y, s=900, color=col, edgecolors="#0b2a35", linewidths=1.5, zorder=5)
        ax.text(x, y, str(n), color="white" if col != C_FRONT else "black",
                fontsize=11, fontweight="bold", ha="center", va="center", zorder=6)
        d = s["dist"][n]
        p = s["prev"][n]
        info = f"costo={'∞' if d == float('inf') else d}\nprev={'-' if p is None else p}"
        ax.text(x, y - 38, info, fontsize=7.5, ha="center", va="top", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaa", alpha=0.9))

    ax.set_title(f"Dijkstra {START} → {TARGET}   |   Paso {state['i']}/{len(STEPS) - 1}: {s['msg']}",
                 fontsize=13)
    ax.text(0.005, 0.005,
            "Azul: sin descubrir | Amarillo: descubierto (en cola) | Naranja: vértice actual / aristas del árbol | "
            "Verde: visitado | Rojo: ruta final",
            transform=ax.transAxes, fontsize=9)
    fig.canvas.draw_idle()


def go(delta):
    state["i"] = max(0, min(len(STEPS) - 1, state["i"] + delta))
    draw()


def reset(_=None):
    state["i"] = 0
    draw()


def last(_=None):
    state["i"] = len(STEPS) - 1
    draw()


b_prev = Button(plt.axes([0.25, 0.02, 0.12, 0.05]), "◀ Anterior")
b_next = Button(plt.axes([0.38, 0.02, 0.12, 0.05]), "Siguiente ▶")
b_rst = Button(plt.axes([0.51, 0.02, 0.12, 0.05]), "Reiniciar")
b_end = Button(plt.axes([0.64, 0.02, 0.12, 0.05]), "Ir al final")
b_prev.on_clicked(lambda e: go(-1))
b_next.on_clicked(lambda e: go(1))
b_rst.on_clicked(reset)
b_end.on_clicked(last)
fig.canvas.mpl_connect("key_press_event",
                       lambda e: go(1) if e.key == "right" else go(-1) if e.key == "left" else None)

draw()
plt.show()