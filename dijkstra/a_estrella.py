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
# 1. ESTRUCTURA DEL GRAFO (la misma de dijkstra.py, SIN ponderaciones)
# ============================================================
EDGES = [
    (29, 7), (29, 21), (21, 11), (21, 24), (7, 11), (7, 14),
    (7, 6), (7, 1), (6, 15), (6, 14), (6, 9), (6, 8),
    (15, 8), (15, 1), (15, 0), (0, 44), (0, 1), (44, 3),
    (44, 5), (5, 3), (5, 20), (5, 69), (3, 2), (3, 69),
    (20, 69), (20, 25), (11, 14), (11, 10), (11, 24),
    (14, 9), (14, 10), (14, 4), (9, 10), (8, 1), (8, 12),
    (8, 4), (10, 1), (1, 2), (1, 23), (1, 12), (2, 69),
    (2, 23), (2, 17), (69, 25), (69, 17), (69, 70),
    (25, 70), (23, 16), (12, 16),          # la arista doble 12-16 ya no hace falta
    (12, 18), (12, 66), (12, 13), (17, 16), (17, 70),
    (16, 13), (10, 4), (10, 27), (27, 4), (27, 66),
    (4, 18), (18, 66), (66, 13),
]

# Posiciones (x, y) tomadas de la imagen (en píxeles)
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
POS = {n: (x, -y) for n, (x, y) in _PIX.items()}        # para dibujar

# Coordenadas de cuadrícula (cada 10 px = 1 unidad) para la distancia de Manhattan
GRID = {n: (round(x / 10), round(y / 10)) for n, (x, y) in _PIX.items()}

G = nx.Graph()
G.add_nodes_from(POS)
G.add_edges_from(EDGES)

START, TARGET = 14, 20


def manhattan(a, b):
    (x1, y1), (x2, y2) = GRID[a], GRID[b]
    return abs(x1 - x2) + abs(y1 - y2)


# Costo de moverse por una arista = distancia de Manhattan entre sus extremos
def costo(u, v):
    return manhattan(u, v)


# h(n): distancia de Manhattan hasta el destino (no depende del paso)
def calcular_h(target):
    return {n: manhattan(n, target) for n in G}

H = calcular_h(TARGET)

# ============================================================
# 2. A*: genera una "foto" del estado por cada paso
#    f(n) = g(n) + h(n)
#    g(n) = costo real acumulado desde START (suma de distancias Manhattan)
#    h(n) = distancia de Manhattan de n a TARGET
# ============================================================
def a_estrella_steps(graph, start, target):
    INF = float("inf")
    g = {n: INF for n in graph}
    f = {n: INF for n in graph}
    prev = {n: None for n in graph}
    prev_edge = {}                  # vertice -> (padre, vertice) usado para llegar
    closed, examined = set(), set()
    closed_order = []               # orden en que se van cerrando los vértices
    g[start] = 0
    f[start] = g[start] + H[start]
    # (f, h, n): desempata por menor h
    abierta = [(f[start], H[start], start)]
    steps = []

    def lista_abierta():
        # ordenada como la sacaría A*: menor f, luego menor h
        return sorted((fv, h, n) for fv, h, n in abierta
                      if n not in closed and fv == f[n])

    def snap(msg, current=None):
        steps.append(dict(g=dict(g), f=dict(f), prev=dict(prev),
                          closed=set(closed), closed_order=list(closed_order),
                          prev_edge=dict(prev_edge),
                          examined=set(examined), abierta=lista_abierta(),
                          current=current, msg=msg, path=None))

    snap(f"Inicio: g({start})=0, h({start})={H[start]}, f({start})={f[start]}")
    while abierta:
        fu, _, u = heapq.heappop(abierta)
        if u in closed or fu != f[u]:
            continue                       # entrada obsoleta de la cola
        closed.add(u)
        closed_order.append(u)
        if u == target:
            path, n = [], target
            while n is not None:
                path.append(n)
                n = prev[n]
            snap(f"Se alcanzó {target}: g={g[u]}. Algoritmo detenido.", u)
            steps[-1]["path"] = path[::-1]
            return steps
        for v in graph.neighbors(u):
            if v in closed:
                continue
            examined.add(frozenset((u, v)))
            ng = g[u] + costo(u, v)
            if ng < g[v]:
                g[v] = ng
                f[v] = ng + H[v]
                prev[v] = u
                prev_edge[v] = frozenset((u, v))
                heapq.heappush(abierta, (f[v], H[v], v))
        snap(f"Se expande {u} (f={f[u]} = g {g[u]} + h {H[u]}) y se actualizan vecinos", u)
    snap("No existe ruta al destino")
    return steps

STEPS = a_estrella_steps(G, START, TARGET)

# ============================================================
# 3. INTERFAZ INTERACTIVA (misma que dijkstra.py)
# ============================================================
C_BASE, C_VISIT, C_CUR, C_FRONT, C_PATH = "#1f6f8b", "#2e9e5b", "#e07b39", "#f2c14e", "#d62828"

fig, ax = plt.subplots(figsize=(16, 9))
plt.subplots_adjust(left=0.19, right=0.99, bottom=0.12)
# Panel izquierdo con las listas ABIERTA y CERRADA
ax_lists = fig.add_axes([0.005, 0.12, 0.18, 0.84])
state = {"i": 0}


def dibujar_listas(s):
    ax_lists.clear()
    ax_lists.set_xticks([])
    ax_lists.set_yticks([])
    for sp in ax_lists.spines.values():
        sp.set_edgecolor("#aaa")

    # ---- Lista ABIERTA (el primero es el siguiente en expandirse) ----
    lineas = ["LISTA ABIERTA", "(orden: menor f, luego h)",
              " #  nodo   g    h    f"]
    if s["abierta"]:
        for k, (fv, h, n) in enumerate(s["abierta"], 1):
            marca = "  ◀ sig." if k == 1 else ""
            lineas.append(f"{k:>2}  {n:>3}  {fmt(s['g'][n]):>4} {h:>4} {fv:>4}{marca}")
    else:
        lineas.append("   (vacía)")
    ax_lists.text(0.04, 0.98, "\n".join(lineas), transform=ax_lists.transAxes,
                  family="monospace", fontsize=8.5, va="top", linespacing=1.35)

    # ---- Lista CERRADA (en orden de expansión) ----
    lineas = ["LISTA CERRADA", "(orden de expansión)",
              " #  nodo   g    h    f"]
    if s["closed_order"]:
        for k, n in enumerate(s["closed_order"], 1):
            lineas.append(f"{k:>2}  {n:>3}  {fmt(s['g'][n]):>4} {H[n]:>4} {fmt(s['f'][n]):>4}")
    else:
        lineas.append("   (vacía)")
    ax_lists.text(0.04, 0.48, "\n".join(lineas), transform=ax_lists.transAxes,
                  family="monospace", fontsize=8.5, va="top", linespacing=1.35)
    ax_lists.axhline(0.5, color="#aaa", lw=1)


def fmt(x):
    return "∞" if x == float("inf") else str(x)


def draw():
    s = STEPS[state["i"]]
    ax.clear()
    ax.axis("off")
    path_edges = set()
    if s["path"]:
        path_edges = {s["prev_edge"][v] for v in s["path"][1:]}
    tree_edges = set(s["prev_edge"].values())

    # aristas (la etiqueta es el costo = distancia de Manhattan)
    for (u, v) in EDGES:
        key = frozenset((u, v))
        (x1, y1), (x2, y2) = POS[u], POS[v]
        if key in path_edges:
            col, lw, z = C_PATH, 4, 3
        elif key in tree_edges:
            col, lw, z = C_CUR, 3, 2
        elif key in s["examined"]:
            col, lw, z = C_FRONT, 1.8, 1
        else:
            col, lw, z = "#7a8a8f", 1.2, 0
        ax.plot([x1, x2], [y1, y2], color=col, lw=lw, zorder=z)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(costo(u, v)), fontsize=8,
                fontweight="bold", ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # vertices
    for n, (x, y) in POS.items():
        if n in (s["path"] or []):
            col = C_PATH
        elif n == s["current"]:
            col = C_CUR
        elif n in s["closed"]:
            col = C_VISIT
        elif s["g"][n] < float("inf"):
            col = C_FRONT
        else:
            col = C_BASE
        ax.scatter(x, y, s=900, color=col, edgecolors="#0b2a35", linewidths=1.5, zorder=5)
        ax.text(x, y, str(n), color="white" if col != C_FRONT else "black",
                fontsize=11, fontweight="bold", ha="center", va="center", zorder=6)
        p = s["prev"][n]
        info = (f"g={fmt(s['g'][n])}  h={H[n]}\n"
                f"f={fmt(s['f'][n])}\n"
                f"prev={'-' if p is None else p}")
        ax.text(x, y - 38, info, fontsize=7.5, ha="center", va="top", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaa", alpha=0.9))

    ax.set_title(f"A* {START} → {TARGET}   |   Paso {state['i']}/{len(STEPS) - 1}: {s['msg']}",
                 fontsize=13)
    dibujar_listas(s)
    ax.text(0.005, 0.005,
            "Azul: sin descubrir | Amarillo: en lista abierta | Naranja: vértice actual / aristas del árbol | "
            "Verde: cerrado | Rojo: ruta final   —   f=g+h, h=Manhattan al destino, aristas=Manhattan",
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