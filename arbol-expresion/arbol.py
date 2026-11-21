"""
Árbol de expresión + gráfica + recorrido paso a paso del rango (PyQt6).

Instalación:  pip install PyQt6 sympy numpy matplotlib
Ejecución:    python arbol_y_grafica_qt.py

Reparación automática (casilla activada por defecto) de expresiones inválidas:
  - División entre 0  -> el denominador nunca vale 0: si |den| < ε se usa ±ε (ε = 1e-9).
  - Raíz par de un negativo (sqrt, x**(1/4), x**0.5) -> se usa el valor absoluto.
  - log(0) / log(negativo)  -> log(|arg|) con |arg| ≥ ε.
  - cot, csc, sec -> se protege el denominador (sen o cos) igual que arriba.
  - Raíz impar (x**(1/3)) -> se toma la raíz real, que sí admite negativos.

Pestañas:
  1. "Árbol y gráfica": árbol de expresión y gráfica completa.
  2. "Paso a paso": divide el rango en n puntos (paso h = (máx-mín)/(n-1)) y
     los evalúa uno a uno, mostrando la sustitución, la tabla de valores y la
     gráfica construyéndose punto por punto (con botones, slider y reproducción).
"""
from dataclasses import dataclass
import csv
import random
import sys
from collections.abc import Callable, Sequence

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QPushButton, QComboBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QSlider, QCheckBox, QFileDialog,
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
NOMBRES = {"Add": "+", "Mul": "×", "Pow": "^"}
EJEMPLOS = ["sin(x)/x + x^2", "tan(x)", "exp(-x^2)*cos(3x)", "sqrt(x) + log(x)",
            "x^3 - 2x + 1", "x^2 + y^2", "sin(x)*cos(y)"]


def guardar_tabla_csv(tabla, ruta):
    """Guarda los encabezados y celdas de un QTableWidget en formato CSV."""
    with open(ruta, "w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow([
            tabla.horizontalHeaderItem(col).text()
            for col in range(tabla.columnCount())
        ])
        for fila in range(tabla.rowCount()):
            escritor.writerow([
                tabla.item(fila, col).text()
                if tabla.item(fila, col) is not None else ""
                for col in range(tabla.columnCount())
            ])


# ================================================= Biblioteca de operadores
ConstructorOperador = Callable[[tuple[sp.Expr, ...]], sp.Expr]


@dataclass(frozen=True)
class Operador:
    """Define una operación generable y el número exacto de argumentos que recibe."""

    nombre: str
    aridad: int
    construir: ConstructorOperador
    peso: float = 1.0

    def aplicar(self, argumentos: Sequence[sp.Expr]) -> sp.Expr:
        argumentos = tuple(argumentos)
        if len(argumentos) != self.aridad:
            raise ValueError(
                f"{self.nombre} espera {self.aridad} argumentos; "
                f"recibió {len(argumentos)}."
            )
        resultado = self.construir(argumentos)
        if not isinstance(resultado, sp.Expr):
            resultado = sp.sympify(resultado)
        return resultado


class BibliotecaOperadores:
    """Registro extensible de operaciones agrupadas por aridad."""

    def __init__(self, operadores: Sequence[Operador] = ()):
        self._operadores: dict[int, list[Operador]] = {}
        for operador in operadores:
            self.registrar(operador)

    def registrar(self, operador: Operador) -> None:
        """Agrega una operación; se pueden registrar varias con la misma aridad."""
        if operador.aridad < 1:
            raise ValueError("La aridad debe ser un entero positivo.")
        if operador.peso <= 0:
            raise ValueError("El peso debe ser mayor que cero.")
        self._operadores.setdefault(operador.aridad, []).append(operador)

    def operadores(self, aridad: int | None = None) -> tuple[Operador, ...]:
        """Devuelve todas las operaciones, o solo las de una aridad."""
        if aridad is not None:
            return tuple(self._operadores.get(aridad, ()))
        return tuple(op for grupo in self._operadores.values() for op in grupo)

    def elegir(self, rng: random.Random) -> Operador:
        """Elige una operación ponderada, incluyendo aridades diferentes."""
        disponibles = self.operadores()
        if not disponibles:
            raise ValueError("La biblioteca no contiene operadores.")
        return rng.choices(disponibles, weights=[op.peso for op in disponibles], k=1)[0]

    @classmethod
    def predeterminada(cls) -> "BibliotecaOperadores":
        """Biblioteca base; puede copiarse y extenderse con registrar()."""
        unarias = (
            ("sin", sp.sin), ("cos", sp.cos), ("tan", sp.tan),
            ("exp", sp.exp), ("sqrt", sp.sqrt), ("log", sp.log),
            ("abs", sp.Abs),
        )
        binarias = (
            ("+", lambda a: a[0] + a[1]),
            ("-", lambda a: a[0] - a[1]),
            ("*", lambda a: a[0] * a[1]),
            ("/", lambda a: a[0] / a[1]),
            ("^", lambda a: a[0] ** a[1]),
            ("max", lambda a: sp.Max(a[0], a[1])),
            ("min", lambda a: sp.Min(a[0], a[1])),
        )
        return cls(
            [Operador(nombre, 1, lambda a, fn=fn: fn(a[0]))
             for nombre, fn in unarias]
            + [Operador(nombre, 2, fn) for nombre, fn in binarias]
        )


BIBLIOTECA_OPERADORES = BibliotecaOperadores.predeterminada()


def generar_expresion_aleatoria(
    profundidad: int = 3,
    variables: Sequence[str | sp.Symbol] = ("x",),
    biblioteca: BibliotecaOperadores | None = None,
    rng: random.Random | None = None,
    probabilidad_hoja: float = 0.25,
    rango_constantes: tuple[int, int] = (-5, 5),
) -> sp.Expr:
    """Genera una expresión SymPy usando operadores registrados de aridad variable.

    ``profundidad`` limita la altura del árbol. ``rng`` permite resultados
    reproducibles y ``biblioteca`` permite añadir operaciones sin tocar esta función.
    """
    if profundidad < 0:
        raise ValueError("La profundidad debe ser mayor o igual que cero.")
    if not variables:
        raise ValueError("Debe proporcionarse al menos una variable.")
    if not 0 <= probabilidad_hoja <= 1:
        raise ValueError("probabilidad_hoja debe estar entre 0 y 1.")
    minimo, maximo = rango_constantes
    if minimo > maximo:
        raise ValueError("El límite inferior de las constantes debe ser menor o igual.")

    generador = rng or random.Random()
    registro = biblioteca or BIBLIOTECA_OPERADORES
    simbolos = tuple(sp.Symbol(v) if isinstance(v, str) else v for v in variables)
    if not registro.operadores():
        raise ValueError("La biblioteca no contiene operadores.")

    def hoja() -> sp.Expr:
        if generador.random() < 0.5:
            return generador.choice(simbolos)
        return sp.Integer(generador.randint(minimo, maximo))

    def construir(restante: int) -> sp.Expr:
        if restante == 0 or generador.random() < probabilidad_hoja:
            return hoja()
        operador = registro.elegir(generador)
        argumentos = tuple(construir(restante - 1) for _ in range(operador.aridad))
        return operador.aplicar(argumentos)

    return construir(profundidad)


# ======================================================== Reparación
EPSILON = 1e-9                      # "muy cercano a cero, pero nunca cero"
protegido = sp.Function("protegido")


def protegido_num(v):
    """Versión numérica de protegido(): si |v| < ε devuelve ±ε (+ε si v == 0)."""
    v = np.asarray(v, dtype=float)
    return np.where(np.abs(v) < EPSILON, np.where(v < 0, -EPSILON, EPSILON), v)


def hacer_funcion(expr, simbolos):
    """lambdify que además entiende la función auxiliar protegido()."""
    return sp.lambdify(simbolos, expr, modules=[{"protegido": protegido_num}, "numpy"])


def reparar(expr):
    """Devuelve (expresión_reparada, lista_de_cambios_en_texto)."""
    cambios = []

    def nota(t):
        if t not in cambios:
            cambios.append(t)

    def pow_segura(base, exp):
        negativo = exp.could_extract_minus_sign()          # denominador
        if base.is_number:
            if base.is_zero and negativo:
                nota(f"División entre 0: se usa {EPSILON:g} en lugar de 0")
                base = sp.Float(EPSILON)
            return sp.Pow(base, exp)

        t = sp.sstr(base)
        raiz_par = ((exp.is_Rational and not exp.is_Integer and exp.q % 2 == 0)
                    or (exp.is_Float and float(exp) != int(float(exp))))
        if raiz_par:
            nota(f"Raíz de {t}: si es negativo se usa su valor absoluto |{t}|")
            base = sp.Abs(base)
        if negativo:
            nota(f"División entre {t}: si vale 0 se usa ±{EPSILON:g}, nunca 0")
            base = protegido(base)
        if exp.is_Rational and not exp.is_Integer and exp.q % 2 == 1:
            nota(f"Raíz impar de {t}: se toma la raíz real (admite negativos)")
            return sp.sign(base) ** abs(exp.p) * sp.Abs(base) ** exp
        return sp.Pow(base, exp)

    def rec(e):
        if not e.args:
            return e
        a = [rec(x) for x in e.args]
        if isinstance(e, sp.Pow):
            return pow_segura(a[0], a[1])
        if isinstance(e, sp.log) and len(a) == 1:
            arg = a[0]
            if arg.is_number:
                if arg.is_zero:
                    nota(f"log(0): se usa log({EPSILON:g})")
                    return sp.log(sp.Float(EPSILON))
                if arg.is_negative:
                    nota(f"log({arg}): se usa log({-arg}) (valor absoluto)")
                    return sp.log(-arg)
                return sp.log(arg)
            t = sp.sstr(arg)
            nota(f"log({t}): se usa log(|{t}|), con |{t}| ≥ {EPSILON:g}")
            return sp.log(protegido(sp.Abs(arg)))
        if isinstance(e, (sp.cot, sp.csc, sp.sec)) and len(a) == 1:
            t = a[0]
            den = sp.sin(t) if isinstance(e, (sp.cot, sp.csc)) else sp.cos(t)
            nota(f"{e.func.__name__}({sp.sstr(t)}): se protege el denominador "
                 f"{sp.sstr(den)} (nunca 0)")
            num = sp.cos(t) if isinstance(e, sp.cot) else sp.Integer(1)
            return num / protegido(den)
        return e.func(*a)

    return rec(expr), cambios


# ================================================================ Árbol
class Nodo:
    def __init__(self, etiqueta, hijos=None):
        self.etiqueta, self.hijos = etiqueta, hijos or []
        self.x = self.y = 0.0


def construir_arbol(expr):
    if not expr.args:
        return Nodo(str(expr))
    nombre = type(expr).__name__
    return Nodo(NOMBRES.get(nombre, nombre.lower()),
                [construir_arbol(a) for a in expr.args])


def posicionar(nodo, profundidad=0, contador=None):
    contador = contador if contador is not None else [0]
    nodo.y = -profundidad
    if not nodo.hijos:
        nodo.x = contador[0]
        contador[0] += 1
    else:
        for h in nodo.hijos:
            posicionar(h, profundidad + 1, contador)
        nodo.x = sum(h.x for h in nodo.hijos) / len(nodo.hijos)


def dibujar_arbol(raiz, ax, titulo="Árbol de expresión"):
    def aristas(n):
        for h in n.hijos:
            ax.plot([n.x, h.x], [n.y, h.y], color="gray", zorder=1)
            aristas(h)

    def nodos(n):
        ax.text(n.x, n.y, n.etiqueta, ha="center", va="center", zorder=2,
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35",
                          fc="#d8f0d8" if not n.hijos else "#cfe2ff", ec="black"))
        for h in n.hijos:
            nodos(h)

    aristas(raiz)
    nodos(raiz)
    ax.set_title(titulo)
    ax.axis("off")
    ax.margins(0.1)


# ============================================================ Evaluación
def a_real(valores, forma):
    """Pasa a float: los resultados complejos (no reales) se vuelven NaN."""
    v = np.broadcast_to(np.asarray(valores, dtype=complex), forma)
    return np.where(np.abs(v.imag) < 1e-9, v.real, np.nan).astype(float)


def cortar_asintotas(ys):
    """Reemplaza por NaN los valores gigantes (asíntotas) para no unirlos."""
    ys = ys.copy()
    ys[~np.isfinite(ys)] = np.nan
    finito = ys[np.isfinite(ys)]
    if finito.size:
        lim = 10 * (np.percentile(np.abs(finito), 95) + 1)
        ys[np.abs(ys) > lim] = np.nan
    return ys


def curva_densa(expr, s, a, b, n=4000):
    f = hacer_funcion(expr, [s])
    xs = np.linspace(a, b, n)
    with np.errstate(all="ignore"):
        ys = cortar_asintotas(a_real(f(xs), xs.shape))
    return xs, ys


def graficar(expr, simbolos, a, b, fig):
    """Gráfica completa (pestaña 1)."""
    if len(simbolos) == 1:
        s = simbolos[0]
        xs, ys = curva_densa(expr, s, a, b)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(xs, ys, lw=2)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3)
        ax.set_xlabel(str(s))
        ax.set_ylabel(f"f({s})")
        ax.set_title(f"f({s}) = {sp.sstr(expr)}")
    elif len(simbolos) == 2:
        s1, s2 = simbolos
        f = hacer_funcion(expr, [s1, s2])
        g = np.linspace(a, b, 120)
        X, Y = np.meshgrid(g, g)
        with np.errstate(all="ignore"):
            Z = cortar_asintotas(a_real(f(X, Y), X.shape).ravel()).reshape(X.shape)
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_xlabel(str(s1)); ax.set_ylabel(str(s2)); ax.set_zlabel("f")
        ax.set_title(f"f({s1},{s2}) = {sp.sstr(expr)}")
    else:
        raise ValueError("Solo se pueden graficar expresiones de 1 o 2 variables.")


# ======================================================== Paso a paso
class PasoAPaso:
    """Evalúa la expresión en n puntos del rango y dibuja hasta el paso k.

    1 variable : n puntos x_i = a + i*h,  h = (b-a)/(n-1)
    2 variables: malla n x n recorrida por filas (x fijo, y recorre el rango)
    """

    def __init__(self, expr, simbolos, a, b, n):
        self.expr, self.simbolos = expr, list(simbolos)
        self.a, self.b, self.n = a, b, n
        self.dim = len(self.simbolos)
        self.h = (b - a) / (n - 1)
        g = np.linspace(a, b, n)

        with np.errstate(all="ignore"):
            if self.dim == 1:
                f = hacer_funcion(expr, self.simbolos)
                ys = a_real(f(g), g.shape)
                self.pts = np.column_stack([g, ys])            # valores reales
                self.curva = curva_densa(expr, self.simbolos[0], a, b)
                datos = self.curva[1]
                if not np.isfinite(datos).any():
                    datos = cortar_asintotas(ys)
            else:
                f = hacer_funcion(expr, self.simbolos)
                X, Y = np.meshgrid(g, g, indexing="ij")
                Z = a_real(f(X, Y), X.shape).ravel()
                self.pts = np.column_stack([X.ravel(), Y.ravel(), Z])   # valores reales
                self.zplot = cortar_asintotas(Z)               # solo para dibujar
                datos = self.zplot
        self.N = len(self.pts)

        fin = datos[np.isfinite(datos)]
        lo, hi = (fin.min(), fin.max()) if fin.size else (-1.0, 1.0)
        if hi - lo < 1e-9:
            lo, hi = lo - 1, hi + 1
        pad = 0.06 * (hi - lo)
        self.lim = (lo - pad, hi + pad)

    # ---- texto -------------------------------------------------------
    def sustitucion(self, k):
        """Expresión con los valores del paso k puestos en lugar de las variables."""
        rep = {s: sp.Symbol(f"({v:.4g})")
               for s, v in zip(self.simbolos, self.pts[k, :-1])}
        return sp.sstr(self.expr.xreplace(rep))

    # ---- dibujo ------------------------------------------------------
    def dibujar(self, fig, k):
        vista = None
        if self.dim == 2 and fig.axes and hasattr(fig.axes[0], "elev"):
            vista = (fig.axes[0].elev, fig.axes[0].azim)   # conserva la rotación
        fig.clear()
        if self.dim == 1:
            self._dibujar_1d(fig, k)
        else:
            self._dibujar_2d(fig, k, vista)

    def _dibujar_1d(self, fig, k):
        ax = fig.add_subplot(111)
        xd, yd = self.curva
        ax.plot(xd, yd, color="lightgray", lw=1.5, label="curva completa")

        x, y = self.pts[:k + 1, 0], self.pts[:k + 1, 1]
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], "o-", color="tab:blue", ms=5, label="puntos calculados")

        # marcas de la partición del rango sobre el eje inferior
        ax.plot(self.pts[:, 0], np.full(self.N, self.lim[0]), "|", color="gray",
                ms=8, clip_on=False)

        xk, yk = self.pts[k, 0], self.pts[k, 1]
        if np.isfinite(yk):
            ydib = min(max(yk, self.lim[0]), self.lim[1])
            fuera = ydib != yk
            marca = "o" if not fuera else ("^" if yk > ydib else "v")
            ax.vlines(xk, 0, ydib, colors="tab:red", linestyles="--", lw=1)
            ax.plot([xk], [ydib], marca, color="tab:red", ms=10, zorder=5)
            texto = f"({xk:.4g}, {yk:.4g})" + ("  ¡fuera de escala!" if fuera else "")
            ax.annotate(texto, (xk, ydib), textcoords="offset points",
                        xytext=(8, -16 if yk > ydib else 10), color="tab:red",
                        fontweight="bold")
        else:
            ax.axvline(xk, color="tab:red", ls="--", lw=1)
            ax.text(xk, self.lim[0], " indefinido", color="tab:red", va="bottom")

        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlim(self.a - 0.02 * (self.b - self.a), self.b + 0.02 * (self.b - self.a))
        ax.set_ylim(*self.lim)
        ax.grid(alpha=0.3)
        ax.set_xlabel(str(self.simbolos[0]))
        ax.set_ylabel(f"f({self.simbolos[0]})")
        ax.set_title(f"Paso {k + 1} de {self.N}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

    def _dibujar_2d(self, fig, k, vista):
        n = self.n
        ax = fig.add_subplot(111, projection="3d")
        X = self.pts[:, 0].reshape(n, n)
        Y = self.pts[:, 1].reshape(n, n)
        Z = self.zplot.reshape(n, n)
        ax.plot_wireframe(X, Y, Z, color="lightgray", lw=0.4, alpha=0.7)

        sub = self.pts[:k + 1]
        zs = self.zplot[:k + 1]
        ok = np.isfinite(zs)
        ax.scatter(sub[ok, 0], sub[ok, 1], zs[ok], c=zs[ok], cmap="viridis",
                   vmin=self.lim[0], vmax=self.lim[1], s=18, depthshade=False)
        xk, yk, _ = self.pts[k]
        zk = self.zplot[k]
        if np.isfinite(zk):
            ax.scatter([xk], [yk], [zk], color="tab:red", s=90, depthshade=False)

        ax.set_xlim(self.a, self.b)
        ax.set_ylim(self.a, self.b)
        ax.set_zlim(*self.lim)
        ax.set_xlabel(str(self.simbolos[0]))
        ax.set_ylabel(str(self.simbolos[1]))
        ax.set_zlabel("f")
        ax.set_title(f"Paso {k + 1} de {self.N}")
        if vista:
            ax.view_init(*vista)
        fig.tight_layout()


# ================================================================== GUI
class Ventana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Árbol de expresión y gráfica")
        self.resize(1350, 780)

        self.expr = None          # expresión evaluada actual
        self.simbolos = []
        self.pasos = None         # PasoAPaso actual
        self.k = 0                # paso actual
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.siguiente_auto)

        # ---------- barra superior (compartida)
        self.entrada = QLineEdit("sin(x)/x + x^2")
        self.entrada.setPlaceholderText("Ej: sin(x)/x + x^2   (use x, o x e y para 3D)")
        self.ejemplos = QComboBox()
        self.ejemplos.addItem("Ejemplos…")
        self.ejemplos.addItems(EJEMPLOS)
        self.ejemplos.activated.connect(self.cargar_ejemplo)
        self.b_aleatoria = QPushButton("Generar aleatoria")
        self.b_aleatoria.clicked.connect(self.generar_aleatoria)
        self.min = self._spin(-10)
        self.max = self._spin(10)
        self.boton = QPushButton("Graficar")
        self.boton.setDefault(True)
        self.boton.clicked.connect(self.actualizar)
        self.entrada.returnPressed.connect(self.actualizar)
        self.chk_reparar = QCheckBox("Reparar expresión")
        self.chk_reparar.setChecked(True)
        self.chk_reparar.setToolTip(
            "÷0 → ±1e-9 (nunca 0) · raíz par de negativo → valor absoluto · log(|·|)")
        self.chk_reparar.toggled.connect(self.actualizar)

        fila = QHBoxLayout()
        fila.addWidget(QLabel("f ="))
        fila.addWidget(self.entrada, 1)
        fila.addWidget(self.ejemplos)
        fila.addWidget(self.b_aleatoria)
        fila.addWidget(QLabel("Mín:"))
        fila.addWidget(self.min)
        fila.addWidget(QLabel("Máx:"))
        fila.addWidget(self.max)
        fila.addWidget(self.chk_reparar)
        fila.addWidget(self.boton)

        # ---------- pestaña 1: árbol y gráfica
        self.fig = Figure(figsize=(12, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        tab1 = QWidget()
        l1 = QVBoxLayout(tab1)
        l1.addWidget(NavigationToolbar2QT(self.canvas, self))
        l1.addWidget(self.canvas, 1)

        # ---------- pestaña 2: paso a paso
        self.fig2 = Figure(figsize=(7, 6))
        self.canvas2 = FigureCanvasQTAgg(self.fig2)

        izq = QWidget()
        li = QVBoxLayout(izq)
        li.setContentsMargins(0, 0, 0, 0)
        li.addWidget(NavigationToolbar2QT(self.canvas2, self))
        li.addWidget(self.canvas2, 1)

        self.tabla = QTableWidget()
        self.tabla.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tabla.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tabla.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.cellClicked.connect(lambda fila, _col: self.ir_a(fila))

        self.b_exportar = QPushButton("Exportar tabla a CSV")
        self.b_exportar.setToolTip("Guarda todos los puntos de la tabla lateral en un archivo CSV")
        self.b_exportar.clicked.connect(self.exportar_csv)
        panel_tabla = QWidget()
        layout_tabla = QVBoxLayout(panel_tabla)
        layout_tabla.setContentsMargins(0, 0, 0, 0)
        layout_tabla.addWidget(self.b_exportar)
        layout_tabla.addWidget(self.tabla, 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(izq)
        split.addWidget(panel_tabla)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        self.spin_n = QSpinBox()
        self.spin_n.setRange(2, 200)
        self.spin_n.setValue(15)
        self.spin_n.setToolTip("Puntos por eje en los que se divide el rango")
        self.spin_n.valueChanged.connect(self.reconstruir_pasos)

        self.b_ini = QPushButton("⏮ Inicio")
        self.b_ant = QPushButton("◀ Anterior")
        self.b_sig = QPushButton("Siguiente ▶")
        self.b_fin = QPushButton("Final ⏭")
        self.b_play = QPushButton("▶ Reproducir")
        self.b_ini.clicked.connect(lambda: self.ir_a(0))
        self.b_ant.clicked.connect(lambda: self.ir_a(self.k - 1))
        self.b_sig.clicked.connect(lambda: self.ir_a(self.k + 1))
        self.b_fin.clicked.connect(lambda: self.ir_a(10 ** 9))
        self.b_play.clicked.connect(self.alternar_reproduccion)

        self.velocidad = QSpinBox()
        self.velocidad.setRange(50, 3000)
        self.velocidad.setSingleStep(50)
        self.velocidad.setValue(500)
        self.velocidad.setSuffix(" ms")
        self.velocidad.valueChanged.connect(
            lambda v: self.timer.isActive() and self.timer.setInterval(v))

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Puntos por eje (n):"))
        ctrl.addWidget(self.spin_n)
        ctrl.addSpacing(12)
        for b in (self.b_ini, self.b_ant, self.b_sig, self.b_fin, self.b_play):
            ctrl.addWidget(b)
        ctrl.addWidget(QLabel("Velocidad:"))
        ctrl.addWidget(self.velocidad)
        ctrl.addStretch(1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.ir_a)

        self.lbl_h = QLabel("")
        self.info = QLabel("")
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.info.setStyleSheet("font-family: monospace; font-size: 13px;")
        self.info.setWordWrap(True)

        tab2 = QWidget()
        l2 = QVBoxLayout(tab2)
        l2.addWidget(split, 1)
        l2.addLayout(ctrl)
        l2.addWidget(self.slider)
        l2.addWidget(self.lbl_h)
        l2.addWidget(self.info)

        # ---------- ensamblado
        self.tabs = QTabWidget()
        self.tabs.addTab(tab1, "Árbol y gráfica")
        self.tabs.addTab(tab2, "Paso a paso")

        self.lbl_rep = QLabel("")
        self.lbl_rep.setWordWrap(True)
        self.lbl_rep.setStyleSheet("color: #b9770e;")
        self.lbl_rep.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.estado = QLabel("")
        self.estado.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(fila)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(self.lbl_rep)
        lay.addWidget(self.estado)
        self.setCentralWidget(central)

        self.actualizar()

    # ---------------------------------------------------------- utilidades
    @staticmethod
    def _spin(valor):
        s = QDoubleSpinBox()
        s.setRange(-1e6, 1e6)
        s.setDecimals(3)
        s.setValue(valor)
        s.setFixedWidth(100)
        return s

    def cargar_ejemplo(self, idx):
        if idx > 0:
            self.entrada.setText(self.ejemplos.itemText(idx))
            self.actualizar()

    def generar_aleatoria(self):
        """Genera una expresión con la biblioteca registrada y la muestra."""
        try:
            expr = generar_expresion_aleatoria(
                profundidad=3,
                variables=("x", "y"),
                rng=random.Random(),
            )
            self.entrada.setText(sp.sstr(expr))
            self.actualizar()
        except Exception as e:
            self.mostrar_error(str(e) or type(e).__name__)

    def exportar_csv(self):
        """Exporta encabezados y filas de la tabla lateral usando formato CSV."""
        if self.tabla.rowCount() == 0 or self.tabla.columnCount() == 0:
            return self.mostrar_error("no hay puntos para exportar.")

        ruta, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar puntos a CSV",
            "puntos.csv",
            "Archivos CSV (*.csv);;Todos los archivos (*)",
        )
        if not ruta:
            return

        try:
            guardar_tabla_csv(self.tabla, ruta)
        except OSError as e:
            return self.mostrar_error(f"no se pudo guardar el CSV: {e}")

        self.estado.setStyleSheet("color: #27632a;")
        self.estado.setText(
            f"CSV exportado: {ruta} ({self.tabla.rowCount()} puntos)")

    def mostrar_error(self, msg):
        self.detener_reproduccion()
        self.lbl_rep.setText("")
        self.estado.setStyleSheet("color: #c0392b;")
        self.estado.setText(f"Error: {msg}")

    # ----------------------------------------------- pestaña 1 / entrada
    def actualizar(self):
        texto = self.entrada.text().strip()
        a, b = self.min.value(), self.max.value()
        if not texto:
            return self.mostrar_error("escribe una expresión.")
        if a >= b:
            return self.mostrar_error("el mínimo debe ser menor que el máximo.")
        try:
            # sin evaluar: conserva la estructura escrita (p. ej. 1/0 no se vuelve zoo)
            expr_arbol = parse_expr(texto, transformations=TRANSFORMS, evaluate=False)
            simbolos = sorted(expr_arbol.free_symbols, key=str) or [sp.Symbol("x")]
            if len(simbolos) > 2:
                raise ValueError("Solo se pueden graficar expresiones de 1 o 2 variables.")

            if self.chk_reparar.isChecked():
                expr_eval, cambios = reparar(expr_arbol)
            else:
                expr_eval, cambios = parse_expr(texto, transformations=TRANSFORMS), []
            expr_tree = expr_eval if cambios else expr_arbol

            raiz = construir_arbol(expr_tree)
            posicionar(raiz)
            self.fig.clear()
            dibujar_arbol(raiz, self.fig.add_subplot(1, 2, 1),
                          "Árbol de expresión (reparada)" if cambios
                          else "Árbol de expresión")
            graficar(expr_eval, simbolos, a, b, self.fig)
            self.fig.tight_layout()
            self.canvas.draw_idle()

            self.expr, self.simbolos = expr_eval, simbolos
            # límite de puntos según dimensión (en 2D son n x n puntos)
            self.spin_n.blockSignals(True)
            self.spin_n.setMaximum(200 if len(simbolos) == 1 else 40)
            self.spin_n.blockSignals(False)
            self.reconstruir_pasos()

            self.estado.setStyleSheet("color: #27632a;")
            self.estado.setText(f"Expresión usada: {sp.sstr(expr_eval)}")
            self.lbl_rep.setText(
                "Reparaciones aplicadas:\n" + "\n".join(f"  • {c}" for c in cambios)
                if cambios else "")
        except Exception as e:
            self.mostrar_error(str(e) or type(e).__name__)

    # ------------------------------------------------------ paso a paso
    def reconstruir_pasos(self):
        if self.expr is None:
            return
        self.detener_reproduccion()
        a, b, n = self.min.value(), self.max.value(), self.spin_n.value()
        try:
            self.pasos = PasoAPaso(self.expr, self.simbolos, a, b, n)
        except Exception as e:
            return self.mostrar_error(str(e) or type(e).__name__)
        p = self.pasos

        extra = "" if p.dim == 1 else " (igual en ambos ejes; se recorre la malla por filas)"
        self.lbl_h.setText(
            f"Rango [{a:g}, {b:g}] dividido en n = {n} puntos  →  "
            f"paso h = (máx − mín)/(n − 1) = ({b:g} − {a:g})/{n - 1} = {p.h:.6g}{extra}")

        # tabla
        cols = ["#"] + [str(s) for s in p.simbolos] + ["f"]
        self.tabla.clear()
        self.tabla.setColumnCount(len(cols))
        self.tabla.setRowCount(p.N)
        self.tabla.setHorizontalHeaderLabels(cols)
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i in range(p.N):
            textos = [str(i + 1)] + [f"{v:.6g}" if np.isfinite(v) else "indefinido"
                                     for v in p.pts[i]]
            for j, t in enumerate(textos):
                it = QTableWidgetItem(t)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tabla.setItem(i, j, it)

        self.slider.blockSignals(True)
        self.slider.setRange(0, p.N - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        self.k = 0
        self._colorear_filas(0, p.N - 1)
        self.ir_a(0)

    def _colorear_filas(self, desde, hasta):
        """Gris para los pasos aún no calculados; color normal para los hechos."""
        gris = QBrush(QColor("#9a9a9a"))
        for i in range(max(0, desde), min(self.tabla.rowCount() - 1, hasta) + 1):
            for j in range(self.tabla.columnCount()):
                it = self.tabla.item(i, j)
                if it is None:
                    continue
                if i <= self.k:
                    it.setData(Qt.ItemDataRole.ForegroundRole, None)
                else:
                    it.setForeground(gris)

    def ir_a(self, k):
        if self.pasos is None:
            return
        p = self.pasos
        k = max(0, min(int(k), p.N - 1))
        anterior = self.k
        self.k = k

        self.slider.blockSignals(True)
        self.slider.setValue(k)
        self.slider.blockSignals(False)

        p.dibujar(self.fig2, k)
        self.canvas2.draw_idle()

        self._colorear_filas(min(anterior, k), max(anterior, k))
        self.tabla.selectRow(k)
        self.tabla.scrollToItem(self.tabla.item(k, 0))

        vars_txt = ",  ".join(f"{s} = {v:.6g}" for s, v in zip(p.simbolos, p.pts[k, :-1]))
        nombre = ", ".join(str(s) for s in p.simbolos)
        val = p.pts[k, -1]
        res = f"{val:.6g}" if np.isfinite(val) else "indefinido (o no real)"
        self.info.setText(f"Paso {k + 1} de {p.N}:   {vars_txt}\n"
                          f"f({nombre}) = {p.sustitucion(k)} = {res}")

        self.b_ini.setEnabled(k > 0)
        self.b_ant.setEnabled(k > 0)
        self.b_sig.setEnabled(k < p.N - 1)
        self.b_fin.setEnabled(k < p.N - 1)

    # ---------------------------------------------------- reproducción
    def alternar_reproduccion(self):
        if self.timer.isActive():
            self.detener_reproduccion()
            return
        if self.pasos is None:
            return
        if self.k >= self.pasos.N - 1:      # si ya terminó, empieza de nuevo
            self.ir_a(0)
        self.timer.start(self.velocidad.value())
        self.b_play.setText("⏸ Pausa")

    def detener_reproduccion(self):
        self.timer.stop()
        self.b_play.setText("▶ Reproducir")

    def siguiente_auto(self):
        if self.pasos is None or self.k >= self.pasos.N - 1:
            self.detener_reproduccion()
        else:
            self.ir_a(self.k + 1)
            if self.k >= self.pasos.N - 1:
                self.detener_reproduccion()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Ventana()
    w.show()
    sys.exit(app.exec())