#!/usr/bin/env python3
"""
Algoritmos Evolutivos: un enfoque práctico - Solución Proyecto 1 (traducción C++ -> Python)

Algoritmo genético con codificación binaria, selección por ruleta con elitismo,
cruce en un punto y mutación por bit.
El AG usa solo la librería estándar; la interfaz gráfica requiere PySide6 y matplotlib:
    pip install PySide6 matplotlib

=======================================================================
INTERFAZ GRÁFICA (paso a paso)
=======================================================================
    python proyecto1_ag.py              # sin argumentos abre la interfaz
    python proyecto1_ag.py --gui        # equivalente

En la interfaz eliges el problema y los parámetros, pulsas "Inicializar" y avanzas
con los botones:  Inicio | Anterior | Siguiente | Reproducir/Pausa | +10 | Final
(o arrastras el slider de generaciones). Arriba se ve la función con la población
de la generación actual; abajo, la convergencia (mejor / media / peor).
Cualquier función registrada con @problema aparece automáticamente en la lista.

=======================================================================
CÓMO AGREGAR UNA NUEVA FUNCIÓN A OPTIMIZAR  (3 pasos, sin tocar el AG)
=======================================================================
1. Ve a la sección "FUNCIONES A OPTIMIZAR" de este archivo.
2. Escribe una función que reciba la lista de fenotipos `x = [x1, ..., xn]`
   y devuelva un número (el valor de la función).
3. Decórala con @problema(...) indicando:
       nombre       -> cómo la invocarás desde la línea de comandos / interfaz
       minimizar    -> True si se busca el mínimo, False si se busca el máximo
       x_min, x_max -> intervalo de búsqueda recomendado (de cada variable)
       num_var      -> nº de variables (int fijo) o None si el usuario lo elige
                       con --num-var (o en la interfaz)
       descripcion  -> texto corto (aparece en --listar y en la interfaz)

   Ejemplo (esfera, mínimo 0 en el origen):

       @problema("esfera", minimizar=True, x_min=-5.12, x_max=5.12,
                 num_var=None, descripcion="f(x) = sum(x_i^2)")
       def f_esfera(x):
           return sum(xi ** 2 for xi in x)

   Y ya se puede ejecutar:
       python proyecto1_ag.py esfera auto auto 0.001 300 500 0.6 0.01 --num-var 5

Comandos útiles:
    python proyecto1_ag.py --listar            # muestra los problemas registrados
    python proyecto1_ag.py maximizar 0.0 32.0 0.0001 500 1000 0.4 0.001
    python proyecto1_ag.py minimizar -250 250 0.0001 500 1000 0.4 0.001
    python proyecto1_ag.py multi 0 100 0.0001 500 1000 0.4 0.001 --num-var 3
    (en x_min / x_max se puede escribir 'auto' para usar los valores del registro)

Parámetros posicionales: problema x_min x_max tol tam_pob num_gen prob_cruce prob_mut
"""
import argparse
import math
import os
import random
import sys
from array import array
from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# REGISTRO DE PROBLEMAS (no hace falta modificarlo)
# ----------------------------------------------------------------------
@dataclass
class Problema:
    nombre: str
    funcion: object          # callable(list[float]) -> float
    minimizar: bool
    x_min: float
    x_max: float
    num_var: object          # int fijo, o None si lo elige el usuario
    descripcion: str = ""


PROBLEMAS = {}


def problema(nombre, minimizar, x_min, x_max, num_var=1, descripcion=""):
    """Decorador que registra una función como problema seleccionable."""
    def registrar(funcion):
        if nombre in PROBLEMAS:
            raise ValueError(f"Ya existe un problema llamado '{nombre}'")
        PROBLEMAS[nombre] = Problema(nombre, funcion, minimizar, x_min, x_max,
                                     num_var, descripcion)
        return funcion
    return registrar


# ======================================================================
# FUNCIONES A OPTIMIZAR  -> AGREGA AQUÍ LAS NUEVAS
# Cada función recibe x = [x1, ..., xn] y devuelve un float.
# ======================================================================
@problema("maximizar", minimizar=False, x_min=0.0, x_max=32.0, num_var=1,
          descripcion="f(x) = 20 + e - 20e^(-0.2|x|) - e^(cos 2πx); máx 22.3136 en 31.5005")
def f_maximizar(x):
    x = x[0]
    return (20 + math.exp(1) - 20 * math.exp(-0.2 * abs(x))
            - math.exp(math.cos(2 * math.pi * x)))


@problema("minimizar", minimizar=True, x_min=-250.0, x_max=250.0, num_var=1,
          descripcion="f(x) = -|x sen(sqrt|x|)|; mín -201.843 en 203.814")
def f_minimizar(x):
    x = x[0]
    return -abs(x * math.sin(math.sqrt(abs(x))))


@problema("multi", minimizar=True, x_min=0.0, x_max=100.0, num_var=None,
          descripcion="f(x_i) = sum(-x_i sen(sqrt|x_i|)); mín -n*63.63498 en x_i=65.54785")
def f_multivariable(x):
    return sum(-xi * math.sin(math.sqrt(abs(xi))) for xi in x)


# ======================================================================
# FIN DE LA SECCIÓN DE FUNCIONES
# ======================================================================


# ----------------------------------------------------------------------
# Representación del individuo (TIndividuo)
# ----------------------------------------------------------------------
@dataclass
class Individuo:
    genes: list                                   # una lista de bits por variable
    x: list = field(default_factory=list)         # fenotipos
    aptitud_bruta: float = 0.0                    # valor de la función
    aptitud: float = 0.0                          # aptitud transformada (para maximizar)
    puntuacion: float = 0.0                       # aptitud / suma de aptitudes
    punt_acu: float = 0.0                         # puntuación acumulada (ruleta)
    elite: bool = False

    def copia(self):
        return Individuo([g[:] for g in self.genes], self.x[:], self.aptitud_bruta,
                         self.aptitud, self.puntuacion, self.punt_acu, self.elite)


# ----------------------------------------------------------------------
# Decodificación
# ----------------------------------------------------------------------
def bin_dec(genes):
    """Binario -> decimal (el gen 0 es el bit menos significativo)."""
    d, pot = 0, 1
    for g in genes:
        d += pot * g
        pot *= 2
    return d


def decod(genes, lcrom, x_min, x_max):
    x = bin_dec(genes) / (2 ** lcrom - 1)
    return x_min + (x_max - x_min) * x


# ----------------------------------------------------------------------
# Adaptación
# ----------------------------------------------------------------------
class AG:
    def __init__(self, funcion, minimizar, x_min, x_max, tol, tam_pob, num_gen,
                 prob_cruce, prob_mut, num_var=1):
        self.funcion = funcion
        self.minimizar = minimizar
        self.num_var = num_var
        self.x_min = [x_min] * num_var
        self.x_max = [x_max] * num_var
        self.tam_pob = tam_pob
        self.num_gen = num_gen
        self.prob_cruce = prob_cruce
        self.prob_mut = prob_mut
        # longitud de cada cromosoma
        self.lcrom = [math.ceil(math.log2(1 + (x_max - x_min) / tol))
                      for _ in range(num_var)]
        self.tam_elite = tam_pob * 2 // 100
        self.pob = []
        self.pos_mejor = 0
        self.generacion = 0          # generaciones ya calculadas

    # -- función: adaptacion -------------------------------------------
    def adaptacion(self, ind):
        ind.x = [decod(ind.genes[k], self.lcrom[k], self.x_min[k], self.x_max[k])
                 for k in range(self.num_var)]
        return self.funcion(ind.x)

    # -- función: genera_indiv -----------------------------------------
    def genera_indiv(self):
        genes = [[0 if random.random() < 0.5 else 1 for _ in range(self.lcrom[k])]
                 for k in range(self.num_var)]
        ind = Individuo(genes)
        ind.aptitud_bruta = self.adaptacion(ind)
        return ind

    # -- función: poblacion_inicial ------------------------------------
    def poblacion_inicial(self):
        return [self.genera_indiv() for _ in range(self.tam_pob)]

    # -- función: revisar_adap_mini ------------------------------------
    def transformar_aptitud(self):
        """Para maximizar: aptitud = aptitud_bruta.
        Para minimizar: aptitud = cmax - aptitud_bruta (convierte min en max)."""
        if not self.minimizar:
            for ind in self.pob:
                ind.aptitud = ind.aptitud_bruta
            return
        cmax = max(ind.aptitud_bruta for ind in self.pob)
        cmax = cmax + abs(cmax) / 100        # margen del 1 % para que ninguna aptitud sea 0
        for ind in self.pob:
            ind.aptitud = cmax - ind.aptitud_bruta

    # -- función: evaluacion -------------------------------------------
    def evaluacion(self):
        suma = sum(ind.aptitud for ind in self.pob)
        mejor = 0
        for i, ind in enumerate(self.pob):
            if ind.aptitud > self.pob[mejor].aptitud:
                mejor = i
        self.pos_mejor = mejor
        acu = 0.0
        for ind in self.pob:
            ind.puntuacion = ind.aptitud / suma if suma > 0 else 1 / len(self.pob)
            acu += ind.puntuacion
            ind.punt_acu = acu

    # -- función: seleccion (ruleta + elitismo) ------------------------
    def seleccion(self):
        pob, n = self.pob, self.tam_pob
        # élite: los TAM_ELITE mejores individuos
        idx_elite = sorted(range(n), key=lambda i: pob[i].aptitud, reverse=True)
        idx_elite = set(idx_elite[:self.tam_elite])

        sel_super = []
        for i in range(n):
            if i in idx_elite:
                sel_super.append(i)
            else:
                prob = random.random()
                pos = 0
                while pos < n and prob > pob[pos].punt_acu:
                    pos += 1
                sel_super.append(pos if pos < n else n - 1)

        # población intermedia
        nueva = [pob[i].copia() for i in sel_super]
        for i, ind in enumerate(nueva):
            ind.elite = i in idx_elite       # solo las posiciones originales de la élite
        self.pob = nueva

    # -- función: cruce ------------------------------------------------
    def cruce(self, padre1, padre2, punto_cruce):
        hijo1 = Individuo([]); hijo2 = Individuo([])
        for k in range(self.num_var):
            p = punto_cruce[k]
            hijo1.genes.append(padre1.genes[k][:p] + padre2.genes[k][p:])
            hijo2.genes.append(padre2.genes[k][:p] + padre1.genes[k][p:])
        hijo1.aptitud_bruta = self.adaptacion(hijo1)
        hijo2.aptitud_bruta = self.adaptacion(hijo2)
        return hijo1, hijo2

    # -- función: reproduccion -----------------------------------------
    def reproduccion(self):
        pob = self.pob
        sel_cruce = [i for i in range(self.tam_pob) if random.random() < self.prob_cruce]
        if len(sel_cruce) % 2 == 1:          # número par de seleccionados
            sel_cruce.pop()

        # punto de cruce al azar (uno por variable), entre 1 y lcrom
        punto_cruce = [random.randint(1, l) for l in self.lcrom]

        for i in range(0, len(sel_cruce), 2):
            a, b = sel_cruce[i], sel_cruce[i + 1]
            hijo1, hijo2 = self.cruce(pob[a], pob[b], punto_cruce)
            # los hijos sustituyen a sus progenitores respetando la élite
            if not pob[a].elite:
                pob[a] = hijo1
            if not pob[b].elite:
                pob[b] = hijo2

    # -- función: mutacion ---------------------------------------------
    def mutacion(self):
        for ind in self.pob:
            if ind.elite:
                continue
            mutado = False
            for k in range(self.num_var):
                for j in range(self.lcrom[k]):
                    if random.random() < self.prob_mut:
                        ind.genes[k][j] = 1 - ind.genes[k][j]
                        mutado = True
            if mutado:
                ind.aptitud_bruta = self.adaptacion(ind)

    # -- ejecución por pasos (la usa la interfaz) ------------------------
    def inicializar(self):
        """Crea y evalúa la población de la generación 0."""
        self.pob = self.poblacion_inicial()
        self.transformar_aptitud()
        self.evaluacion()
        self.generacion = 0

    def paso(self):
        """Avanza UNA generación: selección, reproducción y mutación."""
        self.seleccion()
        self.reproduccion()
        self.mutacion()
        self.transformar_aptitud()
        self.evaluacion()
        self.generacion += 1

    # -- bucle principal -----------------------------------------------
    def ejecutar(self, verbose=False):
        self.inicializar()
        for gen in range(self.num_gen):
            self.paso()
            if verbose and gen % 100 == 0:
                print(f"gen {gen:5d}  mejor = {self.pob[self.pos_mejor].aptitud_bruta:.6f}")
        return self.pob[self.pos_mejor]


# ----------------------------------------------------------------------
# Interfaz gráfica (PySide6 + matplotlib)
# ----------------------------------------------------------------------
def lanzar_gui():
    os.environ.setdefault("QT_API", "pyside6")
    try:
        from PySide6 import QtCore, QtWidgets
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg,
                                                       NavigationToolbar2QT)
        from matplotlib.figure import Figure
    except ImportError as e:
        sys.exit(f"Falta una dependencia ({e}). Instala con: pip install PySide6 matplotlib")

    class Ventana(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Algoritmo genético - visualizador paso a paso")
            self.resize(1250, 800)
            self.ag = None
            self.hist = []                    # una "foto" (xs, fs) por generación
            self.est_mejor, self.est_media, self.est_peor = [], [], []
            self.idx = 0                      # generación mostrada
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.siguiente)
            self._construir_ui()
            self._cambio_problema()
            self._actualizar_botones()

        # ------------------------------------------------------------
        def _construir_ui(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            raiz = QtWidgets.QHBoxLayout(central)

            # ---- panel izquierdo: parámetros ----
            izq = QtWidgets.QVBoxLayout()
            raiz.addLayout(izq, 0)
            caja = QtWidgets.QGroupBox("Parámetros")
            form = QtWidgets.QFormLayout(caja)
            izq.addWidget(caja)

            self.cb_prob = QtWidgets.QComboBox()
            self.cb_prob.addItems(list(PROBLEMAS.keys()))
            self.cb_prob.currentIndexChanged.connect(self._cambio_problema)
            form.addRow("Problema", self.cb_prob)

            self.lbl_desc = QtWidgets.QLabel()
            self.lbl_desc.setWordWrap(True)
            self.lbl_desc.setMaximumWidth(280)
            form.addRow(self.lbl_desc)

            def dspin(lo, hi, dec, val, step=1.0):
                s = QtWidgets.QDoubleSpinBox()
                s.setRange(lo, hi); s.setDecimals(dec); s.setValue(val); s.setSingleStep(step)
                return s

            self.sp_xmin = dspin(-1e6, 1e6, 4, 0.0)
            self.sp_xmax = dspin(-1e6, 1e6, 4, 1.0)
            self.sp_tol = dspin(1e-9, 10, 6, 0.001, 0.0005)
            self.sp_pob = QtWidgets.QSpinBox(); self.sp_pob.setRange(4, 5000); self.sp_pob.setValue(100)
            self.sp_gen = QtWidgets.QSpinBox(); self.sp_gen.setRange(1, 100000); self.sp_gen.setValue(200)
            self.sp_cruce = dspin(0, 1, 3, 0.6, 0.05)
            self.sp_mut = dspin(0, 1, 4, 0.01, 0.001)
            self.sp_nvar = QtWidgets.QSpinBox(); self.sp_nvar.setRange(1, 50); self.sp_nvar.setValue(3)
            self.ed_semilla = QtWidgets.QLineEdit()
            self.ed_semilla.setPlaceholderText("vacío = aleatoria")
            for et, w in (("x mín", self.sp_xmin), ("x máx", self.sp_xmax),
                          ("Tolerancia", self.sp_tol), ("Tamaño población", self.sp_pob),
                          ("Nº generaciones", self.sp_gen), ("Prob. cruce", self.sp_cruce),
                          ("Prob. mutación", self.sp_mut), ("Nº variables", self.sp_nvar),
                          ("Semilla", self.ed_semilla)):
                form.addRow(et, w)

            self.btn_init = QtWidgets.QPushButton("Inicializar / Reiniciar")
            self.btn_init.clicked.connect(self.inicializar)
            izq.addWidget(self.btn_init)

            self.cb_var = QtWidgets.QComboBox()
            self.cb_var.currentIndexChanged.connect(self._dibujar_si_hay)
            f2 = QtWidgets.QFormLayout()
            f2.addRow("Variable a graficar", self.cb_var)
            izq.addLayout(f2)
            nota = QtWidgets.QLabel("Con varias variables, las demás se fijan en los valores "
                                    "del mejor individuo de la generación mostrada.")
            nota.setWordWrap(True); nota.setMaximumWidth(280)
            izq.addWidget(nota)

            self.lbl_info = QtWidgets.QLabel("Pulsa «Inicializar» para empezar.")
            self.lbl_info.setWordWrap(True)
            self.lbl_info.setMaximumWidth(280)
            self.lbl_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            izq.addWidget(self.lbl_info)
            izq.addStretch(1)

            # ---- panel derecho: gráficas + controles ----
            der = QtWidgets.QVBoxLayout()
            raiz.addLayout(der, 1)
            self.fig = Figure(constrained_layout=True)
            self.ax1 = self.fig.add_subplot(2, 1, 1)
            self.ax2 = self.fig.add_subplot(2, 1, 2)
            self.canvas = FigureCanvasQTAgg(self.fig)
            der.addWidget(NavigationToolbar2QT(self.canvas, self))
            der.addWidget(self.canvas, 1)

            fila = QtWidgets.QHBoxLayout()
            der.addLayout(fila)
            self.btn_inicio = QtWidgets.QPushButton("⏮ Inicio")
            self.btn_ant = QtWidgets.QPushButton("◀ Anterior")
            self.btn_sig = QtWidgets.QPushButton("Siguiente ▶")
            self.btn_play = QtWidgets.QPushButton("▶ Reproducir")
            self.btn_10 = QtWidgets.QPushButton("+10")
            self.btn_fin = QtWidgets.QPushButton("Final ⏭")
            self.btn_inicio.clicked.connect(lambda: self.ir_a(0))
            self.btn_ant.clicked.connect(self.anterior)
            self.btn_sig.clicked.connect(self.siguiente)
            self.btn_play.clicked.connect(self.alternar_reproduccion)
            self.btn_10.clicked.connect(lambda: self.avanzar(10))
            self.btn_fin.clicked.connect(self.ir_al_final)
            for b in (self.btn_inicio, self.btn_ant, self.btn_sig, self.btn_play,
                      self.btn_10, self.btn_fin):
                fila.addWidget(b)
            fila.addStretch(1)
            fila.addWidget(QtWidgets.QLabel("Pausa (ms):"))
            self.sp_vel = QtWidgets.QSpinBox()
            self.sp_vel.setRange(10, 3000); self.sp_vel.setValue(150); self.sp_vel.setSingleStep(50)
            self.sp_vel.valueChanged.connect(lambda v: self.timer.setInterval(v))
            fila.addWidget(self.sp_vel)

            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.valueChanged.connect(self._slider_movido)
            der.addWidget(self.slider)

            self._botones = (self.btn_inicio, self.btn_ant, self.btn_sig, self.btn_play,
                             self.btn_10, self.btn_fin, self.slider)

        # ------------------------------------------------------------
        def _cambio_problema(self):
            p = PROBLEMAS[self.cb_prob.currentText()]
            self.sp_xmin.setValue(p.x_min)
            self.sp_xmax.setValue(p.x_max)
            if p.num_var is None:
                self.sp_nvar.setEnabled(True)
            else:
                self.sp_nvar.setValue(p.num_var)
                self.sp_nvar.setEnabled(False)
            obj = "minimizar" if p.minimizar else "maximizar"
            self.lbl_desc.setText(f"[{obj}] {p.descripcion}")

        # ------------------------------------------------------------
        def inicializar(self):
            self.timer.stop()
            self.btn_play.setText("▶ Reproducir")
            p = PROBLEMAS[self.cb_prob.currentText()]
            x_min, x_max = self.sp_xmin.value(), self.sp_xmax.value()
            if x_min >= x_max:
                QtWidgets.QMessageBox.warning(self, "Parámetros", "x mín debe ser menor que x máx.")
                return
            txt = self.ed_semilla.text().strip()
            if txt:
                try:
                    random.seed(int(txt))
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, "Parámetros", "La semilla debe ser un entero.")
                    return
            else:
                random.seed()
            nv = self.sp_nvar.value() if p.num_var is None else p.num_var
            self.ag = AG(p.funcion, p.minimizar, x_min, x_max, self.sp_tol.value(),
                         self.sp_pob.value(), self.sp_gen.value(), self.sp_cruce.value(),
                         self.sp_mut.value(), nv)
            self.ag.inicializar()
            self.hist, self.est_mejor, self.est_media, self.est_peor = [], [], [], []
            self._registrar()
            self.idx = 0
            self.cb_var.blockSignals(True)
            self.cb_var.clear()
            self.cb_var.addItems([f"x{i + 1}" for i in range(nv)])
            self.cb_var.blockSignals(False)
            self._refrescar()

        # ------------------------------------------------------------
        def _registrar(self):
            """Guarda la foto de la generación actual del AG."""
            xs, fs = array('d'), array('d')
            for ind in self.ag.pob:
                xs.extend(ind.x)
                fs.append(ind.aptitud_bruta)
            self.hist.append((xs, fs))
            self.est_mejor.append(min(fs) if self.ag.minimizar else max(fs))
            self.est_peor.append(max(fs) if self.ag.minimizar else min(fs))
            self.est_media.append(sum(fs) / len(fs))

        def _calcular_una(self):
            self.ag.paso()
            self._registrar()

        # ------------------------------------------------------------
        def siguiente(self):
            if self.ag is None:
                return
            if self.idx < len(self.hist) - 1:
                self.idx += 1
            elif self.ag.generacion < self.ag.num_gen:
                self._calcular_una()
                self.idx += 1
            else:
                self.timer.stop()
                self.btn_play.setText("▶ Reproducir")
            self._refrescar()

        def anterior(self):
            if self.ag is not None and self.idx > 0:
                self.idx -= 1
                self._refrescar()

        def ir_a(self, i):
            if self.ag is not None:
                self.idx = max(0, min(i, len(self.hist) - 1))
                self._refrescar()

        def avanzar(self, n):
            if self.ag is None:
                return
            destino = min(self.idx + n, self.ag.num_gen)
            self._calcular_hasta(destino)
            self.idx = destino
            self._refrescar()

        def ir_al_final(self):
            if self.ag is None:
                return
            self._calcular_hasta(self.ag.num_gen)
            self.idx = self.ag.num_gen
            self._refrescar()

        def _calcular_hasta(self, destino):
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                while len(self.hist) - 1 < destino:
                    self._calcular_una()
                    if self.ag.generacion % 25 == 0:
                        QtWidgets.QApplication.processEvents()
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        def alternar_reproduccion(self):
            if self.ag is None:
                return
            if self.timer.isActive():
                self.timer.stop()
                self.btn_play.setText("▶ Reproducir")
            else:
                if self.idx >= self.ag.num_gen:
                    self.idx = 0
                self.timer.start(self.sp_vel.value())
                self.btn_play.setText("⏸ Pausa")

        def _slider_movido(self, v):
            if self.ag is not None and v != self.idx:
                self.idx = v
                self._dibujar()
                self._actualizar_info()

        # ------------------------------------------------------------
        def _actualizar_botones(self):
            activo = self.ag is not None
            for b in self._botones:
                b.setEnabled(activo)
            if activo:
                self.btn_ant.setEnabled(self.idx > 0)
                self.btn_inicio.setEnabled(self.idx > 0)
                fin = self.idx >= self.ag.num_gen
                self.btn_sig.setEnabled(not fin)
                self.btn_10.setEnabled(not fin)
                self.btn_fin.setEnabled(not fin)

        def _refrescar(self):
            self._actualizar_botones()
            self.slider.blockSignals(True)
            self.slider.setRange(0, len(self.hist) - 1)
            self.slider.setValue(self.idx)
            self.slider.blockSignals(False)
            self._dibujar()
            self._actualizar_info()

        def _dibujar_si_hay(self):
            if self.ag is not None:
                self._dibujar()

        def _actualizar_info(self):
            ag, i = self.ag, self.idx
            xs, fs = self.hist[i]
            nv = ag.num_var
            mi = self._mejor_indice(fs)
            vec = ", ".join(f"{xs[mi * nv + j]:.5f}" for j in range(nv))
            self.lbl_info.setText(
                f"Generación {i} / {ag.num_gen}\n"
                f"Mejor f = {fs[mi]:.5f}\nMedia f = {self.est_media[i]:.5f}\n"
                f"Peor f = {self.est_peor[i]:.5f}\nMejor x = [{vec}]\n"
                f"Bits por variable: {ag.lcrom[0]}")

        def _mejor_indice(self, fs):
            f = min if self.ag.minimizar else max
            return f(range(len(fs)), key=fs.__getitem__)

        # ------------------------------------------------------------
        def _dibujar(self):
            ag, i = self.ag, self.idx
            xs, fs = self.hist[i]
            nv, n = ag.num_var, ag.tam_pob
            k = max(0, self.cb_var.currentIndex())
            mi = self._mejor_indice(fs)
            base = [xs[mi * nv + j] for j in range(nv)]

            # --- 1) función + población ---
            ax = self.ax1
            ax.clear()
            lo, hi, M = ag.x_min[0], ag.x_max[0], 600
            ts = [lo + (hi - lo) * j / (M - 1) for j in range(M)]
            ys = []
            for t in ts:
                b = base[:]
                b[k] = t
                try:
                    ys.append(ag.funcion(b))
                except (ValueError, ZeroDivisionError, OverflowError):
                    ys.append(float("nan"))
            ax.plot(ts, ys, color="tab:gray", lw=1.2, label="f(x)")
            px = [xs[j * nv + k] for j in range(n)]
            ax.scatter(px, list(fs), s=14, alpha=0.5, color="tab:blue", label="población")
            ax.scatter([px[mi]], [fs[mi]], marker="*", s=240, color="tab:red",
                       zorder=5, label="mejor")
            titulo = f"Generación {i}"
            if nv > 1:
                titulo += f"  (variable x{k + 1}, resto fijo en el mejor)"
            ax.set_title(titulo)
            ax.set_xlabel(f"x{k + 1}" if nv > 1 else "x")
            ax.set_ylabel("f")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            # --- 2) convergencia ---
            ax = self.ax2
            ax.clear()
            g = range(i + 1)
            ax.plot(g, self.est_mejor[:i + 1], color="tab:red", label="mejor")
            ax.plot(g, self.est_media[:i + 1], color="tab:blue", label="media")
            ax.plot(g, self.est_peor[:i + 1], color="tab:green", alpha=0.6, label="peor")
            ax.axvline(i, color="k", ls="--", lw=0.8)
            ax.set_xlim(0, max(1, ag.num_gen))
            ax.set_title("Convergencia")
            ax.set_xlabel("generación")
            ax.set_ylabel("f (valor de la función)")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            self.canvas.draw_idle()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    ventana = Ventana()
    ventana.show()
    sys.exit(app.exec())


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def listar():
    print("Problemas registrados:")
    for p in PROBLEMAS.values():
        obj = "minimizar" if p.minimizar else "maximizar"
        nv = "n (--num-var)" if p.num_var is None else p.num_var
        print(f"  {p.nombre:<12} [{obj}] x in [{p.x_min}, {p.x_max}]  variables: {nv}")
        if p.descripcion:
            print(f"  {'':<12} {p.descripcion}")


def main():
    if "--listar" in sys.argv:
        listar()
        return
    if "--gui" in sys.argv or len(sys.argv) == 1:
        lanzar_gui()
        return

    ap = argparse.ArgumentParser(description="Proyecto 1 - Algoritmo genético binario")
    ap.add_argument("problema", choices=PROBLEMAS.keys())
    ap.add_argument("x_min", help="límite inferior del intervalo ('auto' = valor del registro)")
    ap.add_argument("x_max", help="límite superior del intervalo ('auto' = valor del registro)")
    ap.add_argument("tol", type=float, help="tolerancia / exactitud")
    ap.add_argument("tam_pob", type=int, help="tamaño de población")
    ap.add_argument("num_gen", type=int, help="número de generaciones")
    ap.add_argument("prob_cruce", type=float, help="tasa de cruce")
    ap.add_argument("prob_mut", type=float, help="tasa de mutación")
    ap.add_argument("--num-var", type=int, default=3,
                    help="nº de variables (solo para problemas con num_var=None)")
    ap.add_argument("--semilla", type=int, default=None, help="semilla aleatoria")
    ap.add_argument("--listar", action="store_true", help="lista los problemas y sale")
    ap.add_argument("--gui", action="store_true", help="abre la interfaz gráfica")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.semilla is not None:
        random.seed(a.semilla)

    p = PROBLEMAS[a.problema]
    x_min = p.x_min if a.x_min == "auto" else float(a.x_min)
    x_max = p.x_max if a.x_max == "auto" else float(a.x_max)
    num_var = a.num_var if p.num_var is None else p.num_var

    ag = AG(p.funcion, p.minimizar, x_min, x_max, a.tol, a.tam_pob, a.num_gen,
            a.prob_cruce, a.prob_mut, num_var)
    mejor = ag.ejecutar(a.verbose)

    xs = ", ".join(f"{v:.5f}" for v in mejor.x)
    print(f"Aptitud del mejor({ag.pos_mejor}) = {mejor.aptitud_bruta:.5f} en {xs}")


if __name__ == "__main__":
    main()