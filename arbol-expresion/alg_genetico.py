"""Ajuste de expresiones mediante un algoritmo genético.

La interfaz se ejecuta con ``python alg_genetico.py``. También se pueden usar
las clases desde otro módulo sin crear la ventana:

    datos = DatosCSV.cargar("puntos.csv")
    resultado = AlgoritmoGenetico(datos, generaciones=100).ejecutar()

El genotipo de un individuo es una expresión SymPy, su fenotipo es la misma
expresión serializada y su adaptación es el error (menor es mejor).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import math
import random
import sys
from typing import Callable, Iterable

import numpy as np
import sympy as sp

from arbol import (
    BIBLIOTECA_OPERADORES,
    BibliotecaOperadores,
    Operador,
    construir_arbol,
    dibujar_arbol,
    generar_expresion_aleatoria,
    hacer_funcion,
    protegido,
    reparar,
)


def biblioteca_genetica() -> BibliotecaOperadores:
    """Biblioteca ampliada, sin modificar la biblioteca usada por arbol.py."""
    biblioteca = BibliotecaOperadores(BIBLIOTECA_OPERADORES.operadores())
    for nombre, funcion in (
        ("sinh", sp.sinh), ("cosh", sp.cosh), ("tanh", sp.tanh),
        ("atan", sp.atan), ("asinh", sp.asinh), ("acosh", sp.acosh),
    ):
        biblioteca.registrar(Operador(nombre, 1, lambda a, fn=funcion: fn(a[0]), 0.7))
    biblioteca.registrar(Operador("promedio", 2, lambda a: (a[0] + a[1]) / 2, 0.8))
    return biblioteca


BIBLIOTECA_GENETICA = biblioteca_genetica()


def _residuos_seguro(real: np.ndarray, pred: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        residuos = np.subtract(real, pred, dtype=np.float64)
    return residuos


def _mse(real: np.ndarray, pred: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        valor = np.mean(np.square(_residuos_seguro(real, pred)))
    return float(valor) if np.isfinite(valor) else math.inf


def _rmse(real: np.ndarray, pred: np.ndarray) -> float:
    mse = _mse(real, pred)
    return float(np.sqrt(mse)) if math.isfinite(mse) else math.inf


def _mae(real: np.ndarray, pred: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        valor = np.mean(np.abs(_residuos_seguro(real, pred)))
    return float(valor) if np.isfinite(valor) else math.inf


def _nmse(real: np.ndarray, pred: np.ndarray) -> float:
    mse = _mse(real, pred)
    varianza = float(np.var(real))
    if not math.isfinite(mse) or not math.isfinite(varianza):
        return math.inf
    return mse / max(varianza, 1e-12)


METRICAS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "MSE": _mse,
    "RMSE": _rmse,
    "MAE": _mae,
    "NMSE": _nmse,
}


@dataclass(frozen=True)
class DatosCSV:
    x: np.ndarray
    y: np.ndarray
    nombre_x: str = "x"
    nombre_y: str = "y"

    @classmethod
    def cargar(cls, ruta: str) -> "DatosCSV":
        """Carga dos columnas numéricas, preferentemente llamadas x y y/f."""
        with open(ruta, newline="", encoding="utf-8-sig") as archivo:
            lector = csv.reader(archivo)
            filas = [fila for fila in lector if any(c.strip() for c in fila)]
        if len(filas) < 2:
            raise ValueError("El CSV debe contener al menos dos puntos.")

        encabezados = [c.strip().lstrip("#").strip().lower() for c in filas[0]]
        try:
            valores = np.asarray(
                [[float(c.strip()) for c in fila] for fila in filas[1:]], dtype=float
            )
        except ValueError:
            try:
                valores = np.asarray(
                    [[float(c.strip()) for c in fila] for fila in filas], dtype=float
                )
                encabezados = ["x", "y"]
            except ValueError as exc:
                raise ValueError("Todas las filas del CSV deben contener números.") from exc
        if valores.ndim != 2 or valores.shape[1] < 2:
            raise ValueError("El CSV debe tener por lo menos dos columnas numéricas.")

        def encontrar(nombres: set[str]) -> int | None:
            return next((i for i, nombre in enumerate(encabezados) if nombre in nombres), None)

        ix = encontrar({"x", "entrada", "input"})
        iy = encontrar({"y", "f", "salida", "output", "objetivo", "target"})
        if ix is None:
            ix = 0
        if iy is None or iy == ix:
            iy = 1 if ix != 1 else 0
        mask = np.isfinite(valores[:, ix]) & np.isfinite(valores[:, iy])
        if mask.sum() < 2:
            raise ValueError("Se necesitan al menos dos puntos finitos.")
        return cls(
            valores[mask, ix],
            valores[mask, iy],
            encabezados[ix] or "x",
            encabezados[iy] or "y",
        )


@dataclass
class Individuo:
    genotipo: sp.Expr
    fenotipo: str = ""
    adaptacion: float = math.inf
    prediccion: np.ndarray | None = field(default=None, repr=False)

    def actualizar(self, datos: DatosCSV, metrica: str) -> None:
        try:
            if not self.genotipo.has(protegido):
                reparada = _reparar_segura(self.genotipo)
                if reparada is None:
                    raise ValueError("Expresión no reparable.")
                self.genotipo = reparada
            self.fenotipo = sp.sstr(self.genotipo)
            if _expresion_invalida(self.genotipo):
                raise ValueError("La expresión contiene valores no reales o indefinidos.")
            funcion = hacer_funcion(self.genotipo, [sp.Symbol(datos.nombre_x)])
            with np.errstate(all="ignore"):
                valores = np.asarray(funcion(datos.x), dtype=complex)
            if np.any(np.abs(np.imag(valores)) > 1e-9):
                raise ValueError("La expresión produjo valores complejos.")
            pred = np.real(valores).astype(float, copy=False)
            if pred.shape == ():
                pred = np.full_like(datos.x, float(pred))
            if pred.shape != datos.x.shape:
                pred = np.broadcast_to(pred, datos.x.shape)
            valido = np.isfinite(pred) & np.isfinite(datos.y)
            self.prediccion = np.where(valido, pred, np.nan)
            if not valido.all():
                self.adaptacion = math.inf
            else:
                self.adaptacion = METRICAS[metrica](datos.y, pred)
                if not math.isfinite(self.adaptacion):
                    self.adaptacion = math.inf
        except (ArithmeticError, KeyError, TypeError, ValueError, ZeroDivisionError):
            self.prediccion = None
            self.adaptacion = math.inf


def _expresion_invalida(expr: sp.Expr) -> bool:
    """Detecta constantes que ``lambdify`` no puede imprimir o evaluar."""
    if expr.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        return True
    if expr.is_real is False:
        return True
    return any(
        constante.is_real is False or constante.is_finite is False
        for constante in expr.atoms(sp.Number)
    )


def _reparar_segura(expr: sp.Expr) -> sp.Expr | None:
    """Repara una expresión solo si SymPy puede recorrerla sin desbordar."""
    if _expresion_invalida(expr):
        return None
    try:
        reparada, _ = reparar(expr)
    except (ArithmeticError, OverflowError, TypeError, ValueError, ZeroDivisionError):
        return None
    return None if _expresion_invalida(reparada) else reparada


def _rutas(expr: sp.Expr, internas: bool | None = None) -> list[tuple[int, ...]]:
    resultado: list[tuple[int, ...]] = []

    def visitar(nodo: sp.Expr, ruta: tuple[int, ...]) -> None:
        es_interno = bool(nodo.args)
        if internas is None or es_interno == internas:
            resultado.append(ruta)
        for i, hijo in enumerate(nodo.args):
            visitar(hijo, ruta + (i,))

    visitar(expr, ())
    return resultado


def _en(expr: sp.Expr, ruta: tuple[int, ...]) -> sp.Expr:
    for indice in ruta:
        expr = expr.args[indice]
    return expr


def _reemplazar(expr: sp.Expr, ruta: tuple[int, ...], nuevo: sp.Expr) -> sp.Expr:
    if not ruta:
        return nuevo
    indice = ruta[0]
    args = list(expr.args)
    args[indice] = _reemplazar(args[indice], ruta[1:], nuevo)
    try:
        return expr.func(*args)
    except (ArithmeticError, OverflowError, TypeError, ValueError, ZeroDivisionError):
        # Min/Max y otras funciones evaluadas por SymPy pueden rechazar una
        # combinación antes de que el individuo llegue a la evaluación numérica.
        return sp.nan


def _profundidad(expr: sp.Expr) -> int:
    """La raíz tiene profundidad cero."""
    return 0 if not expr.args else 1 + max(_profundidad(hijo) for hijo in expr.args)


def _recortar_expr(expr: sp.Expr, limite: int) -> sp.Expr:
    """Devuelve una expresión parcial para animar su construcción."""
    contador = [0]
    marcador = sp.Symbol("...")

    def visitar(nodo: sp.Expr) -> sp.Expr:
        contador[0] += 1
        if contador[0] > limite:
            return marcador
        if not nodo.args:
            return nodo
        return nodo.func(*(visitar(hijo) for hijo in nodo.args))

    return visitar(expr)


def _cantidad_nodos(expr: sp.Expr) -> int:
    return 1 + sum(_cantidad_nodos(hijo) for hijo in expr.args)


def _hoja_aleatoria(
    variables: tuple[str, ...], rng: random.Random
) -> sp.Expr:
    """Terminales de programación genética: variables y constantes reales."""
    if rng.random() < 0.55:
        return sp.Symbol(rng.choice(variables))
    return sp.Float(rng.uniform(-20.0, 20.0), 4)


def _generar_expresion_gp(
    profundidad: int,
    variables: tuple[str, ...],
    biblioteca: BibliotecaOperadores,
    rng: random.Random,
) -> sp.Expr:
    """Genera árboles con constantes reales y presencia frecuente de variables."""
    if profundidad <= 0 or rng.random() < 0.24:
        return _hoja_aleatoria(variables, rng)
    for _ in range(12):
        operador = rng.choice(biblioteca.operadores())
        argumentos = tuple(
            _generar_expresion_gp(profundidad - 1, variables, biblioteca, rng)
            for _ in range(operador.aridad)
        )
        try:
            resultado = operador.aplicar(argumentos)
            if not _expresion_invalida(resultado):
                return resultado
        except (TypeError, ValueError):
            continue
    return _hoja_aleatoria(variables, rng)


class Mutador:
    """Implementa las cuatro mutaciones con probabilidad uniforme."""

    NOMBRES = (
        "mutacion de hojas",
        "mutacion de nodos internos",
        "switching",
        "generacion de nueva rama",
    )

    def __init__(
        self,
        profundidad_maxima: int,
        biblioteca: BibliotecaOperadores = BIBLIOTECA_GENETICA,
        rng: random.Random | None = None,
    ):
        if profundidad_maxima < 0:
            raise ValueError("La profundidad máxima debe ser no negativa.")
        self.profundidad_maxima = profundidad_maxima
        self.biblioteca = biblioteca
        self.rng = rng or random.Random()

    def aplicar(self, expr: sp.Expr, variables: tuple[str, ...]) -> tuple[sp.Expr, str]:
        metodo = self.rng.choice(self.NOMBRES)
        if metodo == self.NOMBRES[0]:
            return self._hojas(expr, variables), metodo
        if metodo == self.NOMBRES[1]:
            return self._internos(expr), metodo
        if metodo == self.NOMBRES[2]:
            return self._switching(expr), metodo
        return self._rama(expr, variables), metodo

    def _hojas(self, expr: sp.Expr, variables: tuple[str, ...]) -> sp.Expr:
        rutas = _rutas(expr, False)
        if not rutas:
            return expr
        simbolo = sp.Symbol(self.rng.choice(variables))
        nuevo = (
            simbolo
            if self.rng.random() < 0.55
            else sp.Float(self.rng.uniform(-20.0, 20.0), 4)
        )
        return _reemplazar(expr, self.rng.choice(rutas), nuevo)

    def _internos(self, expr: sp.Expr) -> sp.Expr:
        rutas = _rutas(expr, True)
        if not rutas:
            return expr
        ruta = self.rng.choice(rutas)
        nodo = _en(expr, ruta)
        opciones = self.biblioteca.operadores(len(nodo.args))
        if not opciones:
            return expr
        operador = self.rng.choice(opciones)
        try:
            nuevo = operador.aplicar(nodo.args)
        except (TypeError, ValueError):
            return expr
        return _reemplazar(expr, ruta, nuevo)

    def _switching(self, expr: sp.Expr) -> sp.Expr:
        rutas = [r for r in _rutas(expr, True) if len(_en(expr, r).args) >= 2]
        if not rutas:
            return expr
        ruta = self.rng.choice(rutas)
        nodo = _en(expr, ruta)
        args = list(nodo.args)
        i, j = self.rng.sample(range(len(args)), 2)
        args[i], args[j] = args[j], args[i]
        return _reemplazar(expr, ruta, nodo.func(*args))

    def _rama(self, expr: sp.Expr, variables: tuple[str, ...]) -> sp.Expr:
        rutas = _rutas(expr)
        ruta = self.rng.choice(rutas)
        nueva = _generar_segura(
            self.profundidad_maxima, variables, self.biblioteca, self.rng
        )
        return _reemplazar(expr, ruta, nueva)


def _generar_segura(
    profundidad: int,
    variables: tuple[str, ...],
    biblioteca: BibliotecaOperadores,
    rng: random.Random,
) -> sp.Expr:
    """Genera y repara antes de aceptar, respetando la profundidad efectiva."""
    for _ in range(32):
        resultado = _generar_expresion_gp(profundidad, variables, biblioteca, rng)
        reparado = _reparar_segura(resultado)
        if reparado is not None and _profundidad(reparado) <= profundidad:
            return reparado
    return sp.Symbol(variables[0])


class AlgoritmoGenetico:
    def __init__(
        self,
        datos: DatosCSV,
        poblacion: int = 60,
        generaciones: int = 100,
        profundidad_maxima: int = 4,
        tasa_mutacion: float = 0.02,
        tasa_cruza: float = 0.8,
        elitismo: int = 2,
        metrica: str = "MSE",
        semilla: int | None = None,
        biblioteca: BibliotecaOperadores = BIBLIOTECA_GENETICA,
    ):
        if poblacion < 2 or generaciones < 1:
            raise ValueError("La población debe ser >= 2 y las generaciones >= 1.")
        if not 0 <= tasa_mutacion <= 0.05:
            raise ValueError("La tasa de mutación debe estar entre 0% y 5%.")
        if not 0 <= tasa_cruza <= 1:
            raise ValueError("Las tasas deben estar entre 0 y 1.")
        if not 0 <= elitismo < poblacion:
            raise ValueError("El elitismo debe estar entre 0 y población - 1.")
        if metrica not in METRICAS:
            raise ValueError(f"Métrica desconocida: {metrica}.")
        self.datos, self.poblacion_n = datos, poblacion
        self.generaciones, self.profundidad = generaciones, profundidad_maxima
        self.tasa_mutacion, self.tasa_cruza, self.elitismo = tasa_mutacion, tasa_cruza, elitismo
        self.metrica, self.rng = metrica, random.Random(semilla)
        self.biblioteca = biblioteca
        self.variables = (datos.nombre_x,)
        self.mutador = Mutador(profundidad_maxima, biblioteca, self.rng)
        self.mejor: Individuo | None = None
        self.historial: list[float] = []
        self.poblaciones: list[list[Individuo]] = []
        self.generacion_actual = 0
        self._actual: list[Individuo] = []

    def _nuevo(self) -> Individuo:
        expr = _generar_segura(
            self.profundidad, self.variables, self.biblioteca, self.rng
        )
        individuo = Individuo(expr)
        individuo.actualizar(self.datos, self.metrica)
        if _profundidad(individuo.genotipo) > self.profundidad:
            individuo = Individuo(
                _generar_segura(
                    self.profundidad, self.variables, self.biblioteca, self.rng
                )
            )
            individuo.actualizar(self.datos, self.metrica)
        return individuo

    def _poblacion_inicial(self) -> list[Individuo]:
        """Incluye una base funcional para que la búsqueda no empiece solo
        con constantes; el resto usa árboles aleatorios ramped."""
        x = sp.Symbol(self.variables[0])
        semillas = [x, x**2, -x, sp.sin(x), sp.cos(x), sp.Abs(x)]
        try:
            for grado in range(1, min(4, len(self.datos.x) - 1) + 1):
                coeficientes = np.polyfit(self.datos.x, self.datos.y, grado)
                polinomio = sp.Integer(0)
                for coeficiente in coeficientes:
                    polinomio = polinomio * x + sp.Float(float(coeficiente), 6)
                if _profundidad(polinomio) <= self.profundidad:
                    semillas.append(polinomio)
            # Las muestras de puntos oscilan; una base armónica permite
            # descubrir frecuencias sin esperar a que una mutación acierte
            # simultáneamente la función y sus coeficientes.
            mejor_armonica: tuple[float, sp.Expr] | None = None
            for frecuencia in np.linspace(0.1, 6.0, 60):
                matriz = np.column_stack((
                    np.ones_like(self.datos.x),
                    np.sin(frecuencia * self.datos.x),
                    np.cos(frecuencia * self.datos.x),
                ))
                coeficientes, _, _, _ = np.linalg.lstsq(
                    matriz, self.datos.y, rcond=None
                )
                prediccion = matriz @ coeficientes
                error = float(np.mean((prediccion - self.datos.y) ** 2))
                expresion = (
                    sp.Float(float(coeficientes[0]), 6)
                    + sp.Float(float(coeficientes[1]), 6) * sp.sin(
                        sp.Float(float(frecuencia), 6) * x
                    )
                    + sp.Float(float(coeficientes[2]), 6) * sp.cos(
                        sp.Float(float(frecuencia), 6) * x
                    )
                )
                if _profundidad(expresion) <= self.profundidad and (
                    mejor_armonica is None or error < mejor_armonica[0]
                ):
                    mejor_armonica = (error, expresion)
            if mejor_armonica is not None:
                semillas.append(mejor_armonica[1])
        except (ArithmeticError, TypeError, ValueError):
            pass
        individuos: list[Individuo] = []
        for expresion in semillas:
            if _profundidad(expresion) <= self.profundidad:
                individuo = Individuo(expresion)
                individuo.actualizar(self.datos, self.metrica)
                if _profundidad(individuo.genotipo) <= self.profundidad:
                    individuos.append(individuo)
        individuos.extend(self._nuevo() for _ in range(self.poblacion_n - len(individuos)))
        return individuos

    def _torneo(self, poblacion: list[Individuo]) -> Individuo:
        return min(self.rng.sample(poblacion, min(3, len(poblacion))), key=lambda i: i.adaptacion)

    def _ruleta(self, poblacion: list[Individuo]) -> Individuo:
        """Ruleta por rango, estable incluso con errores muy diferentes."""
        ordenados = sorted(poblacion, key=lambda i: i.adaptacion)
        finitos = [i for i in ordenados if math.isfinite(i.adaptacion)]
        if not finitos:
            return self.rng.choice(poblacion)
        invalidos = [i for i in ordenados if not math.isfinite(i.adaptacion)]
        candidatos = finitos + invalidos
        pesos = list(range(len(finitos), 0, -1)) + [0] * len(invalidos)
        return self.rng.choices(candidatos, weights=pesos, k=1)[0]

    def _cruzar(self, uno: sp.Expr, dos: sp.Expr) -> sp.Expr:
        rutas_uno, rutas_dos = _rutas(uno), _rutas(dos)
        ruta_uno = self.rng.choice(rutas_uno)
        subarbol = _en(dos, self.rng.choice(rutas_dos))
        hijo = _reemplazar(uno, ruta_uno, subarbol)
        if _expresion_invalida(hijo):
            return _generar_segura(
                self.profundidad, self.variables, self.biblioteca, self.rng
            )
        reparado = _reparar_segura(hijo)
        if reparado is not None and _profundidad(reparado) <= self.profundidad:
            return reparado

        # Se conserva la estructura del padre y se regenera únicamente la
        # rama cruzada con el espacio de profundidad que todavía queda.
        profundidad_padre = len(ruta_uno)
        disponible = self.profundidad - profundidad_padre
        if disponible < 0:
            return _generar_segura(self.profundidad, self.variables, self.biblioteca, self.rng)
        rama = _generar_segura(disponible, self.variables, self.biblioteca, self.rng)
        reparado = _reemplazar(uno, ruta_uno, rama)
        reparado_seguro = _reparar_segura(reparado)
        if (
            reparado_seguro is not None
            and _profundidad(reparado_seguro) <= self.profundidad
        ):
            return reparado_seguro
        return _generar_segura(self.profundidad, self.variables, self.biblioteca, self.rng)

    @staticmethod
    def _copiar(individuo: Individuo) -> Individuo:
        copia = Individuo(individuo.genotipo, individuo.fenotipo, individuo.adaptacion)
        copia.prediccion = None if individuo.prediccion is None else individuo.prediccion.copy()
        return copia

    def inicializar(
        self,
        progreso: Callable[[int, list[Individuo], Individuo], None] | None = None,
    ) -> None:
        """Crea y evalúa la generación 0, sin avanzar el algoritmo."""
        actual = self._poblacion_inicial()
        self.poblaciones.clear()
        self.historial.clear()
        self.mejor = None
        self.generacion_actual = 0
        self._actual = actual
        self._registrar_generacion(progreso)

    def _registrar_generacion(
        self,
        progreso: Callable[[int, list[Individuo], Individuo], None] | None = None,
    ) -> None:
        self._actual.sort(key=lambda i: i.adaptacion)
        if self.mejor is None or self._actual[0].adaptacion < self.mejor.adaptacion:
            self.mejor = self._copiar(self._actual[0])
        self.historial.append(self.mejor.adaptacion)
        snapshot = [self._copiar(individuo) for individuo in self._actual]
        self.poblaciones.append(snapshot)
        if progreso:
            progreso(self.generacion_actual, snapshot, self.mejor)

    def paso(
        self,
        progreso: Callable[[int, list[Individuo], Individuo], None] | None = None,
    ) -> bool:
        """Avanza una generación. Devuelve False cuando ya se llegó al final."""
        if not self._actual:
            self.inicializar(progreso)
        if self.generacion_actual >= self.generaciones:
            return False
        actual = self._actual
        siguiente = [self._copiar(i) for i in sorted(actual, key=lambda i: i.adaptacion)[:self.elitismo]]
        while len(siguiente) < self.poblacion_n:
            padre = self._ruleta(actual)
            expr = padre.genotipo
            if self.rng.random() < self.tasa_cruza:
                expr = self._cruzar(expr, self._ruleta(actual).genotipo)
            if self.rng.random() < self.tasa_mutacion:
                expr, _ = self.mutador.aplicar(expr, self.variables)
            if _profundidad(expr) > self.profundidad:
                expr = _generar_segura(
                    self.profundidad, self.variables, self.biblioteca, self.rng
                )
            hijo = Individuo(expr)
            hijo.actualizar(self.datos, self.metrica)
            if _profundidad(hijo.genotipo) > self.profundidad:
                hijo = self._nuevo()
            siguiente.append(hijo)
        self._actual = siguiente
        self.generacion_actual += 1
        self._registrar_generacion(progreso)
        return True

    def ejecutar(
        self,
        progreso: Callable[[int, list[Individuo], Individuo], None] | None = None,
    ) -> Individuo:
        self.inicializar(progreso)
        while self.generacion_actual < self.generaciones:
            self.paso(progreso)
        assert self.mejor is not None
        return self.mejor


def _spin(valor: int, minimo: int, maximo: int):
    from PyQt6.QtWidgets import QSpinBox
    control = QSpinBox()
    control.setRange(minimo, maximo)
    control.setValue(valor)
    return control


def ejecutar_gui() -> None:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import (
        QApplication, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMainWindow,
        QMessageBox, QPushButton, QComboBox, QVBoxLayout, QWidget, QDoubleSpinBox,
        QSlider, QTabWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    class VentanaGenetica(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Ajuste genético de árboles de expresión")
            self.resize(1200, 780)
            self.ruta = "puntos.csv"
            self.datos: DatosCSV | None = None
            self.algoritmo: AlgoritmoGenetico | None = None
            self.generacion_actual = 0
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.siguiente_generacion)
            self.tabs = QTabWidget()
            self.fig_error = Figure(figsize=(10, 6))
            self.fig_match = Figure(figsize=(10, 6))
            self.fig_convergencia = Figure(figsize=(10, 6))
            self.canvas_error = FigureCanvasQTAgg(self.fig_error)
            self.canvas_match = FigureCanvasQTAgg(self.fig_match)
            self.canvas_convergencia = FigureCanvasQTAgg(self.fig_convergencia)
            self.tabs.addTab(self.canvas_error, "Evolución del error")
            self.tabs.addTab(self.canvas_match, "Match con CSV")
            self.tabs.addTab(self.canvas_convergencia, "Soluciones convergiendo")
            self.archivo = QLabel("Archivo: puntos.csv")
            abrir = QPushButton("Cargar CSV")
            abrir.clicked.connect(self.cargar)
            ejecutar = QPushButton("Ejecutar algoritmo")
            ejecutar.clicked.connect(self.ejecutar)
            self.poblacion = _spin(60, 2, 1000)
            self.generaciones = _spin(100, 1, 10000)
            self.profundidad = _spin(4, 1, 10)
            self.elitismo = _spin(2, 0, 999)
            self.mutacion = QDoubleSpinBox()
            self.mutacion.setRange(0, 5)
            self.mutacion.setDecimals(2)
            self.mutacion.setSingleStep(0.5)
            self.mutacion.setValue(2)
            self.cruza = _spin(80, 0, 100)
            self.metrica = QComboBox()
            self.metrica.addItems(METRICAS)
            self.estado = QLabel("Cargue un CSV y ejecute.")
            formulario = QFormLayout()
            formulario.addRow(self.archivo, abrir)
            formulario.addRow("Población", self.poblacion)
            formulario.addRow("Generaciones", self.generaciones)
            formulario.addRow("Profundidad máxima", self.profundidad)
            formulario.addRow("Elitismo", self.elitismo)
            formulario.addRow("Mutación controlada (0-5%)", self.mutacion)
            formulario.addRow("Cruza (%)", self.cruza)
            formulario.addRow("Error", self.metrica)
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.addLayout(formulario)
            layout.addWidget(ejecutar)
            layout.addWidget(self.estado)
            controles = QHBoxLayout()
            self.inicio = QPushButton("⏮ Inicio")
            self.anterior = QPushButton("◀ Anterior")
            self.siguiente = QPushButton("Siguiente ▶")
            self.reproducir = QPushButton("▶ Reproducir")
            self.avanzar = QPushButton("+10")
            self.final = QPushButton("Final ⏭")
            self.inicio.clicked.connect(lambda: self.mostrar_generacion(0))
            self.anterior.clicked.connect(lambda: self.mostrar_generacion(self.generacion_actual - 1))
            self.siguiente.clicked.connect(self.siguiente_generacion)
            self.reproducir.clicked.connect(self.alternar_reproduccion)
            self.avanzar.clicked.connect(lambda: self.avanzar_generaciones(10))
            self.final.clicked.connect(self.ir_al_final)
            for boton in (
                self.inicio, self.anterior, self.siguiente, self.reproducir,
                self.avanzar, self.final,
            ):
                controles.addWidget(boton)
            self.velocidad = _spin(150, 10, 3000)
            self.velocidad.setSingleStep(50)
            controles.addWidget(QLabel("Pausa (ms):"))
            controles.addWidget(self.velocidad)
            layout.addLayout(controles)
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setEnabled(False)
            self.slider.valueChanged.connect(self._slider_movido)
            layout.addWidget(self.slider)
            self.paso_arbol = QSlider(Qt.Orientation.Horizontal)
            self.paso_arbol.setEnabled(False)
            self.paso_arbol.valueChanged.connect(self._paso_arbol_movido)
            layout.addWidget(QLabel("Construcción del árbol"))
            layout.addWidget(self.paso_arbol)
            layout.addStretch()
            contenedor = QHBoxLayout()
            contenedor.addWidget(panel, 0)
            contenedor.addWidget(self.tabs, 1)
            central = QWidget()
            central.setLayout(contenedor)
            self.setCentralWidget(central)
            self._botones = (
                self.inicio, self.anterior, self.siguiente, self.reproducir,
                self.avanzar, self.final, self.slider, self.paso_arbol,
            )

        def cargar(self):
            ruta, _ = QFileDialog.getOpenFileName(self, "Cargar puntos", "", "CSV (*.csv)")
            if ruta:
                self.ruta = ruta
                self.archivo.setText(f"Archivo: {ruta}")

        def ejecutar(self):
            try:
                self.timer.stop()
                self.reproducir.setText("▶ Reproducir")
                datos = DatosCSV.cargar(self.ruta)
                self.algoritmo = AlgoritmoGenetico(
                    datos, self.poblacion.value(), self.generaciones.value(),
                    self.profundidad.value(), self.mutacion.value() / 100,
                    self.cruza.value() / 100, min(self.elitismo.value(), self.poblacion.value() - 1),
                    self.metrica.currentText(),
                )
                self.datos = datos
                self.algoritmo.inicializar(
                    lambda n, poblacion, b: self.estado.setText(
                        f"Generación {n}/{self.algoritmo.generaciones} | "
                        f"mejor error: {b.adaptacion:.6g}"
                    )
                )
                self.slider.setRange(0, self.algoritmo.generaciones)
                self.slider.setEnabled(True)
                self.generacion_actual = 0
                self._actualizar_controles()
                self.mostrar_generacion(0)
            except (OSError, ValueError, TypeError) as exc:
                QMessageBox.critical(self, "No se pudo ejecutar", str(exc))

        def _calcular_una(self):
            if self.algoritmo is not None:
                self.algoritmo.paso(
                    lambda n, poblacion, b: self.estado.setText(
                        f"Generación {n}/{self.algoritmo.generaciones} | "
                        f"mejor error: {b.adaptacion:.6g}"
                    )
                )

        def siguiente_generacion(self):
            if self.algoritmo is None:
                return
            if self.generacion_actual < self.algoritmo.generaciones:
                if self.generacion_actual >= len(self.algoritmo.poblaciones) - 1:
                    self._calcular_una()
                self.generacion_actual += 1
                self.mostrar_generacion(self.generacion_actual)
            else:
                self.timer.stop()
                self.reproducir.setText("▶ Reproducir")

        def avanzar_generaciones(self, cantidad: int):
            if self.algoritmo is None:
                return
            destino = min(
                self.generacion_actual + cantidad,
                self.algoritmo.generaciones,
            )
            while len(self.algoritmo.poblaciones) - 1 < destino:
                self._calcular_una()
            self.generacion_actual = destino
            self.mostrar_generacion(destino)

        def ir_al_final(self):
            self.avanzar_generaciones(10 ** 9)

        def alternar_reproduccion(self):
            if self.algoritmo is None:
                return
            if self.timer.isActive():
                self.timer.stop()
                self.reproducir.setText("▶ Reproducir")
            else:
                if self.generacion_actual >= self.algoritmo.generaciones:
                    self.generacion_actual = 0
                self.timer.start(self.velocidad.value())
                self.reproducir.setText("⏸ Pausa")

        def _slider_movido(self, indice: int):
            if self.algoritmo is None or indice == self.generacion_actual:
                return
            while len(self.algoritmo.poblaciones) - 1 < indice:
                self._calcular_una()
            self.generacion_actual = indice
            self.mostrar_generacion(indice)

        def _paso_arbol_movido(self, indice: int):
            if self.algoritmo is not None and self.datos is not None:
                self.mostrar_generacion(self.generacion_actual, indice)

        def _actualizar_controles(self):
            activo = self.algoritmo is not None
            for boton in self._botones:
                boton.setEnabled(activo)
            if activo:
                self.inicio.setEnabled(self.generacion_actual > 0)
                self.anterior.setEnabled(self.generacion_actual > 0)
                final = self.generacion_actual >= self.algoritmo.generaciones
                self.siguiente.setEnabled(not final)
                self.avanzar.setEnabled(not final)
                self.final.setEnabled(not final)

        def mostrar_generacion(self, indice: int, paso_arbol: int | None = None):
            if self.algoritmo is None or self.datos is None or not self.algoritmo.poblaciones:
                return
            indice = max(0, min(indice, len(self.algoritmo.poblaciones) - 1))
            self.generacion_actual = indice
            self.slider.blockSignals(True)
            self.slider.setValue(indice)
            self.slider.blockSignals(False)
            poblacion = self.algoritmo.poblaciones[indice]
            mejor = min(poblacion, key=lambda individuo: individuo.adaptacion)
            total_nodos = _cantidad_nodos(mejor.genotipo)
            self.paso_arbol.blockSignals(True)
            self.paso_arbol.setRange(1, total_nodos)
            if paso_arbol is None:
                paso_arbol = total_nodos
            self.paso_arbol.setValue(max(1, min(paso_arbol, total_nodos)))
            self.paso_arbol.setEnabled(True)
            self.paso_arbol.blockSignals(False)

            self.fig_error.clear()
            ax_error = self.fig_error.add_subplot(111)
            ax_error.plot(
                range(indice + 1),
                self.algoritmo.historial[:indice + 1],
                color="#b03a2e",
                linewidth=2,
            )
            ax_error.scatter(indice, self.algoritmo.historial[indice], color="#17202a")
            ax_error.set_title(f"Error hasta la generación {indice}")
            ax_error.set_xlabel("Generación")
            ax_error.set_ylabel(self.algoritmo.metrica)
            ax_error.grid(alpha=0.3)
            self.canvas_error.draw()

            self.fig_match.clear()
            ax_match = self.fig_match.add_subplot(111)
            ax_match.scatter(self.datos.x, self.datos.y, label="Puntos del CSV", zorder=3)
            if mejor.prediccion is not None:
                ax_match.plot(
                    self.datos.x, mejor.prediccion,
                    color="#2874a6", linewidth=2, label=mejor.fenotipo,
                )
            ax_match.set_title(
                f"Generación {indice}: {self.algoritmo.metrica}={mejor.adaptacion:.6g}"
            )
            ax_match.set_xlabel("x")
            ax_match.set_ylabel("y")
            ax_match.legend(fontsize=8)
            ax_match.grid(alpha=0.3)
            self.canvas_match.draw()

            self.fig_convergencia.clear()
            ax_convergencia = self.fig_convergencia.add_subplot(111)
            generaciones_mostradas = sorted(set(np.linspace(
                0, indice, min(10, indice + 1), dtype=int
            )))
            for generacion_mostrada in generaciones_mostradas:
                mejor_generacion = min(
                    self.algoritmo.poblaciones[generacion_mostrada],
                    key=lambda individuo: individuo.adaptacion,
                )
                if mejor_generacion.prediccion is not None:
                    ax_convergencia.plot(
                        self.datos.x,
                        mejor_generacion.prediccion,
                        alpha=0.35 if generacion_mostrada != indice else 1,
                        linewidth=1.2 if generacion_mostrada != indice else 2.5,
                        label=f"G{generacion_mostrada}",
                    )
            ax_convergencia.scatter(
                self.datos.x, self.datos.y, color="black", s=16, zorder=5, label="CSV"
            )
            ax_convergencia.set_title("Soluciones convergiendo generación a generación")
            ax_convergencia.set_xlabel("x")
            ax_convergencia.set_ylabel("f(x)")
            ax_convergencia.grid(alpha=0.3)
            ax_convergencia.legend(fontsize=7, ncol=2)
            self.canvas_convergencia.draw()

            # El árbol se muestra en la pestaña de match para conservar la
            # comparación directa entre solución y datos.
            ax_tree = self.fig_match.add_axes([0.72, 0.58, 0.25, 0.32])
            ax_tree.axis("off")
            try:
                expresion_parcial = _recortar_expr(mejor.genotipo, self.paso_arbol.value())
                raiz = construir_arbol(expresion_parcial)
                from arbol import posicionar
                posicionar(raiz)
                dibujar_arbol(raiz, ax_tree, "Árbol")
            except (TypeError, ValueError):
                ax_tree.text(0.5, 0.5, mejor.fenotipo, ha="center", va="center", wrap=True)
            self.canvas_match.draw()
            self._actualizar_controles()
            self.estado.setText(
                f"Generación {indice}/{self.algoritmo.generaciones} | "
                f"individuos: {len(poblacion)} | mejor: {mejor.fenotipo} | "
                f"error: {mejor.adaptacion:.6g} | "
                f"árbol: {self.paso_arbol.value()}/{total_nodos} nodos"
            )

    app = QApplication.instance() or QApplication(sys.argv)
    ventana = VentanaGenetica()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    ejecutar_gui()