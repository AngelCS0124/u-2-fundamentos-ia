#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Algoritmo Genético para el problema de las N Reinas - Interfaz PyQt5
=====================================================================

Requisitos:   pip install PyQt5
Ejecución:    python nqueens_ga.py

Representación:
    Cada individuo es una PERMUTACIÓN de 0..N-1. La posición i del vector es la
    columna y el valor es la fila de la reina. Así nunca hay conflictos de fila
    ni de columna; el algoritmo solo debe eliminar conflictos en diagonales.

Función de aptitud (se minimiza):
    Número de pares de reinas que se atacan en diagonal. Solución = 0.

Operadores:
    - Selección:  torneo de tamaño k
    - Cruce:      Order Crossover (OX) con probabilidad configurable
    - Mutación:   intercambio (swap) de dos genes con probabilidad configurable
    - Elitismo:   las mejores E soluciones pasan intactas a la siguiente generación
"""

import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, QRegularExpression, Qt, QThread, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
    QPolygonF,
    QRegularExpressionValidator,
)
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# =============================================================================
#  NÚCLEO DEL ALGORITMO GENÉTICO (sin dependencias de interfaz)
# =============================================================================
@dataclass
class GAParams:
    n: int
    pop_size: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int
    elite_count: int
    seed: Optional[int]


def count_conflicts(individual: List[int]) -> int:
    """Pares de reinas en la misma diagonal (filas/columnas ya son únicas)."""
    d1: Dict[int, int] = {}
    d2: Dict[int, int] = {}
    for col, row in enumerate(individual):
        d1[row - col] = d1.get(row - col, 0) + 1
        d2[row + col] = d2.get(row + col, 0) + 1
    return sum(c * (c - 1) // 2 for c in d1.values()) + sum(
        c * (c - 1) // 2 for c in d2.values()
    )


def order_crossover(p1: List[int], p2: List[int], rng: random.Random) -> List[int]:
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child: List[Optional[int]] = [None] * n
    child[a : b + 1] = p1[a : b + 1]
    used = set(p1[a : b + 1])
    remaining = (g for g in p2 if g not in used)
    for i in range(n):
        if child[i] is None:
            child[i] = next(remaining)
    return child  # type: ignore[return-value]


def swap_mutation(ind: List[int], rng: random.Random) -> None:
    i, j = rng.sample(range(len(ind)), 2)
    ind[i], ind[j] = ind[j], ind[i]


class GeneticAlgorithm:
    def __init__(self, params: GAParams):
        self.p = params
        self.rng = random.Random(params.seed)

    def _tournament(self, pop: List[List[int]], fit: List[int]) -> List[int]:
        idx = self.rng.sample(range(len(pop)), self.p.tournament_size)
        return pop[min(idx, key=fit.__getitem__)]

    def run(
        self,
        on_generation: Callable[[int, int, float, List[int]], None],
        should_stop: Callable[[], bool],
    ) -> Tuple[int, int, List[int], bool]:
        """
        Devuelve (generaciones_ejecutadas, mejores_conflictos, mejor_individuo, detenido).
        """
        p, rng = self.p, self.rng
        pop = [rng.sample(range(p.n), p.n) for _ in range(p.pop_size)]
        best_ind: List[int] = pop[0][:]
        best_fit = count_conflicts(best_ind)
        stopped = False
        gen = 0

        for gen in range(p.generations + 1):
            fit = [count_conflicts(ind) for ind in pop]
            gi = min(range(p.pop_size), key=fit.__getitem__)
            if fit[gi] < best_fit:
                best_fit, best_ind = fit[gi], pop[gi][:]

            on_generation(gen, best_fit, sum(fit) / len(fit), best_ind)

            if best_fit == 0 or gen == p.generations:
                break
            if should_stop():
                stopped = True
                break

            order = sorted(range(p.pop_size), key=fit.__getitem__)
            new_pop = [pop[i][:] for i in order[: p.elite_count]]
            while len(new_pop) < p.pop_size:
                p1 = self._tournament(pop, fit)
                p2 = self._tournament(pop, fit)
                if rng.random() < p.crossover_rate:
                    child = order_crossover(p1, p2, rng)
                else:
                    child = p1[:]
                if rng.random() < p.mutation_rate:
                    swap_mutation(child, rng)
                new_pop.append(child)
            pop = new_pop

        return gen, best_fit, best_ind, stopped


# =============================================================================
#  HILO DE TRABAJO (evita congelar la interfaz)
# =============================================================================
class GAWorker(QThread):
    progress = pyqtSignal(int, int, float, list)  # gen, mejor, promedio, individuo
    finished_run = pyqtSignal(int, int, list, float, bool)

    def __init__(self, params: GAParams):
        super().__init__()
        self.params = params
        self._stop = False
        self._last_emit = 0.0

    def request_stop(self) -> None:
        self._stop = True

    def _callback(self, gen: int, best: int, avg: float, ind: List[int]) -> None:
        now = time.monotonic()
        # Limita la frecuencia de actualización de la UI (~30 fps)
        if gen == 0 or best == 0 or now - self._last_emit >= 0.033:
            self._last_emit = now
            self.progress.emit(gen, best, avg, list(ind))

    def run(self) -> None:
        t0 = time.monotonic()
        ga = GeneticAlgorithm(self.params)
        gen, best, ind, stopped = ga.run(self._callback, lambda: self._stop)
        self.finished_run.emit(gen, best, list(ind), time.monotonic() - t0, stopped)


# =============================================================================
#  WIDGETS DE VISUALIZACIÓN
# =============================================================================
class BoardWidget(QWidget):
    """Tablero de ajedrez con las reinas. Las reinas en conflicto se pintan en rojo."""

    def __init__(self):
        super().__init__()
        self.solution: List[int] = []
        self.attacked: set = set()
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_solution(self, sol: List[int]) -> None:
        self.solution = list(sol)
        self.attacked = set()
        for c1 in range(len(sol)):
            for c2 in range(c1 + 1, len(sol)):
                if abs(sol[c1] - sol[c2]) == c2 - c1:
                    self.attacked.update((c1, c2))
        self.update()

    def clear(self) -> None:
        self.solution, self.attacked = [], set()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        n = len(self.solution)
        if n == 0:
            painter.setPen(QColor("#888"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Sin solución para mostrar")
            return

        side = min(self.width(), self.height()) - 4
        cell = side / n
        x0 = (self.width() - side) / 2
        y0 = (self.height() - side) / 2

        light, dark = QColor("#f0d9b5"), QColor("#b58863")
        painter.setPen(Qt.NoPen)
        for r in range(n):
            for c in range(n):
                painter.setBrush(light if (r + c) % 2 == 0 else dark)
                painter.drawRect(QRectF(x0 + c * cell, y0 + r * cell, cell, cell))

        font = QFont()
        font.setPixelSize(max(6, int(cell * 0.65)))
        painter.setFont(font)
        for col, row in enumerate(self.solution):
            rect = QRectF(x0 + col * cell, y0 + row * cell, cell, cell)
            bad = col in self.attacked
            painter.setPen(QPen(QColor("#000"), max(1.0, cell * 0.03)))
            painter.setBrush(QColor("#d32f2f") if bad else QColor("#2e7d32"))
            painter.drawEllipse(rect.adjusted(cell * 0.12, cell * 0.12, -cell * 0.12, -cell * 0.12))
            if cell >= 14:
                painter.setPen(QColor("white"))
                painter.drawText(rect, Qt.AlignCenter, "♛")


class ChartWidget(QWidget):
    """Gráfica simple: mejor (rojo) y promedio (azul) de conflictos por generación."""

    def __init__(self):
        super().__init__()
        self.gens: List[int] = []
        self.best: List[float] = []
        self.avg: List[float] = []
        self.setMinimumHeight(170)

    def add_point(self, gen: int, best: float, avg: float) -> None:
        self.gens.append(gen)
        self.best.append(best)
        self.avg.append(avg)
        self.update()

    def clear(self) -> None:
        self.gens, self.best, self.avg = [], [], []
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        ml, mr, mt, mb = 44, 12, 22, 24
        plot = QRectF(ml, mt, w - ml - mr, h - mt - mb)

        text_color = self.palette().text().color()
        painter.setPen(text_color)
        painter.drawText(6, 14, "Conflictos por generación")
        painter.setPen(QColor("#d32f2f"))
        painter.drawText(int(w - 200), 14, "● Mejor")
        painter.setPen(QColor("#1976d2"))
        painter.drawText(int(w - 130), 14, "● Promedio")

        painter.setPen(QPen(QColor(128, 128, 128), 1))
        painter.drawRect(plot)

        if not self.gens:
            return

        max_y = max(max(self.avg), max(self.best), 1.0)
        max_x = max(self.gens[-1], 1)

        painter.setPen(text_color)
        painter.drawText(4, int(plot.top() + 10), f"{max_y:.0f}")
        painter.drawText(4, int(plot.bottom()), "0")
        painter.drawText(int(plot.left()), h - 6, "0")
        painter.drawText(int(plot.right() - 40), h - 6, f"gen {max_x}")

        def to_point(g: int, v: float) -> QPointF:
            return QPointF(
                plot.left() + g / max_x * plot.width(),
                plot.bottom() - v / max_y * plot.height(),
            )

        for series, color in ((self.avg, "#1976d2"), (self.best, "#d32f2f")):
            painter.setPen(QPen(QColor(color), 2))
            poly = QPolygonF([to_point(g, v) for g, v in zip(self.gens, series)])
            painter.drawPolyline(poly)


# =============================================================================
#  VALIDACIÓN DE ENTRADAS
# =============================================================================
# Límites de cada parámetro: (mínimo, máximo)
LIMITS = {
    "n": (4, 200),
    "pop_size": (10, 10000),
    "generations": (1, 100000),
    "crossover_rate": (0.0, 1.0),
    "mutation_rate": (0.0, 1.0),
    "tournament_size": (2, 10000),
    "elite_count": (0, 9999),
    "seed": (0, 2**31 - 1),
}

LABELS = {
    "n": "Número de reinas (N)",
    "pop_size": "Tamaño de población",
    "generations": "Máx. generaciones",
    "crossover_rate": "Prob. de cruce",
    "mutation_rate": "Prob. de mutación",
    "tournament_size": "Tamaño del torneo",
    "elite_count": "Elitismo (nº de élite)",
    "seed": "Semilla",
}

DEFAULTS = {
    "n": "8",
    "pop_size": "100",
    "generations": "1000",
    "crossover_rate": "0.9",
    "mutation_rate": "0.2",
    "tournament_size": "3",
    "elite_count": "2",
    "seed": "",
}


def _parse_int(text: str, key: str, errors: Dict[str, str]) -> Optional[int]:
    lo, hi = LIMITS[key]
    text = text.strip()
    if not text:
        errors[key] = "El campo no puede estar vacío."
        return None
    try:
        value = int(text)
    except ValueError:
        errors[key] = "Debe ser un número entero."
        return None
    if not lo <= value <= hi:
        errors[key] = f"Debe estar entre {lo} y {hi}."
        return None
    return value


def _parse_float(text: str, key: str, errors: Dict[str, str]) -> Optional[float]:
    lo, hi = LIMITS[key]
    text = text.strip().replace(",", ".")
    if not text:
        errors[key] = "El campo no puede estar vacío."
        return None
    try:
        value = float(text)
    except ValueError:
        errors[key] = "Debe ser un número decimal (ej. 0.8)."
        return None
    if value != value or value in (float("inf"), float("-inf")):
        errors[key] = "Valor numérico no válido."
        return None
    if not lo <= value <= hi:
        errors[key] = f"Debe estar entre {lo} y {hi}."
        return None
    return value


def validate_params(raw: Dict[str, str]) -> Tuple[Optional[GAParams], Dict[str, str]]:
    """Valida todos los campos (tipo, rango y reglas cruzadas)."""
    errors: Dict[str, str] = {}

    n = _parse_int(raw["n"], "n", errors)
    pop = _parse_int(raw["pop_size"], "pop_size", errors)
    gens = _parse_int(raw["generations"], "generations", errors)
    cx = _parse_float(raw["crossover_rate"], "crossover_rate", errors)
    mut = _parse_float(raw["mutation_rate"], "mutation_rate", errors)
    tour = _parse_int(raw["tournament_size"], "tournament_size", errors)
    elite = _parse_int(raw["elite_count"], "elite_count", errors)

    seed: Optional[int] = None
    if raw["seed"].strip():
        seed = _parse_int(raw["seed"], "seed", errors)

    # Reglas que dependen de varios campos
    if pop is not None:
        if tour is not None and "tournament_size" not in errors and tour > pop:
            errors["tournament_size"] = "No puede ser mayor que el tamaño de población."
        if elite is not None and "elite_count" not in errors and elite >= pop:
            errors["elite_count"] = "Debe ser menor que el tamaño de población."

    if errors:
        return None, errors

    return (
        GAParams(
            n=n,  # type: ignore[arg-type]
            pop_size=pop,  # type: ignore[arg-type]
            generations=gens,  # type: ignore[arg-type]
            crossover_rate=cx,  # type: ignore[arg-type]
            mutation_rate=mut,  # type: ignore[arg-type]
            tournament_size=tour,  # type: ignore[arg-type]
            elite_count=elite,  # type: ignore[arg-type]
            seed=seed,
        ),
        {},
    )


# =============================================================================
#  VENTANA PRINCIPAL
# =============================================================================
INVALID_STYLE = "QLineEdit { border: 1px solid #d32f2f; background: #fdecea; color: #000; }"

TOOLTIPS = {
    "n": "Tamaño del tablero y cantidad de reinas (4 a 200).\nPara N=2 y N=3 no existe solución.",
    "pop_size": "Cantidad de individuos en cada generación (10 a 10000).",
    "generations": "Límite de generaciones si no se encuentra solución antes.",
    "crossover_rate": "Probabilidad de aplicar cruce OX entre dos padres (0 a 1).",
    "mutation_rate": "Probabilidad de que un hijo mute (intercambio de 2 genes) (0 a 1).",
    "tournament_size": "Individuos que compiten en cada torneo de selección (≥ 2, ≤ población).",
    "elite_count": "Mejores individuos copiados sin cambios a la siguiente generación (< población).",
    "seed": "Opcional. Entero para reproducir resultados. Vacío = aleatorio.",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Algoritmo Genético - Problema de las N Reinas")
        self.resize(1100, 700)
        self.worker: Optional[GAWorker] = None
        self.inputs: Dict[str, QLineEdit] = {}
        self._build_ui()
        self._live_validate()

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # ---- Panel izquierdo: parámetros
        left = QVBoxLayout()
        box = QGroupBox("Parámetros")
        form = QFormLayout(box)

        int_re = QRegularExpression(r"^\d{0,10}$")
        float_re = QRegularExpression(r"^\d{0,3}([.,]\d{0,6})?$")
        float_keys = {"crossover_rate", "mutation_rate"}

        for key in LABELS:
            edit = QLineEdit(DEFAULTS[key])
            edit.setToolTip(TOOLTIPS[key])
            edit.setValidator(
                QRegularExpressionValidator(float_re if key in float_keys else int_re)
            )
            if key == "seed":
                edit.setPlaceholderText("(aleatoria)")
            edit.textChanged.connect(self._live_validate)
            self.inputs[key] = edit
            form.addRow(LABELS[key] + ":", edit)
        left.addWidget(box)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #d32f2f;")
        left.addWidget(self.error_label)

        self.btn_run = QPushButton("▶ Ejecutar")
        self.btn_stop = QPushButton("■ Detener")
        self.btn_reset = QPushButton("Restaurar valores")
        self.btn_stop.setEnabled(False)
        self.btn_run.clicked.connect(self.start_run)
        self.btn_stop.clicked.connect(self.stop_run)
        self.btn_reset.clicked.connect(self.reset_defaults)
        for b in (self.btn_run, self.btn_stop, self.btn_reset):
            left.addWidget(b)

        self.status = QLabel("Listo.")
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        left.addWidget(self.log, 1)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(330)
        layout.addWidget(left_w)

        # ---- Panel derecho: tablero y gráfica
        right = QVBoxLayout()
        self.board = BoardWidget()
        self.chart = ChartWidget()
        right.addWidget(self.board, 3)
        right.addWidget(self.chart, 1)
        layout.addLayout(right, 1)

    # ------------------------------------------------------- validación
    def _raw_values(self) -> Dict[str, str]:
        return {k: e.text() for k, e in self.inputs.items()}

    def _live_validate(self) -> bool:
        """Marca en rojo los campos inválidos y muestra el detalle. Devuelve True si todo es válido."""
        _, errors = validate_params(self._raw_values())
        for key, edit in self.inputs.items():
            if key in errors:
                edit.setStyleSheet(INVALID_STYLE)
                edit.setToolTip(f"⚠ {errors[key]}")
            else:
                edit.setStyleSheet("")
                edit.setToolTip(TOOLTIPS[key])
        self.error_label.setText(
            "\n".join(f"• {LABELS[k]}: {m}" for k, m in errors.items())
        )
        if self.worker is None or not self.worker.isRunning():
            self.btn_run.setEnabled(not errors)
        return not errors

    # --------------------------------------------------------- acciones
    def reset_defaults(self) -> None:
        for k, e in self.inputs.items():
            e.setText(DEFAULTS[k])

    def _set_running(self, running: bool) -> None:
        for e in self.inputs.values():
            e.setEnabled(not running)
        self.btn_reset.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.btn_run.setEnabled(not running)

    def start_run(self) -> None:
        params, errors = validate_params(self._raw_values())
        if errors or params is None:
            msg = "\n".join(f"• {LABELS[k]}: {m}" for k, m in errors.items())
            QMessageBox.warning(self, "Parámetros inválidos", msg)
            self.inputs[next(iter(errors))].setFocus()
            return

        self.board.clear()
        self.chart.clear()
        self.log.clear()
        self.log.appendPlainText(
            f"N={params.n}, población={params.pop_size}, generaciones={params.generations}, "
            f"cruce={params.crossover_rate}, mutación={params.mutation_rate}, "
            f"torneo={params.tournament_size}, élite={params.elite_count}, "
            f"semilla={'aleatoria' if params.seed is None else params.seed}"
        )
        self.status.setText("Ejecutando…")
        self._set_running(True)

        self.worker = GAWorker(params)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_run.connect(self._on_finished)
        self.worker.start()

    def stop_run(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.btn_stop.setEnabled(False)
            self.status.setText("Deteniendo…")

    def _on_progress(self, gen: int, best: int, avg: float, ind: list) -> None:
        self.board.set_solution(ind)
        self.chart.add_point(gen, best, avg)
        self.status.setText(f"Generación {gen} | Mejor: {best} conflictos | Promedio: {avg:.2f}")

    def _on_finished(self, gen: int, best: int, ind: list, elapsed: float, stopped: bool) -> None:
        self.board.set_solution(ind)
        if best == 0:
            result = f"✔ Solución encontrada en la generación {gen}"
        elif stopped:
            result = f"■ Detenido en la generación {gen} (mejor: {best} conflictos)"
        else:
            result = f"✘ Sin solución tras {gen} generaciones (mejor: {best} conflictos)"
        self.status.setText(f"{result}\nTiempo: {elapsed:.2f} s")
        self.log.appendPlainText(result)
        self.log.appendPlainText(f"Tiempo: {elapsed:.2f} s")
        self.log.appendPlainText(f"Individuo (fila por columna): {ind}")
        self._set_running(False)
        self._live_validate()

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(3000)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
