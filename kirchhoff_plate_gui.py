"""Jednoduché GUI pro zadání vstupů výpočtu Kirchhoffovy desky."""

import datetime
import json
import traceback
import os
import subprocess
import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


DEFAULT_POLYGON = [
    [0.0, 0.0],
    [4.0, 0.0],
    [3.0, 3.0],
    [0.0, 3.0],
]

DEFAULT_ULOZENI = [
    [1, 1],
    [1, 0],
    [0, 0],
    [1, 1],
]

DEFAULT_LINE_PAR_X = [0.0, 3.6, 1.0]
DEFAULT_LINE_PAR_Y = [0.0, 3.0, 1.0]
LOG_FILE_NAME = "kirchhoff_gui.log"


def _setup_runtime_logging():
    """Přesměruje stdout/stderr do log souboru (užitečné pro PyInstaller --windowed)."""
    if getattr(_setup_runtime_logging, "_initialized", False):
        return

    log_path = Path.cwd() / LOG_FILE_NAME
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = log_file
    sys.stderr = log_file

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\n=== Start aplikace: {timestamp} ===")
    print(f"Python executable: {sys.executable}")
    print(f"CWD: {Path.cwd()}")
    print(f"argv: {sys.argv}")

    def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            if original_stderr is not None:
                original_stderr.write("KeyboardInterrupt\n")
            return

        print("Nezachycená výjimka:")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=log_file)
        log_file.flush()

        if original_stderr is not None:
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=original_stderr)

        app = QApplication.instance()
        if app is not None:
            QMessageBox.critical(
                None,
                "Nezachycená chyba",
                f"Nastala neočekávaná chyba:\n{exc_value}\n\n"
                f"Podrobnosti jsou uloženy v logu:\n{log_path}",
            )
            app.processEvents()
        else:
            if original_stderr is not None:
                original_stderr.write(
                    f"Nastala neočekávaná chyba: {exc_value}\n"
                    f"Podrobnosti jsou uložené v logu: {log_path}\n"
                )

    sys.excepthook = _log_uncaught_exception
    _setup_runtime_logging._initialized = True


def _build_default_edges_text():
    rows = ["# x,y,w,phi_n"]
    for point, bc in zip(DEFAULT_POLYGON, DEFAULT_ULOZENI):
        rows.append(f"{point[0]},{point[1]},{bc[0]},{bc[1]}")
    return "\n".join(rows)


class KirchhoffWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kirchhoffova deska – zadání vstupů")

        central = QWidget(self)
        layout = QVBoxLayout(central)
        form = QFormLayout()

        self.q_input = self._create_double(15.0, 0.0, 1e6, 2)
        self.lc_input = self._create_double(0.06, 0.005, 1.0, 3)
        self.d_input = self._create_double(0.2, 0.01, 5.0, 3)
        self.e_input = self._create_double(35e6, 1e3, 1e9, 0)
        self.nu_input = self._create_double(0.25, 0.0, 0.49, 3)
        self.n_pts_input = QSpinBox()
        self.n_pts_input.setRange(10, 2000)
        self.n_pts_input.setValue(100)

        self.line_x_start_input = self._create_double(DEFAULT_LINE_PAR_X[0], -1e4, 1e4, 3)
        self.line_x_stop_input = self._create_double(DEFAULT_LINE_PAR_X[1], -1e4, 1e4, 3)
        self.line_x_const_input = self._create_double(DEFAULT_LINE_PAR_X[2], -1e4, 1e4, 3)

        self.line_y_start_input = self._create_double(DEFAULT_LINE_PAR_Y[0], -1e4, 1e4, 3)
        self.line_y_stop_input = self._create_double(DEFAULT_LINE_PAR_Y[1], -1e4, 1e4, 3)
        self.line_y_const_input = self._create_double(DEFAULT_LINE_PAR_Y[2], -1e4, 1e4, 3)

        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("např. deska_01")
        self.project_name_input.setText("projekt_01")

        form.addRow("Jméno projektu", self.project_name_input)
        form.addRow("Spojité zatížení q [kN/m²]", self.q_input)
        form.addRow("Tloušťka desky d [m]", self.d_input)
        form.addRow("Modul pružnosti E [kPa]", self.e_input)
        form.addRow("Poissonův poměr nu [-]", self.nu_input)
        form.addRow("Délka strany prvku lc [m]", self.lc_input)
        

        self.edges_input = QTextEdit()
        self.edges_input.setPlaceholderText("x,y,w,phi_n")
        self.edges_input.setPlainText(_build_default_edges_text())

        layout.addLayout(form)
        layout.addWidget(QLabel("Geometrie + okrajové podmínky (1 řádek = 1 vrchol + podmínka následující hrany):"))
        layout.addWidget(QLabel("Formát: x, y, w=0?, phi_n=0?   | řádky s komentářem začínají #"))
        layout.addWidget(self.edges_input)

        line_form = QFormLayout()
        line_form.addRow("Linie X: x_start", self.line_x_start_input)
        line_form.addRow("Linie X: x_stop", self.line_x_stop_input)
        line_form.addRow("Linie X: y_konstantní", self.line_x_const_input)
        line_form.addRow("Linie Y: y_start", self.line_y_start_input)
        line_form.addRow("Linie Y: y_stop", self.line_y_stop_input)
        line_form.addRow("Linie Y: x_konstantní", self.line_y_const_input)
        line_form.addRow("Počet bodů liniových grafů", self.n_pts_input)

        layout.addWidget(QLabel("Parametry pro liniové grafy:"))
        layout.addLayout(line_form)

        self.status = QLabel("Nastavte hodnoty a spusťte výpočet.")

        self.run_button = QPushButton("Spustit výpočet")
        self.run_button.clicked.connect(self.run_solver)
        self.mesh_button = QPushButton("Generovat síť")
        self.mesh_button.clicked.connect(self.preview_mesh)
        self.save_button = QPushButton("Uložit zadání (JSON)")
        self.save_button.clicked.connect(self.save_input_to_json)
        self.load_button = QPushButton("Načíst zadání (JSON)")
        self.load_button.clicked.connect(self.load_input_from_json)
        self._solver_process = None
        self._mesh_process = None

        layout.addWidget(self.save_button)
        layout.addWidget(self.load_button)
        layout.addWidget(self.mesh_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.status)
        self.setCentralWidget(central)

    @staticmethod
    def _create_double(value, minimum, maximum, decimals):
        box = QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setSingleStep(10 ** (-decimals) if decimals > 0 else 1000.0)
        return box

    def _parse_edges_text(self):
        polygon = []
        ulozeni = []
        lines = self.edges_input.toPlainText().splitlines()

        for index, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                QMessageBox.critical(
                    self,
                    "Neplatný řádek",
                    f"Řádek {index}: očekáván formát x,y,w,phi_n.",
                )
                return None

            try:
                x = float(parts[0])
                y = float(parts[1])
                w = int(parts[2])
                phi_n = int(parts[3])
            except ValueError:
                QMessageBox.critical(
                    self,
                    "Neplatná data",
                    f"Řádek {index}: x,y musí být čísla a w,phi_n musí být 0/1.",
                )
                return None

            if w not in (0, 1) or phi_n not in (0, 1):
                QMessageBox.critical(
                    self,
                    "Neplatná data",
                    f"Řádek {index}: w a phi_n mohou být pouze 0 nebo 1.",
                )
                return None

            polygon.append([x, y])
            ulozeni.append([w, phi_n])

        if len(polygon) < 3:
            QMessageBox.critical(self, "Neplatný polygon", "Je potřeba zadat alespoň 3 vrcholy.")
            return None

        return polygon, ulozeni

    def _build_line_params(self):
        line_par_x = [
            self.line_x_start_input.value(),
            self.line_x_stop_input.value(),
            self.line_x_const_input.value(),
        ]
        line_par_y = [
            self.line_y_start_input.value(),
            self.line_y_stop_input.value(),
            self.line_y_const_input.value(),
        ]
        return line_par_x, line_par_y

    def _collect_input_data(self):
        line_par_x, line_par_y = self._build_line_params()
        return {
            "project_name": self.project_name_input.text().strip(),
            "q": self.q_input.value(),
            "lc": self.lc_input.value(),
            "d": self.d_input.value(),
            "E": self.e_input.value(),
            "nu": self.nu_input.value(),
            "n_query_pts": self.n_pts_input.value(),
            "edges_text": self.edges_input.toPlainText(),
            "line_par_x": line_par_x,
            "line_par_y": line_par_y,
        }

    def _apply_input_data(self, data):
        self.project_name_input.setText(str(data.get("project_name", "")))
        self.q_input.setValue(float(data["q"]))
        self.lc_input.setValue(float(data["lc"]))
        self.d_input.setValue(float(data["d"]))
        self.e_input.setValue(float(data["E"]))
        self.nu_input.setValue(float(data["nu"]))
        self.n_pts_input.setValue(int(data["n_query_pts"]))

        self.edges_input.setPlainText(str(data["edges_text"]))

        line_par_x = data["line_par_x"]
        line_par_y = data["line_par_y"]
        if len(line_par_x) != 3 or len(line_par_y) != 3:
            raise ValueError("line_par_x a line_par_y musí mít přesně 3 prvky.")

        self.line_x_start_input.setValue(float(line_par_x[0]))
        self.line_x_stop_input.setValue(float(line_par_x[1]))
        self.line_x_const_input.setValue(float(line_par_x[2]))

        self.line_y_start_input.setValue(float(line_par_y[0]))
        self.line_y_stop_input.setValue(float(line_par_y[1]))
        self.line_y_const_input.setValue(float(line_par_y[2]))

    def save_input_to_json(self):
        project_name = self.project_name_input.text().strip()
        default_path = "kirchhoff_input.json"
        if project_name:
            default_path = str((Path.cwd() / project_name / "kirchhoff_input.json").resolve())

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Uložit zadání",
            default_path,
            "JSON files (*.json);;All files (*)",
        )
        if not file_path:
            return

        payload = self._collect_input_data()
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
        except OSError as error:
            QMessageBox.critical(self, "Chyba uložení", f"Nepodařilo se uložit soubor: {error}")
            return

        self.status.setText(f"Zadání bylo uloženo do: {file_path}")

    def load_input_from_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Načíst zadání",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            self._apply_input_data(data)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            QMessageBox.critical(
                self,
                "Chyba načtení",
                f"Soubor se nepodařilo načíst nebo má neplatný formát:\n{error}",
            )
            return

        self.status.setText(f"Zadání bylo načteno ze souboru: {file_path}")

    def run_solver(self):
        if self._solver_process is not None and self._solver_process.poll() is None:
            self.status.setText("Výpočet už běží. Počkejte na dokončení aktuálního běhu nebo zavřete všechna jeho grafická okna.")
            return

        parsed = self._parse_edges_text()
        if parsed is None:
            return

        polygon, ulozeni = parsed
        line_par_x, line_par_y = self._build_line_params()

        project_name = self.project_name_input.text().strip()
        if not project_name:
            QMessageBox.critical(self, "Chybí jméno projektu", "Vyplňte prosím pole 'Jméno projektu'.")
            return

        output_dir = Path.cwd() / project_name
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            QMessageBox.critical(
                self,
                "Chyba vytvoření složky",
                f"Nepodařilo se vytvořit složku projektu '{project_name}': {error}",
            )
            return

        env = self._build_solver_env(polygon, ulozeni, line_par_x, line_par_y, output_dir)

        self.status.setText("Výpočet běží…")
        self.repaint()

        process = self._launch_solver_process(env)
        if process is None:
            self.status.setText("Výpočet selhal – nepodařilo se spustit výpočetní proces.")
            return

        self._solver_process = process
        self.status.setText(
            f"Výpočet byl spuštěn na pozadí. Výstupy budou ve složce: {output_dir}"
        )

    def preview_mesh(self):
        if self._mesh_process is not None and self._mesh_process.poll() is None:
            self.status.setText("Náhled sítě už běží. Zavřete okno sítě a zkuste to znovu.")
            return

        parsed = self._parse_edges_text()
        if parsed is None:
            return

        polygon, ulozeni = parsed
        line_par_x, line_par_y = self._build_line_params()

        output_dir = Path.cwd() / "_mesh_preview"
        env = self._build_solver_env(polygon, ulozeni, line_par_x, line_par_y, output_dir)
        env["KIRCHHOFF_PREVIEW_MESH"] = "1"

        self.status.setText("Generuji síť…")
        self.repaint()

        process = self._launch_solver_process(env)
        if process is None:
            self.status.setText("Náhled sítě selhal – nepodařilo se spustit výpočetní proces.")
            return

        self._mesh_process = process
        self.status.setText("Otevřen náhled sítě. Po kontrole zavřete okno a stiskněte 'Spustit výpočet'.")

    def _build_solver_env(self, polygon, ulozeni, line_par_x, line_par_y, output_dir):
        env = os.environ.copy()
        env["KIRCHHOFF_Q"] = str(self.q_input.value())
        env["KIRCHHOFF_LC"] = str(self.lc_input.value())
        env["KIRCHHOFF_D"] = str(self.d_input.value())
        env["KIRCHHOFF_E"] = str(self.e_input.value())
        env["KIRCHHOFF_NU"] = str(self.nu_input.value())
        env["KIRCHHOFF_N_QUERY_PTS"] = str(self.n_pts_input.value())
        env["KIRCHHOFF_POLYGON"] = json.dumps(polygon)
        env["KIRCHHOFF_ULOZENI"] = json.dumps(ulozeni)
        env["KIRCHHOFF_LINE_PAR_X"] = json.dumps(line_par_x)
        env["KIRCHHOFF_LINE_PAR_Y"] = json.dumps(line_par_y)
        env["KIRCHHOFF_INPUT_FILE"] = str(output_dir / "kirchhoff_input.json")
        env["KIRCHHOFF_PLOTS_DIR"] = str(output_dir / "kirchhoff_plots")
        env["KIRCHHOFF_REPORT_FILE"] = str(output_dir / "kirchhoff_report.pdf")
        return env

    def _launch_solver_process(self, env):
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--run-solver"]
        else:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-solver"]

        try:
            # Důležité: proces nespouštíme jako "detached".
            # U některých prostředí (hlavně Windows + matplotlib) to bránilo otevření grafických oken.
            return subprocess.Popen(cmd, env=env)
        except OSError as error:
            QMessageBox.critical(self, "Chyba spuštění", f"Nepodařilo se spustit výpočet: {error}")
            return None


def run_solver_entrypoint():
    import kirchhoff_plate
    if os.getenv("KIRCHHOFF_PREVIEW_MESH") == "1":
        kirchhoff_plate.preview_mesh_main()
    else:
        kirchhoff_plate.main()

    

def main():
    _setup_runtime_logging()

    if "--run-solver" in sys.argv:
        run_solver_entrypoint()
        return

    app = QApplication(sys.argv)
    window = KirchhoffWindow()
    window.resize(760, 820)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

