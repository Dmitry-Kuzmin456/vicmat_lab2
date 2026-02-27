import FreeSimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json
from models import EQUATIONS, SYSTEMS
from solvers import EquationSolvers, SystemSolvers


class LabApp:
    def __init__(self):
        sg.theme("Default1")
        self.window = sg.Window("Лабораторная работа №2", self._layout(), finalize=True)
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, self.window["-CANVAS-"].TKCanvas)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

    def _layout(self):
        eq_names = [e["name"] for e in EQUATIONS]
        sys_names = [s["name"] for s in SYSTEMS]
        return [
            [
                sg.TabGroup(
                    [
                        [
                            sg.Tab(
                                "Уравнения",
                                [
                                    [
                                        sg.Combo(
                                            eq_names,
                                            k="-EQ-",
                                            readonly=True,
                                            default_value=eq_names[0],
                                        )
                                    ],
                                    [
                                        sg.Combo(
                                            ["Хорд", "Секущих", "МПИ"],
                                            k="-EQ_M-",
                                            readonly=True,
                                            default_value="Хорд",
                                        )
                                    ],
                                    [
                                        sg.Text("a:"),
                                        sg.In("1", k="-A-", s=5),
                                        sg.Text("b:"),
                                        sg.In("2", k="-B-", s=5),
                                        sg.Text("eps:"),
                                        sg.In("0.01", k="-E1-", s=7),
                                    ],
                                    [
                                        sg.Button("Решить уравнение"),
                                        sg.Button("Загрузить JSON"),
                                    ],
                                ],
                            ),
                            sg.Tab(
                                "Системы",
                                [
                                    [
                                        sg.Combo(
                                            sys_names,
                                            k="-SYS-",
                                            readonly=True,
                                            default_value=sys_names[0],
                                        )
                                    ],
                                    [sg.Text("Метод: Ньютона (для систем)")],
                                    [
                                        sg.Text("x0:"),
                                        sg.In("0.5", k="-X0-", s=5),
                                        sg.Text("y0:"),
                                        sg.In("0.5", k="-Y0-", s=5),
                                        sg.Text("eps:"),
                                        sg.In("0.01", k="-E2-", s=7),
                                    ],
                                    [sg.Button("Решить систему")],
                                ],
                            ),
                        ]
                    ]
                )
            ],
            [sg.Multiline(k="-OUT-", s=(60, 6), disabled=True)],
            [sg.Canvas(key="-CANVAS-")],
            [sg.Button("Выход")],
        ]

    def plot(self, mode, data_idx, res=None, a=None, b=None):
        self.ax.clear()
        if mode == "eq":
            f = EQUATIONS[data_idx]["f"]
            x = np.linspace(a - 0.5, b + 0.5, 200)
            self.ax.plot(x, [f(v) for v in x], label="f(x)")
            self.ax.axhline(0, color="black", lw=1)
            if res:
                self.ax.plot(res, f(res), "ro")
        else:
            data = SYSTEMS[data_idx]
            x, y = np.meshgrid(np.linspace(-2, 3, 50), np.linspace(-2, 3, 50))
            self.ax.contour(x, y, data["plot"][0](x, y), [0], colors="blue")
            self.ax.contour(x, y, data["plot"][1](x, y), [0], colors="red")
            if res is not None:
                self.ax.plot(res[0], res[1], "go")
        self.ax.grid(True)
        self.canvas.draw()

    def load_json(self):
        path = sg.popup_get_file(
            "Выберите JSON файл", file_types=(("JSON Files", "*.json"),)
        )
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)
            if data["mode"] == "equation":
                self.window["-EQ-"].update(EQUATIONS[data["index"]]["name"])
                self.window["-EQ_M-"].update(data["method"])
                self.window["-A-"].update(data["a"])
                self.window["-B-"].update(data["b"])
                self.window["-E1-"].update(data["eps"])
            else:
                self.window["-SYS-"].update(SYSTEMS[data["index"]]["name"])
                self.window["-X0-"].update(data["x0"])
                self.window["-Y0-"].update(data["y0"])
                self.window["-E2-"].update(data["eps"])

    def run(self):
        while True:
            ev, vals = self.window.read()
            if ev in (None, "Выход"):
                break
            if ev == "Загрузить JSON":
                self.load_json()

            if ev == "Решить уравнение":
                idx = [e["name"] for e in EQUATIONS].index(vals["-EQ-"])
                f_d = EQUATIONS[idx]
                a, b, eps = float(vals["-A-"]), float(vals["-B-"]), float(vals["-E1-"])
                if f_d["f"](a) * f_d["f"](b) > 0:
                    sg.popup_error("На концах интервала функция одного знака!")
                    continue
                m = vals["-EQ_M-"]
                if m == "Хорд":
                    res, it = EquationSolvers.chord(f_d["f"], f_d["ddf"], a, b, eps)
                elif m == "Секущих":
                    res, it = EquationSolvers.secant(f_d["f"], a, b, eps)
                else:
                    res, it = EquationSolvers.simple_iteration(
                        f_d["f"], f_d["df"], a, b, eps
                    )

                if res is not None:
                    self.window["-OUT-"].update(
                        f"Уравнение. Метод: {m}\nКорень: {res:.6f}\nf(x): {f_d['f'](res):.8f}\nИтераций: {it}"
                    )
                    self.plot("eq", idx, res, a, b)
                else:
                    self.window["-OUT-"].update(it)

            if ev == "Решить систему":
                idx = [s["name"] for s in SYSTEMS].index(vals["-SYS-"])
                s_d = SYSTEMS[idx]
                x0, y0, eps = (
                    float(vals["-X0-"]),
                    float(vals["-Y0-"]),
                    float(vals["-E2-"]),
                )
                res, it, errs, msg = SystemSolvers.newton(s_d, x0, y0, eps)
                self.window["-OUT-"].update(
                    f"{msg}\nx: {res[0]:.4f}, y: {res[1]:.4f}\nИтераций: {it}\nПогрешность: {errs[-1] if errs else 'N/A'}"
                )
                self.plot("sys", idx, res)

        self.window.close()
