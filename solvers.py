import numpy as np


class EquationSolvers:
    @staticmethod
    def chord(f, ddf, a, b, eps):
        # Выбор неподвижного конца x0: f(x0)*f''(x0) > 0
        x0 = a if f(a) * ddf(a) > 0 else b
        x = b if x0 == a else a
        it = 0
        while True:
            it += 1
            x_next = x - f(x) * (x - x0) / (f(x) - f(x0))
            if abs(x_next - x) < eps:
                return x_next, it
            x = x_next
            if it > 500:
                break
        return x, it

    @staticmethod
    def secant(f, a, b, eps):
        x_prev, x = a, b
        it = 0
        while True:
            it += 1
            x_next = x - f(x) * (x - x_prev) / (f(x) - f(x_prev))
            if abs(x_next - x) < eps:
                return x_next, it
            x_prev, x = x, x_next
            if it > 500:
                break
        return x, it

    @staticmethod
    def simple_iteration(f, df, a, b, eps):
        vals = np.linspace(a, b, 100)
        derivs = [df(v) for v in vals]
        max_df = max(abs(d) for d in derivs)
        lmbd = ((-1 / max_df) + 1) if df((a + b) / 2) > 0 else ((1 / max_df) + 1)
        q = max(abs(1 + lmbd * d) for d in derivs)
        if q >= 1:
            return None, f"Не сходится (q={q:.2f})"

        x = (a + b) / 2
        it = 0
        while True:
            it += 1
            x_next = x + lmbd * f(x)
            if abs(x_next - x) < eps:
                return x_next, it
            x = x_next
            if it > 500:
                break
        return x, it


class SystemSolvers:
    @staticmethod
    def newton(sys_data, x0, y0, eps):
        it = 0
        curr = np.array([x0, y0], dtype=float)
        errors = []
        while it < 100:
            it += 1
            f_v = np.array(sys_data["f_vec"](*curr))
            jac = np.array(sys_data["jacobian"](*curr))
            try:
                delta = np.linalg.solve(jac, -f_v)
            except np.linalg.LinAlgError:
                return curr, it, [], "Ошибка: Определитель = 0"
            curr += delta
            err = [abs(delta[0]), abs(delta[1])]
            errors.append(err)
            if max(err) < eps:
                return curr, it, errors, "Метод Ньютона сошелся"
        return curr, it, errors, "Лимит итераций"
