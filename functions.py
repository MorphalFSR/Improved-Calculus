import numpy as np
import math

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
OPERATORS = ['+', '-', '*', '/', '^']
UNDEFINED = 'undefined'


def strip_parentheses(string):
    return string.strip("[").strip("]").strip("(").strip(")")


def breakdown_str(func_str):

    func_str = func_str.replace(" ", "")
    if len(func_str) == 1:
        return [func_str]

    i_open = 0
    open_par = 0
    i_open_func = 0
    open_func = 0
    parentheses = []
    for i in range(len(func_str)):

        if open_func == 0 and open_par == 0 and func_str[i] in OPERATORS:
            if len(parentheses) > 0:
                if func_str[parentheses[-1][0]] in OPERATORS:
                    parentheses.append((parentheses[-1][0] + 1, i))
            parentheses.append((i, i + 1))
            continue

        if open_func == 0:
            if func_str[i] == "[":
                if open_par == 0:
                    i_open = i
                open_par += 1

            elif func_str[i] == "]":
                open_par -= 1
                assert open_par >= 0, "Invalid parentheses."
                if open_par == 0:
                    parentheses.append((i_open+1, i))

        if func_str[i] == "(":
            if open_func == 0:
                i_open_func = i
            open_func += 1

        elif func_str[i] == ")":
            func = None
            open_func -= 1
            assert open_func >= 0, "Invalid parentheses."
            if open_func == 0:
                for j in range(1, i_open_func):
                    if func_str[i_open_func - j] not in ALPHABET:
                        func = func_str[i_open_func - j + 1:i_open_func]
                        break
                func = func or func_str[:i_open_func]
                parentheses.append((i_open_func - len(func), i, func))

    if len(parentheses) == 0:
        return [func_str]
    else:
        if parentheses[0][0] > 0:
            parentheses = [(0, parentheses[0][0] if parentheses[0][0] > 0 else 0)] + parentheses
        if parentheses[-1][1] < len(func_str) - 1 or func_str[-1] not in "[]()":
            parentheses = parentheses + [(parentheses[-1][1], len(func_str))]

        breakdown = []
        for t in parentheses:
            if len(t) == 2:
                i, j = t
                breakdown.append(breakdown_str(strip_parentheses(func_str[i:j])))
            else:
                i, j, func = t
                breakdown.append((func, breakdown_str(strip_parentheses(func_str[i + len(func) + 1:j]))))

        return breakdown


class Function:

    def __init__(self, params, func_str, derivatives=None, eval_func=None, display=None):
        self.params = params
        self.display = display
        self.derivatives = derivatives or {param: None for param in params}
        if eval_func:
            self.eval_func = eval_func
        else:
            self.breakdown = breakdown_str(func_str)
            self.base_func = self.setup_func(self.breakdown)
            self.eval_func = self.base_func.evaluate

    def set_eval(self, func):
        self.eval_func = func

    def setup_func(self, breakdown):
        if len(breakdown) == 1:
            part = breakdown[0]
            if type(part) == tuple:
                return KNOWN_FUNCTIONS[part[0]](self.params, self.setup_func(part[1]))
            elif part not in OPERATORS:
                if part in self.params:
                    return identity(part)
                else:
                    return constant(eval(part))
            elif part in OPERATORS:
                return part

        funcs = [self.setup_func(part) for part in breakdown]
        while len(funcs) > 1:
            op_index = sorted(range(1, len(funcs), 2), key=lambda op: OPERATORS.index(funcs[op]))[-1]
            op_func = OPERATOR_FUNCTIONS[funcs[op_index]](self.params, funcs[op_index-1], funcs[op_index+1])
            funcs = funcs[:op_index-1] + [op_func] + funcs[op_index+2:]

        return funcs[0]

    def evaluate(self, params, continuation=False):
        assert type(params) == dict, "Parameters must be dictionary of parameter names and values."
        try:
            value = self.eval_func(params)
            if type(value) == str:
                raise ArithmeticError
            else:
                return value
        except Exception as e:
            if continuation:
                right_limit = limit(self, params, {param: 1 for param in params})
                left_limit = limit(self, params, {param: -1 for param in params})
                if right_limit == left_limit:
                    return right_limit
                else:
                    return UNDEFINED
            return UNDEFINED

    def derivative(self, param):
        if param not in self.params:
            return constant(0)
        elif self.derivatives[param]:
            return self.derivatives[param]()
        else:
            return self.base_func.derivative(param)

    def __str__(self):
        if self.display:
            return self.display()
        else:
            return str(self.base_func)

    def __add__(self, other):
        return add(self.params, self, other)

    def __sub__(self, other):
        return sub(self.params, self, other)

    def __mul__(self, other):
        return mul(self.params, self, other)

    def __floordiv__(self, other):
        return div(self.params, self, other)

    def __pow__(self, other):
        return power(self.params, self, other)


# Elementary Functions


def constant(v):
    return Function([],
                    func_str=str(v),
                    derivatives=dict(),
                    eval_func=lambda p, c=False: v,
                    display=lambda: str(v))


def identity(var):
    return Function([var],
                    func_str=var,
                    derivatives={var: lambda: constant(1)},
                    eval_func=lambda p, c=False: p[var],
                    display=lambda: var)


def add(params, f1, f2):
    if str(f1) == '0':
        return f2
    elif str(f2) == '0':
        return f1
    return Function(params,
                    func_str=None,
                    derivatives={param: lambda: f1.derivative(param) + f2.derivative(param) for param in params},
                    eval_func=lambda p, c=False: f1.evaluate(p, c) + f2.evaluate(p, c),
                    display=lambda: f"{f1.display()} + {f2.display()}")


def sub(params, f1, f2):
    if str(f2) == '0':
        return f1
    return Function(params,
                    func_str=None,
                    derivatives={param: lambda: f1.derivative(param) - f2.derivative(param) for param in params},
                    eval_func=lambda p, c=False: f1.evaluate(p, c) - f2.evaluate(p, c),
                    display=lambda: f"{f1.display()} - {f2.display()}")


def mul(params, f1, f2):
    if str(f1) == '0' or str(f2) == '0':
        return constant(0)
    elif str(f1) == '1':
        return f2
    elif str(f2) == '1':
        return f1
    return Function(params,
                    func_str=None,
                    derivatives={param: lambda: f1.derivative(param) * f2 + f1 * f2.derivative(param) for param in params},
                    eval_func=lambda p, c=False: f1.evaluate(p, c) * f2.evaluate(p, c),
                    display=lambda: f"{f1.display()} * {f2.display()}")


def div(params, f1, f2, skip_der=False):
    return Function(params,
                    func_str=None,
                    derivatives={param: lambda: (f1.derivative(param) * f2 - f1 * f2.derivative(param)) // (f2 ** constant(2)) for param in params},
                    eval_func=lambda p, c=False: f1.evaluate(p, c) / f2.evaluate(p, c),
                    display=lambda: f"{f1.display()} / {f2.display()}")


def power(params, f1, f2):
    return Function(params,
                    func_str=None,
                    derivatives={param: lambda: ((f2 // f1) * f1.derivative(param) + ln(params, f1) * f2.derivative(param)) * (f1 ** f2) for param in params},
                    eval_func=lambda p, c=False: f1.evaluate(p, c) ** f2.evaluate(p, c),
                    display=lambda: f"{f1.display()} ^ {f2.display()}")


def exp(params, f):
    return Function(params,
                    func_str=None,
                    derivatives={param: lambda: f.derivative(param) * exp(params, f) for param in params},
                    eval_func=lambda p, c=False: np.exp(f.evaluate(p, c)),
                    display=lambda: f"exp({f.display()})")


def ln(params, f):
    return Function(params,
                    func_str=None,
                    derivatives={param: lambda: f.derivative(param) // f for param in params},
                    eval_func=lambda p, c=False: np.log(f.evaluate(p, c)),
                    display=lambda: f"ln({f.display()})")


OPERATOR_FUNCTIONS = {'+': add, '-': sub, '*': mul, '/': div, '^': power}
KNOWN_FUNCTIONS = {'exp': exp}


def limit(f, params, dparams=None):
    dparams = dparams or {param: 1 for param in params}
    zeroes = {param: 0 for param in params}
    value = f.evaluate({param: params[param] + dparams[param] for param in params})
    while dparams != zeroes:
        dparams = {param: dparams[param] / 2 for param in dparams}
        new_value = f.evaluate({param: params[param] + dparams[param] for param in params})
        if new_value == UNDEFINED or math.isnan(new_value):
            return value

        value = new_value
    return value

