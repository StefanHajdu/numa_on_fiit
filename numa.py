import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit
import scipy.integrate as sint
from math import sqrt

pd.set_option("display.precision", 16)

def plot_fn(f, symbol, x_axis=(-10, 10, 100)):
    """
    Args:
        f: funkcia, ktorej koren hladame
        symbol: premenna, je to x v f(x)
        x_axis: tuple, obsahuje rozsah intervalu a pocet dielikov
    """
    f = sp.lambdify(symbol, f)
    x = np.linspace(x_axis[0], x_axis[1], x_axis[2])
    y = f(x)

    # clear current plot
    plt.clf()

    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.plot(x,y)
    plt.show()

def plot_fn_diff(f, symbol, x_axis=(-10, 10, 100)):
    """
    Args:
        f: funkcia, ktorej koren hladame
        symbol: premenna, je to x v f(x)
        x_axis: tuple, obsahuje rozsah intervalu a pocet dielikov
    """
    f = f.diff(symbol)
    f = sp.lambdify(symbol, f)
    x = np.linspace(x_axis[0], x_axis[1], x_axis[2])
    y = f(x)

    # clear current plot
    plt.clf()

    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.plot(x,y)
    plt.show()

def plot_fn_system(_vars, _funcs, x_axis=(-10, 10), y_axis=(-10, 10), bins=100):
    """
    Args:
        _vars: dict(str, int), tabulka neznamych a ich pociatocnych (x0, y0) hodnot
        _funcs: list(str), zoznam sustavy funckii v tvare f(x,y) = 0
        x_axis: tuple, obsahuje rozsah intervalu a pocet dielikov
    """
    xy_sym = list(_vars.keys())
    _funcs = sp.sympify(_funcs)
    x = np.linspace(x_axis[0], x_axis[1], bins)
    y = np.linspace(y_axis[0], y_axis[1], bins)
    f1 = sp.lambdify(xy_sym, _funcs[0])
    f2 = sp.lambdify(xy_sym, _funcs[1])
    X, Y = np.meshgrid(x,y)
    val1 = f1(X, Y)
    val2 = f2(X, Y)

    # clear current plot
    plt.clf()

    plt.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")
    plt.ylabel("y")
    plt.xlabel("x")
    plt.contour(X, Y, val1, [0])
    plt.contour(X, Y, val2, [0])
    plt.show()

def tetivova_metoda(f, symbol, x=[0, 1.0], tolerancia=10e-10, r=10):
    """
    Args:
        f: funkcia, ktorej koren hladame
        symbol: premenna, je to x v f(x)
        x: zoznam, ktory obsahuje approximaciu korenov, v prvej iteracii obsahuje bod x0 a x1
        tolerancia: float, presnost vysledku
        r: int, pocet iteracii/opakovani algoritmu
    """
    f = sp.lambdify(symbol, f)
    
    results = pd.DataFrame({'i': pd.Series(dtype='int'),
                   'x(i)': pd.Series(dtype='float'),
                   '|x(i+1)-x(i)|': pd.Series(dtype='float')})
    for i in range(1, r+1):
        x_plus_1 = x[i] - f(x[i]) * ((x[i] - x[i-1]) / (f(x[i]) - f(x[i-1])))
        relative_err = abs(x[i]-x[i-1])
        results.loc[len(results)] = [i-1, x[i-1], relative_err]
        if relative_err < tolerancia:
            break
        x.append(x_plus_1)
    
    print(results)
    print(f'\nSKUSKA: {f(x[-1]):.16f}')
    print('\n')


def newtonova_metoda(f, symbol, x=[1], tolerancia=10e-10, r=10):
    """
    Args:
        f: funkcia, ktorej koren hladame
        symbol: premenna, je to x v f(x)
        x: zoznam, ktory obsahuje approximaciu korenov, v prvej iteracii obsahuje bod x0
        tolerancia: float, presnost vysledku
        r: int, pocet iteracii/opakovani algoritmu
    """

    # zderivuj funkciu
    d_f = f.diff(symbol)
    
    f = sp.lambdify(symbol, f)
    d_f = sp.lambdify(symbol, d_f)

    results = pd.DataFrame({'i': pd.Series(dtype='int'),
                   'x(i)': pd.Series(dtype='float'),
                   '|x(i+1)-x(i)|': pd.Series(dtype='float')})

    for i in range(r):
        x_plus_1 = x[i] - (f(x[i]) / d_f(x[i]))
        x.append(x_plus_1)
        relative_err = abs(x[i]-x[i-1])
        results.loc[len(results)] = [i, x[i], relative_err]
        if relative_err < tolerancia:
            break
    
    print(results)
    print(f'\nSKUSKA: {f(x[-1]):.16f}')
    print('\n')


def newton_ralphsonova_metoda(f, symbol, k, x=[1], tolerancia=10e-10, r=10):
    """
    Args:
        f: funkcia, ktorej koren hladame
        symbol: premenna, je to x v f(x)
        k: int, nasobnost korena
        x: zoznam, ktory obsahuje approximaciu korenov, v prvej iteracii obsahuje bod x0
        tolerancia: float, presnost vysledku
        r: int, pocet iteracii/opakovani algoritmu
    """
    # zderivuj funkciu
    d_f = f.diff(symbol)
    
    d_f = sp.lambdify(symbol, d_f)
    f = sp.lambdify(symbol, f)

    results = pd.DataFrame({'i': pd.Series(dtype='int'),
                   'x(i)': pd.Series(dtype='float'),
                   '|x(i+1)-x(i)|': pd.Series(dtype='float')})

    for i in range(0, r):
        x_plus_1 = x[i] - k*(f(x[i]) / d_f(x[i]))
        x.append(x_plus_1)
        relative_err = abs(x[i]-x[i-1])
        results.loc[len(results)] = [i, x[i], relative_err]
        if relative_err < tolerancia:
            break
    
    print(results)
    print(f'\nSKUSKA: {f(x[-1]):.16f}')
    print('\n')
    


def metoda_pevneho_bodu(f, g, symbol, x=[1], tolerancia=10e-10, r=10):
    """
    Args:
        f: funkcia, ktorej koren hladame
        g: pomocna funkcia
        symbol: premenna, je to x v f(x)
        x: zoznam, ktory obsahuje approximaciu korenov, v prvej iteracii obsahuje bod x0
        tolerancia: float, presnost vysledku
        r: int, pocet iteracii/opakovani algoritmu
    """

    f = sp.lambdify(symbol, f)
    g = sp.lambdify(symbol, g)
    
    results = pd.DataFrame({'i': pd.Series(dtype='int'),
                   'x(i)': pd.Series(dtype='float'),
                   '|x(i+1)-x(i)|': pd.Series(dtype='float')})

    for i in range(0, r):
        x_plus_1 = g(x[i])
        x.append(x_plus_1)
        relative_err = abs(x[i]-x[i-1])
        results.loc[len(results)] = [i, x[i], relative_err]
        if relative_err < tolerancia:
            break
    
    print(results)
    print(f'\nSKUSKA: {f(x[-1]):.16f}')
    print('\n')

def newton_sustava_rovnic(_vars, _funcs, tolerancia=10e-16, r=10):
    """
    Args:
        _vars: dict(str, int), tabulka neznamych a ich pociatocnych (x0, y0) hodnot
        _funcs: list(str), zoznam sustavy funckii v tvare f(x,y) = 0
        tolerancia: float, presnost vysledku
        r: int, pocet iteracii/opakovani algoritmu

    """
    xy_sym = list(_vars.keys())
    xy_val = list(_vars.values())
    _funcs = sp.sympify(_funcs)
    J = sp.Matrix(_funcs).jacobian(xy_sym)
    
    J = sp.lambdify(xy_sym, J)
    f1 = sp.lambdify(xy_sym, _funcs[0])
    f2 = sp.lambdify(xy_sym, _funcs[1])

    results = pd.DataFrame({'x(i)': pd.Series(dtype='float'),
                            'y(i)': pd.Series(dtype='float'),
                            'err': pd.Series(dtype='float')})

    xy_mat = [[xy_val[0]], [xy_val[1]]]
    for i in range(r):
        xy_func = [[f1(xy_mat[0][0], xy_mat[1][0])], [f2(xy_mat[0][0], xy_mat[1][0])]]
        xy_next = np.array(xy_mat) - np.linalg.inv(J(xy_mat[0][0], xy_mat[1][0])) @ np.array(xy_func)
        
        err = max(abs(xy_next[0][0]-xy_mat[0][0]), abs(xy_next[1][0]-xy_mat[1][0]))
        results.loc[i] = [xy_mat[0][0], xy_mat[1][0], err]
        if err < tolerancia:
            break

        xy_mat[0][0] = xy_next[0][0]
        xy_mat[1][0] = xy_next[1][0]

    print(results)
    print(f'\nSKUSKA F1: {f1(xy_mat[0][0], xy_mat[1][0]):.16f}')
    print(f'SKUSKA F2: {f2(xy_mat[0][0], xy_mat[1][0]):.16f}')
    print('\n')

def chord_sustava_rovnic(_vars, _funcs, tolerancia=10e-16, r=10):
    """
    Args:
        _vars: dict(str, int), tabulka neznamych a ich pociatocnych (x0, y0) hodnot
        _funcs: list(str), zoznam sustavy funckii v tvare f(x,y) = 0
        tolerancia: float, presnost vysledku
        r: int, pocet iteracii/opakovani algoritmu

    """
    xy_sym = list(_vars.keys())
    xy_val = list(_vars.values())
    _funcs = sp.sympify(_funcs)
    J = sp.Matrix(_funcs).jacobian(xy_sym)
    
    J = sp.lambdify(xy_sym, J)
    J_fixed = np.linalg.inv(J(xy_val[0], xy_val[1]))
    f1 = sp.lambdify(xy_sym, _funcs[0])
    f2 = sp.lambdify(xy_sym, _funcs[1])

    results = pd.DataFrame({'x(i)': pd.Series(dtype='float'),
                            'y(i)': pd.Series(dtype='float'),
                            'err': pd.Series(dtype='float')})

    xy_mat = [[xy_val[0]], [xy_val[1]]]
    for i in range(r):
        xy_func = [[f1(xy_mat[0][0], xy_mat[1][0])], [f2(xy_mat[0][0], xy_mat[1][0])]]
        xy_next = np.array(xy_mat) - J_fixed @ np.array(xy_func)
        
        err = max(abs(xy_next[0][0]-xy_mat[0][0]), abs(xy_next[1][0]-xy_mat[1][0]))
        results.loc[i] = [xy_mat[0][0], xy_mat[1][0], err]
        if err < tolerancia:
            break

        xy_mat[0][0] = xy_next[0][0]
        xy_mat[1][0] = xy_next[1][0]

    print(results)
    print(f'\nSKUSKA F1: {f1(xy_mat[0][0], xy_mat[1][0]):.16f}')
    print(f'SKUSKA F2: {f2(xy_mat[0][0], xy_mat[1][0]):.16f}')
    print('\n')

def interpolacia_newtonov_polynom(x, y, plotting=True):
    """
    Args:
        x: list vektor x hodnot nameranych data
        y: list, vektor y hodnot nameranych data
        plotting: bool, urcuje ci treba nakreslit interpolaciu
    """
    def get_coeficients(x, y):
        n = len(y)
        coef = np.zeros([n, n])
        coef[:,0] = y
        
        for j in range(1,n):
            for i in range(n-j):
                coef[i][j] = \
            (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j]-x[i])
                
        return coef[0, :]

    coefs = get_coeficients(x, y)

    if plotting:
        def interpolate(coef, x, y):
            n = len(x) - 1 
            p = coef[n]
            for k in range(1,n+1):
                p = coef[n-k] + (x -x[n-k])*p
            return p
        
        y_new = interpolate(coefs, x, y)
        plt.plot(x, y_new)
        plt.plot(x, y, 'bo')
        plt.show()
    
    polynom = []
    print('Newtonov polynom (forward tvar):\n')
    for i in range(len(coefs)):
        s = f'{coefs[i]}'
        for j in range(i):
            if not i:
                break
            else:
                s += f'*(x - {x[j]})'
        
        print(f'{s} +')
        polynom.append(s)
    
    polynom = '+'.join(polynom)
    print()
    return polynom

def interpolacia_hodnota_bod(polynom, x):
    """
    Args:
        polynom: algebraicky predpis polynom
        x: bod, ktorom sa ma vypocit hodnota f(x)
    """
    polynom = sp.sympify(polynom)
    polynom = sp.lambdify(['x'], polynom)
    print(f'Hodnota v bode x={x} je {polynom(x)}')

def fit_defined_curve(x, y, method='line'):
    """
    Args:
        x: numpy.array, vektor x hodnot nameranych data
        y: numpy.array, vektor y hodnot nameranych data
        method: str, aky typ funkcie sa bude fittovat
    """
    def plot_fitted_curve(x, y, fn, **kwargs):
        x_fitted = np.linspace(np.min(x), np.max(x), 100)
        y_fitted = fn(x_fitted, **kwargs)

        # Plot
        ax = plt.axes()
        ax.scatter(x, y, label='Raw data')
        ax.plot(x_fitted, y_fitted, 'k', label='Fitted curve')
        ax.set_ylabel('y-Values')
        ax.set_xlabel('x-Values')
        ax.legend()

    def line(x, **kwargs):
        return kwargs['a']*x + kwargs['b']

    def parabola(x, **kwargs):
        return kwargs['a']*x**2 + kwargs['b']*x + kwargs['c']

    def exponential(x, **kwargs):
        return kwargs['c'] * np.exp(kwargs['a']*x)

    def fitted_curve_error(x, y, fn, **kwargs):
        fitted_y = fn(x, **kwargs)
        l2 = np.sum(np.power((y - fitted_y), 2))
        return l2
    
    params = {}
    if method == 'exp':
        popt, pcov = curve_fit(lambda x, a, c: c*np.exp(a*x), x, y)
        a = params['a'] = popt[0]
        c = params['c'] = popt[1]
        print(f'Parameters:\n{params}')
        print(f'Exp function: {c}*e^({a}x)')
        plot_fitted_curve(x, y, exponential, **params)
        print(f'\nL2 error:\n{fitted_curve_error(x, y, exponential, **params)}')
    elif method == 'parabola':
        popt, pcov = curve_fit(lambda x, a, b, c: a*x**2+b*x+c, x, y)
        a = params['a'] = popt[0]
        b = params['b'] = popt[1]
        c = params['c'] = popt[2]
        print(f'Parameters:\n{params}')
        print(f'Parabola function: {a}x^2 + {b}x + {c}')
        plot_fitted_curve(x, y, parabola, **params)
        print(f'\nL2 error:\n{fitted_curve_error(x, y, parabola, **params)}')
    elif method == 'line':
        popt, pcov = curve_fit(lambda x, a, b: a*x+b, x, y)
        a = params['a'] = popt[0]
        b = params['b'] = popt[1]
        print(f'Parameters:\n{params}')
        print(f'Line function: {a}x + {b}')
        plot_fitted_curve(x, y, line, **params)
        print(f'\nL2 error:\n{fitted_curve_error(x, y, line, **params)}')
    else:
        print(f'{method} no defined')

def all_integrals(f, a, b, h=0.1):
    """
    Args:
        f: lambda/function, funkcia, ktora sa bude ingrovat
        a: float, spodna hranica integralu
        b: flaot, horna hranica integralu
        h: float, dlzka kroku pre zlozene metody
    """
    def reference_integral(f, a, b):
        x = np.linspace(a, b, 100)
        integral = sint.trapz(f(x), x)
        return integral

    def trapz_simple(f, a, b):
        exact = reference_integral(f, a, b)
        h = (b-a)
        approx = h/2 * (f(a) + f(b))
        return approx, abs(exact - approx)

    def simpson_simple(f, a, b):
        exact = reference_integral(f, a, b)
        h = (b-a)/2
        approx = h/3 * (f(a) + 4*f((a+b)/2) + f(b))
        return approx, abs(exact - approx)

    def simpson_3_8_simple(f, a, b):
        exact = reference_integral(f, a, b)
        h = (b-a)/3
        approx = 3*h/8 * (f(a) + 3*f(a+h) + 3*f(a+2*h) + f(b))
        return approx, abs(exact - approx)

    def boole_simple(f, a, b):
        exact = reference_integral(f, a, b)
        h = (b-a)/4
        approx = 2*h/45 * (7*f(a) + 32*f(a + 1/4*(b-a)) + 12*f(a + 2/4*(b-a)) + 32*f(a+3/4*(b-a)) + 7*f(b))
        return approx, abs(exact - approx)

    def trapz_composite(f, a, b, h=0.1):
        exact = reference_integral(f, a, b)
        x = np.arange(a, b, h)
        y = f(x)
        approx = (h/2) * (f(a) + 2*sum(y[1:]) + f(b))
        return approx, abs(exact - approx)

    def simpson_composite(f, a, b, h=0.1):
        exact = reference_integral(f, a, b)
        x = np.arange(a, b, h)
        y = f(x)
        approx = (h/3) * (f(a) + 4*sum(y[1::2]) + 2*sum(y[2::2]) + f(b))
        return approx, abs(exact - approx)

    print(f'Exact value (by scipy): {reference_integral(f, a, b)}\n')

    results = pd.DataFrame({'metoda': pd.Series(dtype='int'),
                    'approx': pd.Series(dtype='float'),
                    '|exact - approx|': pd.Series(dtype='float')})

    x, err = trapz_simple(f, a, b)
    results.loc[0] = ['trapz_simple', x, err]
    x, err = simpson_simple(f, a, b)
    results.loc[1] = ['simpson_simple', x, err]
    x, err = simpson_3_8_simple(f, a, b)
    results.loc[2] = ['simpson_3_8_simple', x, err]
    x, err = boole_simple(f, a, b)
    results.loc[3] = ['boole_simple', x, err]
    x, err = trapz_composite(f, a, b, h)
    results.loc[4] = ['trapz_composite', x, err]
    x, err = simpson_composite(f, a, b, h)
    results.loc[5] = ['simpson_composite', x, err]

    print(results)

def all_integrals_approx_err(f, a, b, f_sp, symbol, h=0.1):
    """
    Args:
        f: lambda/function, funkcia, ktora sa bude ingrovat
        a: float, spodna hranica integralu
        b: float, horna hranica integralu
        h: float, dlzka kroku pre zlozene metody
    """
    def derivation(f_sp, symbol, degree):
        return f_sp.diff(symbol, degree)

    def reference_integral(f, a, b):
        x = np.linspace(a, b, 100)
        integral = sint.trapz(f(x), x)
        return integral

    def trapz_simple(f, a, b):
        exact = reference_integral(f, a, b)
        h = (b-a)
        approx = h/2 * (f(a) + f(b))
        f_d = derivation(f_sp, symbol, 2)
        max_of_abs = sqrt(sp.maximum(f_d**2, symbol, sp.Interval(a, b)))
        err_approx = (h**3/12)*max_of_abs
        return approx, abs(exact - approx), err_approx

    def simpson_simple(f, a, b):
        exact = reference_integral(f, a, b)
        h = (b-a)/2
        approx = h/3 * (f(a) + 4*f((a+b)/2) + f(b))
        f_d = derivation(f_sp, symbol, 4)
        max_of_abs = sqrt(sp.maximum(f_d**2, symbol, sp.Interval(a, b)))
        err_approx = (h**5/90)*max_of_abs
        return approx, abs(exact - approx), err_approx

    def simpson_3_8_simple(f, a, b):
        exact = reference_integral(f, a, b)
        h = (b-a)/3
        approx = 3*h/8 * (f(a) + 3*f(a+h) + 3*f(a+2*h) + f(b))
        f_d = derivation(f_sp, symbol, 5)
        max_of_abs = sqrt(sp.maximum(f_d**2, symbol, sp.Interval(a, b)))
        err_approx = 3*h**5/(80*max_of_abs)
        return approx, abs(exact - approx), err_approx

    def boole_simple(f, a, b):
        exact = reference_integral(f, a, b)
        h = (b-a)/4
        approx = 2*h/45 * (7*f(a) + 32*f(a + 1/4*(b-a)) + 12*f(a + 2/4*(b-a)) + 32*f(a+3/4*(b-a)) + 7*f(b))
        f_d = derivation(f_sp, symbol, 6)
        max_of_abs = sqrt(sp.maximum(f_d**2, symbol, sp.Interval(a, b)))
        err_approx = (8*h**7/945)*max_of_abs
        return approx, abs(exact - approx), err_approx

    def trapz_composite(f, a, b, h=0.1):
        exact = reference_integral(f, a, b)
        x = np.arange(a, b, h)
        y = f(x)
        approx = (h/2) * (f(a) + 2*sum(y[1:]) + f(b))
        f_d = derivation(f_sp, symbol, 2)
        max_of_abs = sqrt(sp.maximum(f_d**2, symbol, sp.Interval(a, b)))
        err_approx = ((b-a)*h**2/12)*max_of_abs
        return approx, abs(exact - approx), err_approx

    def simpson_composite(f, a, b, h=0.1):
        exact = reference_integral(f, a, b)
        x = np.arange(a, b, h)
        y = f(x)
        approx = (h/3) * (f(a) + 4*sum(y[1::2]) + 2*sum(y[2::2]) + f(b))
        f_d = derivation(f_sp, symbol, 4)
        max_of_abs = sqrt(sp.maximum(f_d**2, symbol, sp.Interval(a, b)))
        err_approx = ((b-a)*h**4/180)*max_of_abs
        return approx, abs(exact - approx), err_approx

    print(f'Exact value (by scipy): {reference_integral(f, a, b)}\n')

    results = pd.DataFrame({'metoda': pd.Series(dtype='int'),
                    'approx': pd.Series(dtype='float'),
                    '|exact - approx|': pd.Series(dtype='float'),
                    'err_approx': pd.Series(dtype='float')})

    x, err, err_approx = trapz_simple(f, a, b)
    results.loc[0] = ['trapz_simple', x, err, err_approx]
    x, err, err_approx = simpson_simple(f, a, b)
    results.loc[1] = ['simpson_simple', x, err, err_approx]
    x, err, err_approx = simpson_3_8_simple(f, a, b)
    results.loc[2] = ['simpson_3_8_simple', x, err, err_approx]
    x, err, err_approx = boole_simple(f, a, b)
    results.loc[3] = ['boole_simple', x, err, err_approx]
    x, err, err_approx = trapz_composite(f, a, b, h)
    results.loc[4] = ['trapz_composite', x, err, err_approx]
    x, err, err_approx = simpson_composite(f, a, b, h)
    results.loc[5] = ['simpson_composite', x, err, err_approx]

    print(results)

def partial_matrices(A):
    Diag = np.diag(np.diag(A))
    U = np.triu(A) - Diag
    L = np.tril(A) - Diag
    return Diag, U, L

def jaccobi_systems_method(A, b, r=20):
    """
    Args:
        A: np.array/matrix, stvorcova, diagonalne dominantna matica
        b: np.array/vektor, vektor pravych stran
        r: int, pocet iteracii/opakovani algoritmu
    """
    Diag, U, L = partial_matrices(A)
    H = -np.linalg.inv(Diag) @ (L+U)
    g = np.linalg.inv(Diag) @ b
    print(H)
    print()
    print(g)
    print()

    results = pd.DataFrame({'x(i)': pd.Series(dtype='str'),
                            'err': pd.Series(dtype='float')})

    x_vector_len = H.shape[0]
    x_prev = np.zeros(x_vector_len)
    for i in range(r):
        x_next = H @ x_prev + g
        err = abs(x_next[0]-x_prev[0])
        x_prev = x_next
        results.loc[i] = [str(x_next), err]
    
    print(results)
    print()

def gauss_seidl_systems_method(A, b, r=20):
    """
    Args:
        A: np.array/matrix, stvorcova, diagonalne dominantna matica
        b: np.array/vektor, vektor pravych stran
        r: int, pocet iteracii/opakovani algoritmu
    """
    Diag, U, L = partial_matrices(A)
    H = -np.linalg.inv(L+Diag) @ (U)
    g = np.linalg.inv(L+Diag) @ b
    print(H)
    print()
    print(g)
    print()

    results = pd.DataFrame({'x(i)': pd.Series(dtype='str'),
                            'err': pd.Series(dtype='float')})

    x_vector_len = H.shape[0]
    x_prev = np.zeros(x_vector_len)
    for i in range(r):
        x_next = H @ x_prev + g
        err = abs(x_next[0]-x_prev[0])
        x_prev = x_next
        results.loc[i] = [str(x_next), err]
    
    print(results)
    print()

def primitive_iterative_systems_method(H, g, r=20):
    """
    Args:
        H: np.array/matrix, stvorcova, diagonalne dominantna matica po uprave
        g: np.array/vektor, vektor pravych stran, po uprave
        r: int, pocet iteracii/opakovani algoritmu
    """
    results = pd.DataFrame({'x(i)': pd.Series(dtype='str'),
                            'err': pd.Series(dtype='float')})

    x_vector_len = H.shape[0]
    x_prev = np.zeros(x_vector_len)
    for i in range(r):
        x_next = H @ x_prev + g
        err = abs(x_next[0]-x_prev[0])
        x_prev = x_next
        results.loc[i] = [str(x_next), err]
    
    print(results)
    print()