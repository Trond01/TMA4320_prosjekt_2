# Import stuff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.linalg import norm

## Defining the necessary constants
sigma_0 = 1000
sigma   = 500
R       = 10
A_s     = 1/2*np.pi*R**2
h_CM = 4 * R / (3 * np.pi)  # avstand M - C

## ANIMATION CODE - HANDED OUT ##
'''Denne koden animerer bevegelsen til båtet, gitt at dere sender inn arrays som inneholder tidsverdier t,
skipets helningsvinkel theta, x- og y-koordinatet til skipets massesenter. Disse arraysene brukes i funksjonen
definert nederst "animate_deck_movement". Den grønne sirkelen viser posisjonen til skipets massesenter
Man kan sende inn optional argumenter, disse står beskrevet i funksjonen. For eksempel kan man sende inn et array
som inneholder lastens posisjon relativt metasenteret. Lasten vil da animeres som en rød sirkel.
'''
# M = metasenteret = midt paa dekk
# C = skipets tyngdepunkt

def init_anim():
    """ Initialises the animation.
    """

    global ax, boat, deck, last, CM, venstre_gjerde, høyre_gjerde, textbox_theory
    boat, = plt.plot([], [],
                    color="k", linewidth=1)
    deck, = plt.plot([], [], color="k", linewidth=1)
    sea_surface, = plt.plot([-R*10, R* 10], [0, 0], color='blue', linewidth=2)  # The surface
    last, = plt.plot([], [], color="r", marker="o", markersize=10)
    CM, = plt.plot([], [], color="g", marker="o", markersize=10)
    venstre_gjerde, = plt.plot([], [], color="k", marker="|", markersize=25)
    høyre_gjerde, = plt.plot([], [], color="k", marker="|", markersize=25)
    ax.set_xlim([-R*1.3, R* 1.3])
    ax.set_ylim([-R*1.1, R* 1.1])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect("equal")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textbox_theory = ax.text(0.775, 0.95, '', transform=ax.transAxes, fontsize=12,
                             verticalalignment='top', bbox=props)

    return ax, boat, deck, last, CM,venstre_gjerde,høyre_gjerde, textbox_theory

def animate(M, theta, t, x_C, y_C, s_L, gjerde=False):
    global ax, boat, deck, last, CM, venstre_gjerde, høyre_gjerde, textbox_theory
    ax.set_xlim([-R * 1.1 + np.amin(x_C), R * 1.1 + np.amax(x_C)])
    ax.set_ylim([-R*1.1, R* 1.1])
    angle_values = np.linspace(0, np.pi, 100)
    metasenter_x = x_C[M] - h_CM * np.sin(theta[M])
    metasenter_y = y_C[M] + h_CM * np.cos(theta[M])
    xs = R * np.cos(angle_values + np.pi + theta[M]) + metasenter_x
    ys = R * np.sin(angle_values + np.pi + theta[M]) + metasenter_y
    boat.set_data(xs, ys)
    deck.set_data([xs[0], xs[-1]], [ys[0], ys[-1]])
    if s_L[M] !=-42:
        last.set_data(metasenter_x + s_L[M] * np.cos(theta[M]),
                      metasenter_y + s_L[M] * np.sin(theta[M]))
    CM.set_data(x_C[M], y_C[M])
    if gjerde:
        venstre_gjerde.set_data([metasenter_x - R * np.cos(theta[M])], [metasenter_y - R * np.sin(theta[M])])
        høyre_gjerde.set_data([metasenter_x + R * np.cos(theta[M])], [metasenter_y + R * np.sin(theta[M])])
    theta_string= r'$\theta = %.2f$' % (theta[M] * 180 / np.pi) + r"$\degree$"
    time_string = '$t =  %.2f$' % (t[M])
    textbox_theory.set_text(
        theta_string + '\n' + time_string)

    M += 1
    return ax, boat, deck, last, CM, venstre_gjerde, høyre_gjerde, textbox_theory

def animate_deck_movement(t, theta, x_C, y_C, s_L=[], gjerde=False, stepsize=0.01, vis_akse_verdier=False):
    """

    :param t: Array som inneholder tidsverdiene man har beregnet \vec{w} for systemet
    :param theta: Array som inneholder utslagsvinkelen til skipet
    :param x_C: Array som inneholder massesenterets x-koordinat
    :param y_C: Array som inneholder massesenterets y-koordinat
    :param s_L: Optional array som inneholder lastens posisjon relativt massesenteret
    :param gjerde: Optional Boolean som forteller om vi skal tegne inn gjerder på skipet
    :param stepsize: Hvor lang tid som skal gå mellom hver frame
    :param vis_akse_verdier: Hvis akse-verdier vises går animasjonen litt mer hakkete, men man kan se tallverdier
    :return: Animasjon som viser dynamikken til skipet
    """
    global fig, ax
    fig, ax = plt.subplots()
    dt = t[1] - t[0]
    skips = max(int(stepsize / dt), 1)
    theta_anim = theta[::skips]
    t_anim = t[::skips]
    x_C_anim = x_C[::skips]
    y_C_anim = y_C[::skips]
    if len(s_L) == 0:
        s_L_anim = -42 * np.ones(len(theta_anim))
    else:
        s_L_anim = s_L[::skips]
    h_anim = animation.FuncAnimation(fig, animate, init_func=init_anim, frames=len(t_anim) - 1, interval=1,
                                     blit=not vis_akse_verdier,
                                     fargs=(theta_anim, t_anim, x_C_anim, y_C_anim, s_L_anim, gjerde))
    plt.show()

## 1a NEWTONS METHOD

def newton(f, df, x0, tol=1.e-8, max_iter=30, variable = "x"):
    """
    brief: Solves the equation f(x)=0 with Newtons method
    :param f: the function f(x)
    :param df: the derivative of f(x)
    :param x0: initial value
    :param tol: tolerance, if f(x)<tol we accept x
    :return: the accepted root, number of iterations
    """
    x = x0
    for k in range(max_iter):
        print(f"k ={k:3d}, {variable} = {x:18.15f}, f(x) = {f(x):10.3e}")
        fx = f(x)
        if norm(fx) < tol: # Accept solution 
            break 
        x = x - fx/df(x)   # One Newton-iteration
    return x, k+1

def f(b):
    return b - np.sin(b) - np.pi*sigma/sigma_0
def df(b):
    return 1 - np.cos(b)

beta = newton(f, df, 2, variable = "\u03B2")[0]
print(beta)

## 1b
yM0 = R*np.cos(beta/2)
yC0 = R*np.cos(beta/2) - 4*R/(3*np.pi)
yB0 = R*np.cos(beta/2) - R*4*np.sin(beta/2)**3/(3*(beta - np.sin(beta)))
yD0 = R*np.cos(beta/2) - R

## 1c
#TODO: Verdien til disse?
F_B  = 1000
h_CM = 4*R/(3*np.pi)
m    = sigma         # mass of one metre boat
I_C  = 1/2*m*R**2*(1 - 32/(9*np.pi**2))

def f_harmonic_oscillator(t, w):
    """
    brief: the derivative of the vector w = [theta, omega]
           only including the harmonic restorative buoyant force
    :param theta: current angle
    :param omega: current angular velocity
    :return: the derivative
    """
    return np.array([w[1], - F_B*h_CM/I_C*np.sin(w[0])])

## 1d
def euler_method(f, t, w, h):
    """
    brief: One step of Eulers method
    :param f: the function f(t, w)
    :param t: current t-value
    :param w: current w-value
    :param h: stepsize
    :return: the next value for tn and wn
    """
    t_next = t + h
    w_next = w + h * f(t, w)
    return t_next, w_next

def ode_solver(f, t0, t_end, w0, h, method=euler_method):
    """
    brief: Solves the ode w' = f(t, w) with method of choice
    :param f: the function f(x)
    :param t0: initial t-value
    :param t_end: the final t-value
    :param w0: initial w-value
    :param h: stepsize
    :param method: the method used in each step
    :return: the next value for tn and wn
    """
    # Create arrays to store values
    t_num = np.array([t0])
    w_num = np.array([w0])

    # Values that are updated for each iteration
    tn = t0
    wn = w0

    # Main loop
    while tn < t_end - 1.e-10:  # Buffer for truncation errors
        if t_end - tn < h:
            tn, wn = method(f, tn, wn, t_end - tn)
        else:
            tn, wn = method(f, tn, wn, h)  # Do one step

        # Add values to array
        t_num = np.append(t_num, tn)
        w_num = np.concatenate((w_num, np.array([wn])))

    return t_num, w_num

# 1d) Bruk uttrykket deres fra oppgave 1c) til ˚a løse ODE-en i ligning (13) numerisk
# med Eulers metode. Bruk initialverdiene θ(t = 0) = 20° og ω(t = 0) = 0. Eksperimenter
# med forskjellige skrittstørrelser h, og velg en passende h. Rettferdiggjør valget deres.
# Plott θ som funksjon av t fra t = 0 til t = 20 s.

t0 = 0
t_end = 20
w0 = np.array([20*np.pi/180, 0])
h = 1
t_num, y_num = ode_solver(f_harmonic_oscillator, t0, t_end, w0, h, method=euler_method)
#plt.plot(t_num, y_num)
#plt.show()

## 1e
def f_harmonic_approxilator(t, w):
    """
    brief: the derivative of the vector w = [theta, omega]
           only including the harmonic restorative buoyant force,
           now with small angle approximation
    :param theta: current angle
    :param omega: current angular velocity
    :return: the derivative
    """
    return np.array([w[1], - F_B*h_CM/I_C*w[0]])

t_num_SA, y_num_SA = ode_solver(f_harmonic_approxilator, t0, t_end, w0, h, method=euler_method)
#plt.plot(t_num, y_num)
#plt.show()


## 1f
def RK4_method(f, t, w, h):
    """
    brief: One step of RK4 method
    :param f: the function f(t, w)
    :param t: current t-value
    :param w: current w-value
    :param h: stepsize
    :return: the next value for tn and wn
    """
    k1 = f(t, w)
    k2 = f(t + h / 2, w + h * k1 / 2)
    k3 = f(t + h / 2, w + h * k2 / 2)
    k4 = f(t + h, w + h * k3)

    t_next = t + h
    w_next = w + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return t_next, w_next

## 1g
def analytic_harmonic_oscillator(time_array, theta0):
    return theta0*np.cos(np.sqrt(F_B*h_CM/I_C)*time_array)

w0 = [0.01, 0]
h = 0.1
t, w_euler = ode_solver(f_harmonic_approxilator, t0, t_end, w0, h, method = euler_method)
t, w_RK4   = ode_solver(f_harmonic_approxilator, t0, t_end, w0, h, method = RK4_method)
theta_anal = analytic_harmonic_oscillator(t, w0[0])
#plt.plot(t, w_euler)
#plt.plot(t, theta_anal, "g--")

##1h
N = 100
h_array = np.linspace(1e-2, 1e-3, N)
global_error_euler = np.zeros(N)
global_error_RK4   = np.zeros(N)
w0 = [0.01, 0]
for i in range(len(h_array)):
    h = h_array[i]
    theta_anal  = analytic_harmonic_oscillator(t_end, w0[0])
    o, w_euler = ode_solver(f_harmonic_approxilator, t0, t_end, w0, h, method=euler_method)
    # TODO: burde vi bruke 4h paa RK4?
    o, w_RK4   = ode_solver(f_harmonic_approxilator, t0, t_end, w0, 4*h, method=RK4_method)
    global_error_euler[i] = np.abs(theta_anal - w_euler[-1][0])
    global_error_RK4[i]   = np.abs(theta_anal - w_RK4[-1][0])

plt.plot(h_array, global_error_euler)
plt.title("Euler error")
plt.show()
plt.plot(h_array, global_error_RK4)
plt.title("RK4 error")
plt.show()
