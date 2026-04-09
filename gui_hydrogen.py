import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spe
import scipy.constants as const
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, PowerNorm, Normalize
import warnings
import time
import tkinter as tk
from tkinter import ttk, messagebox
from concurrent.futures import ThreadPoolExecutor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    from scipy.interpolate import RectBivariateSpline
    INTERP_AVAILABLE = True
except ImportError:
    INTERP_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["Times New Roman", "Arial"]

class HydrogenWavefunction:
    def __init__(self):
        self.a0 = const.value("Bohr radius")

    def classical_turning_points(self, n, l):
        epsilon = l / n
        r_min = n**2 * self.a0 * (1 - np.sqrt(1 - epsilon**2))
        r_max = n**2 * self.a0 * (1 + np.sqrt(1 - epsilon**2))
        return r_min, r_max

    def log_factorial(self, n):
        return spe.loggamma(n + 1)

    def laguerre_wkb_optimized(self, rho, n, alpha):
        k = n + (alpha + 1) / 2
        x_tp = 4 * k - 2 * alpha - 2
        zeta = (rho - x_tp) / (2 * k**(1/3))
        zeta = np.where(np.abs(zeta) < 1e-12, 1e-12, zeta)
        result = np.zeros_like(rho)
        mask_small = np.abs(zeta) < 0.1
        mask_positive = (zeta >= 0.1)
        mask_negative = (zeta <= -0.1)
        if np.any(mask_small):
            zeta_small = zeta[mask_small]
            result[mask_small] = 0.355028 * (1 - 0.729*zeta_small + 0.265*zeta_small**2)
        if np.any(mask_positive):
            zeta_pos = zeta[mask_positive]
            result[mask_positive] = (np.exp(-2/3 * zeta_pos**1.5) / (2 * np.sqrt(np.pi) * zeta_pos**0.25))
        if np.any(mask_negative):
            zeta_neg = zeta[mask_negative]
            result[mask_negative] = (np.sin(2/3 * (-zeta_neg)**1.5 - np.pi/4) / (np.sqrt(np.pi) * (-zeta_neg)**0.25))
        norm = 2 * np.exp(rho/2) / (np.pi**0.5 * k**(1/6))
        return norm * result

    def R_nl(self, r, n, l):
        r_safe = np.where(r < 1e-100 * self.a0, 1e-100 * self.a0, r)
        try:
            if n - l - 1 > 100:
                log_fact1 = (n - l - 1) * np.log(n - l - 1) - (n - l - 1)
                log_fact2 = (n + l) * np.log(n + l) - (n + l)
            else:
                log_fact1 = self.log_factorial(n - l - 1)
                log_fact2 = self.log_factorial(n + l)
            log_coeff = 0.5 * (3 * np.log(2.0/(self.a0*n)) + log_fact1 - np.log(2.0*n) - log_fact2)
            rho = 2.0 * r_safe / (self.a0 * n)
            log_exp = -r_safe/(self.a0*n) + l * np.log(np.where(rho > 0, rho, 1e-100))
            if n - l - 1 <= 500:
                try:
                    laguerre = spe.assoc_laguerre(rho, n-l-1, 2*l+1)
                    log_laguerre = np.log(np.abs(laguerre) + 1e-100)
                except:
                    log_laguerre = np.log(np.abs(self.laguerre_wkb_optimized(rho, n-l-1, 2*l+1)) + 1e-100)
            else:
                log_laguerre = np.log(np.abs(self.laguerre_wkb_optimized(rho, n-l-1, 2*l+1)) + 1e-100)
            log_result = log_coeff + log_exp + log_laguerre
            log_result = np.clip(log_result, -600, 600)
            result = np.exp(log_result)
        except Exception:
            result = np.zeros_like(r)
        result = np.where(np.isnan(result), 0, result)
        result = np.where(np.isinf(result), 0, result)
        result = np.where(r < 1e-100 * self.a0, np.where(l == 0, result, 0), result)
        return result

    def sph_harm_compat(self, m, l, phi, theta):
        try:
            return spe.sph_harm(m, l, phi, theta)
        except AttributeError:
            return spe.sph_harm(m, l, phi, theta)

    def create_extended_grid(self, n, l, grid_size=400):
        r_min, r_max = self.classical_turning_points(n, l)
        if n < 5:
            scale_factor = 3.0
        else:
            scale_factor = 1.8
        scale_extended = scale_factor * r_max
        x = np.linspace(-scale_extended, scale_extended, grid_size)
        z = np.linspace(-scale_extended, scale_extended, grid_size)
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)
        return X, Y, Z, scale_extended

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = np.arccos(z / np.where(r > 0, r, 1))
    phi = np.arctan2(y, x)
    return r, theta, phi

def get_cut_mask(points, mode, which_parts=None):
    x, y, z = points[:,0], points[:,1], points[:,2]
    if mode == 'none':
        return np.ones(len(points), dtype=bool)
    if mode == 'half_x':
        return x >= 0
    elif mode == 'half_y':
        return y >= 0
    elif mode == 'half_z':
        return z >= 0
    if mode == 'quadrant':
        quadrant = np.zeros(len(points), dtype=int)
        quadrant[(x>=0) & (y>=0)] = 0
        quadrant[(x<0) & (y>=0)] = 1
        quadrant[(x<0) & (y<0)] = 2
        quadrant[(x>=0) & (y<0)] = 3
        if which_parts is None:
            which_parts = [0,1,2,3]
        return np.isin(quadrant, which_parts)
    if mode == 'octant':
        octant = ((x>=0).astype(int) << 2) + ((y>=0).astype(int) << 1) + (z>=0).astype(int)
        if which_parts is None:
            which_parts = list(range(8))
        return np.isin(octant, which_parts)
    if mode == 'hexadecant':
        phi = np.arctan2(y, x)
        theta = np.arccos(np.clip(z / np.linalg.norm(points, axis=1), -1, 1))
        phi_bin = ((phi + np.pi) // (np.pi/2)).astype(int) % 4
        theta_bin = (theta // (np.pi/4)).astype(int)
        theta_bin = np.clip(theta_bin, 0, 3)
        hex_index = phi_bin * 4 + theta_bin
        if which_parts is None:
            which_parts = list(range(16))
        return np.isin(hex_index, which_parts)
    return np.ones(len(points), dtype=bool)

class HydrogenGUI:
    def __init__(self, master):
        self.master = master
        master.title("Hydrogen Wavefunction Visualizer")
        master.geometry("900x700")
        self.calc = HydrogenWavefunction()
        self.a0 = self.calc.a0
        self.X = None
        self.Z = None
        self.scale = None
        self.density = None
        self.shell_density_norm = None
        self.log_density = None
        self.x_classical = None
        self.z_classical = None
        self.r_min = None
        self.r_max = None
        self.plot_options = {i: {} for i in range(1, 14)}
        self.set_default_options()
        self.create_widgets()

    def set_default_options(self):
        self.plot_options[1] = {'show_classical_boundary': True, 'show_box': True, 'show_labels': True}
        self.plot_options[2] = {'show_box': True, 'show_labels': True}
        self.plot_options[3] = {'show_contour_labels': True, 'show_box': True, 'show_axis_labels': True}
        self.plot_options[4] = {'quality': 'ultra', 'show_box': True, 'show_labels': True}
        self.plot_options[5] = {'show_classical': True, 'show_box': True, 'show_labels': True}
        self.plot_options[6] = {'show_box': True, 'show_labels': True}
        self.plot_options[7] = {'show_white_contour': True, 'show_classical_boundary': True, 'show_info_box': True, 'show_box': True, 'show_labels': True}
        self.plot_options[8] = {'show_classical_boundary': True, 'show_box': True, 'show_labels': True}
        self.plot_options[9] = {'quality': 'ultra', 'show_box': True, 'show_labels': True}
        self.plot_options[10] = {'show_box': True, 'show_labels': True}
        self.plot_options[11] = {'quality': 'ultra', 'show_box': True, 'show_labels': True}
        self.plot_options[12] = {'quality': 'ultra', 'show_box': True, 'show_labels': True}
        self.plot_options[13] = {'resolution': 80, 'num_cores': 4, 'cut_mode': 'none', 'which_parts': None, 'show_box': True, 'show_labels': True}

    def create_widgets(self):
        self.main_frame = ttk.Frame(self.master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.create_input_frame()
        self.create_plots_frame()
        self.create_bottom_frame()

    def create_input_frame(self):
        input_frame = ttk.LabelFrame(self.main_frame, text="Quantum Numbers", padding="5")
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text="n (principal):").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.n_var = tk.IntVar(value=1)
        ttk.Spinbox(input_frame, from_=1, to=200, textvariable=self.n_var, width=10).grid(row=0, column=1, padx=5)
        ttk.Label(input_frame, text="l (azimuthal):").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.l_var = tk.IntVar(value=0)
        ttk.Spinbox(input_frame, from_=0, to=199, textvariable=self.l_var, width=10).grid(row=0, column=3, padx=5)
        ttk.Label(input_frame, text="m (magnetic):").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.m_var = tk.IntVar(value=0)
        ttk.Spinbox(input_frame, from_=-199, to=199, textvariable=self.m_var, width=10).grid(row=0, column=5, padx=5)
        ttk.Label(input_frame, text="Grid Size:").grid(row=0, column=6, sticky=tk.W, padx=5)
        self.grid_var = tk.IntVar(value=400)
        ttk.Spinbox(input_frame, from_=50, to=1000, textvariable=self.grid_var, width=10).grid(row=0, column=7, padx=5)
        ttk.Button(input_frame, text="Start!", command=self.start_calculation).grid(row=0, column=8, padx=10)

    def create_plots_frame(self):
        self.plots_frame = ttk.LabelFrame(self.main_frame, text="Select Plots", padding="5")
        self.plots_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        canvas = tk.Canvas(self.plots_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(self.plots_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.plot_vars = {}
        self.plot_option_btns = {}
        plot_names = [
            "1. Linear Scale (|ψ|²)",
            "2. Logarithmic Scale (log10|ψ|²)",
            "3. Contour Plot",
            "4. 3D Surface (|ψ|²)",
            "5. Radial Density (r²R²)",
            "6. Angular Distribution (|Y|²)",
            "7. Enhanced 2D Map",
            "8. 2D Shell Density (4πr²|ψ|²)",
            "9. 3D Shell Density (4πr²|ψ|²)",
            "10. Angular + Radial",
            "11. Half-Cut 3D Shell Density",
            "12. Half-Cut 3D |ψ|²",
            "13. Realistic 3D Wavefunction (scatter)"
        ]
        for i, name in enumerate(plot_names, start=1):
            frame = ttk.Frame(self.scrollable_frame)
            frame.pack(fill=tk.X, pady=2)
            var = tk.BooleanVar(value=False)
            self.plot_vars[i] = var
            cb = ttk.Checkbutton(frame, text=name, variable=var)
            cb.pack(side=tk.LEFT)
            btn = ttk.Button(frame, text="Options", command=lambda idx=i: self.edit_plot_options(idx))
            btn.pack(side=tk.RIGHT, padx=5)
            self.plot_option_btns[i] = btn

    def create_bottom_frame(self):
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="Generate Selected", command=self.generate_plots).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Recover", command=self.recover).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Quit", command=self.master.quit).pack(side=tk.RIGHT, padx=5)
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=2)

    def recover(self):
        self.main_frame.destroy()
        self.__init__(self.master)

    def edit_plot_options(self, plot_num):
        opt = self.plot_options[plot_num]
        win = tk.Toplevel(self.master)
        win.title(f"Plot {plot_num} Options")
        if plot_num == 1:
            self.add_bool_option(win, "Show classical boundary", opt, 'show_classical_boundary')
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels (title, axis labels, colorbar)", opt, 'show_labels')
        elif plot_num == 2:
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels", opt, 'show_labels')
        elif plot_num == 3:
            self.add_bool_option(win, "Show contour labels", opt, 'show_contour_labels')
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show axis labels and title", opt, 'show_axis_labels')
        elif plot_num == 4:
            self.add_quality_option(win, opt)
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels", opt, 'show_labels')
        elif plot_num == 5:
            self.add_bool_option(win, "Show classical distribution and boundaries", opt, 'show_classical')
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels", opt, 'show_labels')
        elif plot_num == 6:
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show title and colorbar", opt, 'show_labels')
        elif plot_num == 7:
            self.add_bool_option(win, "Show white contour lines", opt, 'show_white_contour')
            self.add_bool_option(win, "Show classical boundary", opt, 'show_classical_boundary')
            self.add_bool_option(win, "Show information box", opt, 'show_info_box')
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels", opt, 'show_labels')
        elif plot_num == 8:
            self.add_bool_option(win, "Show classical boundary", opt, 'show_classical_boundary')
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels", opt, 'show_labels')
        elif plot_num == 9:
            self.add_quality_option(win, opt)
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels", opt, 'show_labels')
        elif plot_num == 10:
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels", opt, 'show_labels')
        elif plot_num == 11:
            self.add_quality_option(win, opt)
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels", opt, 'show_labels')
        elif plot_num == 12:
            self.add_quality_option(win, opt)
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels", opt, 'show_labels')
        elif plot_num == 13:
            self.add_resolution_option(win, opt)
            self.add_cores_option(win, opt)
            self.add_cut_mode_option(win, opt)
            self.add_bool_option(win, "Show axes box", opt, 'show_box')
            self.add_bool_option(win, "Show labels", opt, 'show_labels')
        ttk.Button(win, text="OK", command=win.destroy).pack(pady=5)

    def add_bool_option(self, parent, label, opt_dict, key):
        var = tk.BooleanVar(value=opt_dict.get(key, True))
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.pack(anchor=tk.W, padx=10, pady=2)
        var.trace('w', lambda *args, k=key, v=var: opt_dict.update({k: v.get()}))

    def add_quality_option(self, parent, opt_dict):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(frame, text="Quality:").pack(side=tk.LEFT)
        quality_var = tk.StringVar(value=opt_dict.get('quality', 'ultra'))
        ttk.Radiobutton(frame, text="default", variable=quality_var, value='default').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(frame, text="high", variable=quality_var, value='high').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(frame, text="ultra", variable=quality_var, value='ultra').pack(side=tk.LEFT, padx=5)
        quality_var.trace('w', lambda *args: opt_dict.update({'quality': quality_var.get()}))

    def add_resolution_option(self, parent, opt_dict):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(frame, text="Resolution:").pack(side=tk.LEFT)
        res_var = tk.IntVar(value=opt_dict.get('resolution', 80))
        ttk.Spinbox(frame, from_=20, to=200, textvariable=res_var, width=6).pack(side=tk.LEFT, padx=5)
        res_var.trace('w', lambda *args: opt_dict.update({'resolution': res_var.get()}))

    def add_cores_option(self, parent, opt_dict):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(frame, text="CPU cores:").pack(side=tk.LEFT)
        cores_var = tk.IntVar(value=opt_dict.get('num_cores', 4))
        ttk.Spinbox(frame, from_=1, to=8, textvariable=cores_var, width=6).pack(side=tk.LEFT, padx=5)
        cores_var.trace('w', lambda *args: opt_dict.update({'num_cores': cores_var.get()}))

    def add_cut_mode_option(self, parent, opt_dict):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=2)
        ttk.Label(frame, text="Cut mode:").pack(side=tk.LEFT)
        cut_var = tk.StringVar(value=opt_dict.get('cut_mode', 'none'))
        modes = ['none', 'half_x', 'half_y', 'half_z', 'quadrant', 'octant', 'hexadecant']
        ttk.Combobox(frame, textvariable=cut_var, values=modes, width=12).pack(side=tk.LEFT, padx=5)
        cut_var.trace('w', lambda *args: opt_dict.update({'cut_mode': cut_var.get()}))

    def start_calculation(self):
        try:
            n = self.n_var.get()
            l = self.l_var.get()
            m = self.m_var.get()
            grid_size = self.grid_var.get()
            if n < 1 or l < 0 or l >= n or m < -l or m > l:
                messagebox.showerror("Error", "Invalid quantum numbers")
                return
            self.status_var.set("Calculating wavefunction...")
            self.master.update()
            self.compute_wavefunction(n, l, m, grid_size)
            self.status_var.set("Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Computation failed: {e}")
            self.status_var.set("Ready")

    def compute_wavefunction(self, n, l, m, grid_size):
        self.r_min, self.r_max = self.calc.classical_turning_points(n, l)
        X, Y, Z, scale = self.calc.create_extended_grid(n, l, grid_size)
        r, theta, phi = cart2sph(X, Y, Z)
        radial = self.calc.R_nl(r, n, l)
        angular = self.calc.sph_harm_compat(m, l, phi, theta)
        psi = radial * angular
        density = np.abs(psi)**2
        max_density = np.max(density)
        if max_density > 0:
            density = density / max_density
        r_safe = np.where(r < 1e-100 * self.a0, 1e-100 * self.a0, r)
        shell_density = 4 * np.pi * r_safe**2 * density
        max_shell = np.max(shell_density)
        if max_shell > 0:
            shell_density_norm = shell_density / max_shell
        else:
            shell_density_norm = shell_density
        log_density = np.log10(density + 1e-100)
        theta_circle = np.linspace(0, 2*np.pi, 200)
        x_classical = self.r_max * np.cos(theta_circle)
        z_classical = self.r_max * np.sin(theta_circle)
        self.X = X
        self.Z = Z
        self.scale = scale
        self.density = density
        self.shell_density_norm = shell_density_norm
        self.log_density = log_density
        self.x_classical = x_classical
        self.z_classical = z_classical
        self.n = n
        self.l = l
        self.m = m
        self.grid_size = grid_size

    def generate_plots(self):
        selected = [i for i, var in self.plot_vars.items() if var.get()]
        if not selected:
            messagebox.showinfo("Info", "No plots selected")
            return
        for plot_num in selected:
            self.status_var.set(f"Generating plot {plot_num}...")
            self.master.update()
            try:
                if plot_num == 8:
                    self.show_plot8_window()
                else:
                    self.generate_single_plot(plot_num)
            except Exception as e:
                messagebox.showerror("Error", f"Plot {plot_num} failed: {e}")
        self.status_var.set("Ready")

    def generate_single_plot(self, plot_num):
        opt = self.plot_options[plot_num]
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        if plot_num == 1:
            self.plot_linear_scale(fig, opt)
        elif plot_num == 2:
            self.plot_log_scale(fig, opt)
        elif plot_num == 3:
            self.plot_contour(fig, opt)
        elif plot_num == 4:
            self.plot_3d_surface(fig, opt)
        elif plot_num == 5:
            self.plot_radial_density(fig, opt)
        elif plot_num == 6:
            self.plot_angular_distribution(fig, opt)
        elif plot_num == 7:
            self.plot_enhanced_2d(fig, opt)
        elif plot_num == 9:
            self.plot_shell_3d(fig, opt)
        elif plot_num == 10:
            self.plot_angular_radial(fig, opt)
        elif plot_num == 11:
            self.plot_shell_3d_half(fig, opt)
        elif plot_num == 12:
            self.plot_density_3d_half(fig, opt)
        elif plot_num == 13:
            self.plot_3d_realistic(fig, opt)
        win = tk.Toplevel(self.master)
        win.title(f"Plot {plot_num}")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_plot8_window(self):
        opt = self.plot_options[8]
        win = tk.Toplevel(self.master)
        win.title("Plot 8 - Shell Density Cross-Section")
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        restart_btn = ttk.Button(win, text="ReStart", command=lambda: self.restart_plot8(win, fig, canvas, opt))
        restart_btn.pack(side=tk.BOTTOM, pady=5)
        self.update_plot8(fig, canvas, opt)

    def restart_plot8(self, win, fig, canvas, opt):
        self.update_plot8(fig, canvas, opt)

    def update_plot8(self, fig, canvas, opt):
        fig.clear()
        dialog = tk.Toplevel(self.master)
        dialog.title("Plot 8 Options")
        color_var = tk.StringVar(value="1")
        section_var = tk.StringVar(value="X")
        ttk.Label(dialog, text="Color style:").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(dialog, text="Enhanced (like plot7)", variable=color_var, value="1").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(dialog, text="Original inferno", variable=color_var, value="2").pack(anchor=tk.W, padx=20)
        ttk.Label(dialog, text="Cross-section:").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(dialog, text="X=0", variable=section_var, value="X").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(dialog, text="Y=0", variable=section_var, value="Y").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(dialog, text="Z=0", variable=section_var, value="Z").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(dialog, text="XYZ (x+y+z=0)", variable=section_var, value="XYZ").pack(anchor=tk.W, padx=20)
        show_boundary_var = tk.BooleanVar(value=opt.get('show_classical_boundary', True))
        show_box_var = tk.BooleanVar(value=opt.get('show_box', True))
        show_labels_var = tk.BooleanVar(value=opt.get('show_labels', True))
        ttk.Checkbutton(dialog, text="Show classical boundary", variable=show_boundary_var).pack(anchor=tk.W, padx=10)
        ttk.Checkbutton(dialog, text="Show axes box", variable=show_box_var).pack(anchor=tk.W, padx=10)
        ttk.Checkbutton(dialog, text="Show labels", variable=show_labels_var).pack(anchor=tk.W, padx=10)
        def apply():
            dialog.destroy()
            cmap = 'inferno' if color_var.get() == '2' else 'nipy_spectral'
            use_enhanced_norm = (color_var.get() == '1')
            section = section_var.get()
            show_boundary = show_boundary_var.get()
            show_box = show_box_var.get()
            show_labels = show_labels_var.get()
            self.draw_plot8(fig, canvas, cmap, use_enhanced_norm, section, show_boundary, show_box, show_labels)
        ttk.Button(dialog, text="Generate", command=apply).pack(pady=5)

    def draw_plot8(self, fig, canvas, cmap, use_enhanced_norm, section, show_boundary, show_box, show_labels):
        fig.clear()
        ax = fig.add_subplot(111)
        plane_map = {'X': 'x', 'Y': 'y', 'Z': 'z', 'XYZ': 'xyz'}
        plane = plane_map[section]
        shell_slice, extent, xlabel, ylabel, _, _ = self.compute_cross_section(
            plane, self.grid_size, self.scale, self.calc, self.n, self.l, self.m, self.a0)
        if np.all(np.isnan(shell_slice)):
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
            canvas.draw()
            return
        if use_enhanced_norm:
            if self.n >= 20:
                valid = shell_slice[~np.isnan(shell_slice)]
                vmin = max(np.min(valid[valid > 0]), 1e-6) if np.any(valid > 0) else 1e-6
                norm = LogNorm(vmin=vmin, vmax=1)
            else:
                valid = shell_slice[~np.isnan(shell_slice)]
                if len(valid) == 0:
                    norm = PowerNorm(gamma=0.3, vmin=0, vmax=1)
                else:
                    vmin = valid.min()
                    vmax = valid.max()
                    if vmin == vmax:
                        if vmin == 0:
                            vmin, vmax = 0, 1
                        else:
                            vmin = vmin * 0.9
                            vmax = vmax * 1.1
                    norm = PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)
        else:
            valid = shell_slice[~np.isnan(shell_slice)]
            if len(valid) == 0:
                norm = Normalize(vmin=0, vmax=1)
            else:
                vmin = valid.min()
                vmax = valid.max()
                if vmin == vmax:
                    if vmin == 0:
                        vmin, vmax = 0, 1
                    else:
                        vmin = vmin * 0.9
                        vmax = vmax * 1.1
                norm = Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(shell_slice, extent=extent, origin='lower', cmap=cmap, norm=norm, aspect='equal')
        if show_labels:
            title = f'Shell Density on {section}=0 Plane' if section != 'XYZ' else 'Shell Density on x+y+z=0 Plane'
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        if show_boundary and section != 'XYZ':
            theta = np.linspace(0, 2*np.pi, 200)
            if section == 'X':
                y_circ = self.r_max * np.cos(theta)
                z_circ = self.r_max * np.sin(theta)
                ax.plot(y_circ, z_circ, 'w--', lw=2, alpha=0.8, label='Classical Boundary')
            elif section == 'Y':
                x_circ = self.r_max * np.cos(theta)
                z_circ = self.r_max * np.sin(theta)
                ax.plot(x_circ, z_circ, 'w--', lw=2, alpha=0.8, label='Classical Boundary')
            elif section == 'Z':
                x_circ = self.r_max * np.cos(theta)
                y_circ = self.r_max * np.sin(theta)
                ax.plot(x_circ, y_circ, 'w--', lw=2, alpha=0.8, label='Classical Boundary')
            if show_labels:
                ax.legend()
        if not show_box:
            ax.set_axis_off()
        if show_labels:
            fig.colorbar(im, ax=ax, label=r'$4\pi r^2 |\psi|^2$ (norm)')
        fig.tight_layout()
        canvas.draw()

    def compute_cross_section(self, plane, grid_size, scale, calc, n, l, m, a0):
        if plane == 'x':
            y = np.linspace(-scale, scale, grid_size)
            z = np.linspace(-scale, scale, grid_size)
            Y, Z = np.meshgrid(y, z)
            X = np.zeros_like(Y)
            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            extent = [-scale, scale, -scale, scale]
            xlabel, ylabel = 'Y (m)', 'Z (m)'
            grid1, grid2 = Y, Z
        elif plane == 'y':
            x = np.linspace(-scale, scale, grid_size)
            z = np.linspace(-scale, scale, grid_size)
            X, Z = np.meshgrid(x, z)
            Y = np.zeros_like(X)
            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            extent = [-scale, scale, -scale, scale]
            xlabel, ylabel = 'X (m)', 'Z (m)'
            grid1, grid2 = X, Z
        elif plane == 'z':
            x = np.linspace(-scale, scale, grid_size)
            y = np.linspace(-scale, scale, grid_size)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            extent = [-scale, scale, -scale, scale]
            xlabel, ylabel = 'X (m)', 'Y (m)'
            grid1, grid2 = X, Y
        else:
            u_vec = np.array([1, -1, 0]) / np.sqrt(2)
            v_vec = np.array([1, 1, -2]) / np.sqrt(6)
            u = np.linspace(-scale*1.5, scale*1.5, grid_size)
            v = np.linspace(-scale*1.5, scale*1.5, grid_size)
            U, V = np.meshgrid(u, v)
            X = U * u_vec[0] + V * v_vec[0]
            Y = U * u_vec[1] + V * v_vec[1]
            Z = U * u_vec[2] + V * v_vec[2]
            mask = (np.abs(X) <= scale) & (np.abs(Y) <= scale) & (np.abs(Z) <= scale)
            points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            extent = [u.min(), u.max(), v.min(), v.max()]
            xlabel, ylabel = 'u (m)', 'v (m)'
            grid1, grid2 = U, V
        r = np.sqrt(points[:,0]**2 + points[:,1]**2 + points[:,2]**2)
        theta = np.arccos(np.clip(points[:,2] / (r + 1e-100), -1, 1))
        phi = np.arctan2(points[:,1], points[:,0])
        radial = calc.R_nl(r, n, l)
        angular = calc.sph_harm_compat(m, l, phi, theta)
        psi = radial * angular
        dens = np.abs(psi)**2
        max_dens = np.max(dens)
        if max_dens > 0:
            dens_norm = dens / max_dens
        else:
            dens_norm = dens
        r_safe = np.where(r < 1e-100*a0, 1e-100*a0, r)
        shell = 4 * np.pi * r_safe**2 * dens_norm
        shell_grid = shell.reshape(grid1.shape)
        if plane == 'xyz':
            shell_grid[~mask] = np.nan
        max_shell = np.nanmax(shell_grid)
        if max_shell > 0:
            shell_grid = shell_grid / max_shell
        return shell_grid, extent, xlabel, ylabel, grid1, grid2

    def plot_linear_scale(self, fig, opt):
        ax = fig.add_subplot(111)
        im = ax.imshow(self.density, extent=[-self.scale, self.scale, -self.scale, self.scale],
                       origin='lower', cmap='viridis', aspect='equal')
        if opt.get('show_labels', True):
            ax.set_title('Linear Scale')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
        if opt.get('show_classical_boundary', True):
            ax.plot(self.x_classical, self.z_classical, 'r--', lw=2, alpha=0.8, label='Classical Boundary')
            if opt.get('show_labels', True):
                ax.legend()
        if not opt.get('show_box', True):
            ax.set_axis_off()
        if opt.get('show_labels', True):
            fig.colorbar(im, ax=ax, label='Probability Density (normalized)')

    def plot_log_scale(self, fig, opt):
        ax = fig.add_subplot(111)
        im = ax.imshow(self.log_density, extent=[-self.scale, self.scale, -self.scale, self.scale],
                       origin='lower', cmap='plasma', aspect='equal')
        if opt.get('show_labels', True):
            ax.set_title('Logarithmic Scale')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
        if not opt.get('show_box', True):
            ax.set_axis_off()
        if opt.get('show_labels', True):
            fig.colorbar(im, ax=ax, label='log10(Probability Density)')

    def plot_contour(self, fig, opt):
        ax = fig.add_subplot(111)
        max_dens = np.max(self.density)
        if max_dens > 0:
            num_levels = 30 if self.n >= 5 else 15
            levels = np.logspace(np.log10(max_dens*1e-5), np.log10(max_dens), num_levels)
            try:
                contour = ax.contour(self.X, self.Z, self.density, levels=levels, cmap='hot', linewidths=1.5)
                if opt.get('show_contour_labels', True):
                    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1e')
            except:
                contour = ax.contour(self.X, self.Z, self.density, levels=num_levels, cmap='hot', linewidths=1.5)
                if opt.get('show_contour_labels', True):
                    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1e')
        if opt.get('show_axis_labels', True):
            ax.set_title('Contour Plot')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
        ax.set_xlim(-self.scale, self.scale)
        ax.set_ylim(-self.scale, self.scale)
        if opt.get('show_axis_labels', True):
            ax.grid(True, alpha=0.3)
        if not opt.get('show_box', True):
            ax.set_axis_off()

    def plot_3d_surface(self, fig, opt):
        ax = fig.add_subplot(111, projection='3d')
        quality = opt.get('quality', 'ultra')
        if quality == 'default':
            stride = max(1, self.X.shape[0] // 80)
            X_plot, Z_plot, dens_plot = self.X[::stride,::stride], self.Z[::stride,::stride], self.density[::stride,::stride]
        elif quality == 'high':
            stride = max(1, self.X.shape[0] // 200)
            X_plot, Z_plot, dens_plot = self.X[::stride,::stride], self.Z[::stride,::stride], self.density[::stride,::stride]
        else:
            if not INTERP_AVAILABLE:
                stride = max(1, self.X.shape[0] // 200)
                X_plot, Z_plot, dens_plot = self.X[::stride,::stride], self.Z[::stride,::stride], self.density[::stride,::stride]
            else:
                x = self.X[0,:]; z = self.Z[:,0]
                interp = RectBivariateSpline(z, x, self.density)
                n_new = 2000
                x_new = np.linspace(x.min(), x.max(), n_new)
                z_new = np.linspace(z.min(), z.max(), n_new)
                X_new, Z_new = np.meshgrid(x_new, z_new)
                dens_new = interp(z_new, x_new)
                X_plot, Z_plot, dens_plot = X_new, Z_new, dens_new
        surf = ax.plot_surface(X_plot, Z_plot, dens_plot, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
        if opt.get('show_labels', True):
            ax.set_title(r'3D Surface ($|\psi|^2$)')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Probability Density')
        if not opt.get('show_box', True):
            ax.set_axis_off()
        if opt.get('show_labels', True):
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=r'$|\psi|^2$ (normalized)')

    def plot_radial_density(self, fig, opt):
        ax = fig.add_subplot(111)
        r_vals = np.linspace(0, self.scale, 2000)
        R_vals = self.calc.R_nl(r_vals, self.n, self.l)
        quantum = R_vals**2 * r_vals**2
        ax.semilogy(r_vals/self.a0, quantum + 1e-20, 'b-', lw=2, label='Quantum')

        if opt.get('show_classical', True):
            r_min, r_max = self.r_min, self.r_max
            if self.l == self.n - 1:
       
                r0 = (r_min + r_max) / 2        
                ax.axvline(r0/self.a0, color='red', ls='-', lw=2, alpha=0.7, label='Classical (δ)')
            else:
               
                eps = 1e-9 * (r_max - r_min)
                r_classical = np.linspace(r_min + eps, r_max - eps, 2000)
                classical = 1.0 / (np.pi * np.sqrt((r_max - r_classical) * (r_classical - r_min)))
             
                ax.semilogy(r_classical/self.a0, classical + 1e-20, 'r-', lw=2, alpha=0.7, label='Classical')

      
            ax.axvline(r_min/self.a0, color='red', ls='--', alpha=0.5)
            ax.axvline(r_max/self.a0, color='red', ls='--', alpha=0.5)

        if opt.get('show_labels', True):
            ax.set_xlabel(r'$r / a_0$')
            ax.set_ylabel('Radial Probability Density')
            ax.set_title(f'Radial Density for $n={self.n}$, $l={self.l}$')
            ax.legend()
            ax.grid(True, alpha=0.3)
        if not opt.get('show_box', True):
            ax.set_axis_off()

    def plot_angular_distribution(self, fig, opt):
        ax = fig.add_subplot(111, projection='polar')
        theta_vals = np.linspace(0, 2*np.pi, 500)
        phi_vals = np.linspace(0, np.pi, 500)
        Theta, Phi = np.meshgrid(theta_vals, phi_vals)
        Y_lm = np.abs(self.calc.sph_harm_compat(self.m, self.l, Theta, Phi))**2
        contour = ax.contourf(Theta, Phi, Y_lm, 50, cmap='cool')
        if opt.get('show_labels', True):
            ax.set_title(f'Angular Distribution: $l={self.l}$, $m={self.m}$')
        if not opt.get('show_box', True):
            ax.set_axis_off()
        if opt.get('show_labels', True):
            fig.colorbar(contour, ax=ax, label=r'$|Y|^2$')

    def plot_enhanced_2d(self, fig, opt):
        ax = fig.add_subplot(111)
        if self.n >= 20:
            norm = LogNorm(vmin=np.max(self.density)*1e-6, vmax=np.max(self.density))
        else:
            norm = PowerNorm(gamma=0.3, vmin=0, vmax=1)
        im = ax.imshow(self.density, extent=[-self.scale, self.scale, -self.scale, self.scale],
                       origin='lower', cmap='nipy_spectral', norm=norm, aspect='equal')
        if opt.get('show_white_contour', True):
            levels = np.logspace(np.log10(np.max(self.density)*1e-6), np.log10(np.max(self.density)), 20)
            CS = ax.contour(self.X, self.Z, self.density, levels=levels, colors='white', alpha=0.7, linewidths=0.8)
            if opt.get('show_labels', True):
                ax.clabel(CS, inline=True, fontsize=8, fmt='%1.0e')
        if opt.get('show_classical_boundary', True):
            ax.plot(self.x_classical, self.z_classical, 'w--', lw=2.5, alpha=0.9, label='Classical Boundary')
            if opt.get('show_labels', True):
                ax.legend()
        if opt.get('show_labels', True):
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
            ax.set_title(f'Hydrogen Atom State ($n={self.n}$, $l={self.l}$, $m={self.m}$)', fontsize=16, fontweight='bold')
        if opt.get('show_info_box', True) and opt.get('show_labels', True):
            info = f'$n={self.n}$, $l={self.l}$, $m={self.m}$\nClassical radius: {self.r_max/self.a0:.0f} $a_0$\nBohr radius: {self.a0:.2e} m\nCapture range: {self.scale/self.a0:.0f} $a_0$'
            ax.text(0.02, 0.98, info, transform=ax.transAxes, va='top', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        if not opt.get('show_box', True):
            ax.set_axis_off()
        if opt.get('show_labels', True):
            fig.colorbar(im, ax=ax, label='Probability Density (normalized)')

    def plot_shell_3d(self, fig, opt):
        ax = fig.add_subplot(111, projection='3d')
        quality = opt.get('quality', 'ultra')
        if quality == 'default':
            stride = max(1, self.X.shape[0] // 80)
            X_plot, Z_plot, shell_plot = self.X[::stride,::stride], self.Z[::stride,::stride], self.shell_density_norm[::stride,::stride]
        elif quality == 'high':
            stride = max(1, self.X.shape[0] // 200)
            X_plot, Z_plot, shell_plot = self.X[::stride,::stride], self.Z[::stride,::stride], self.shell_density_norm[::stride,::stride]
        else:
            if not INTERP_AVAILABLE:
                stride = max(1, self.X.shape[0] // 200)
                X_plot, Z_plot, shell_plot = self.X[::stride,::stride], self.Z[::stride,::stride], self.shell_density_norm[::stride,::stride]
            else:
                x = self.X[0,:]; z = self.Z[:,0]
                interp = RectBivariateSpline(z, x, self.shell_density_norm)
                n_new = 2000
                x_new = np.linspace(x.min(), x.max(), n_new)
                z_new = np.linspace(z.min(), z.max(), n_new)
                X_new, Z_new = np.meshgrid(x_new, z_new)
                shell_new = interp(z_new, x_new)
                X_plot, Z_plot, shell_plot = X_new, Z_new, shell_new
        surf = ax.plot_surface(X_plot, Z_plot, shell_plot, cmap='inferno', alpha=0.8, linewidth=0, antialiased=True)
        if opt.get('show_labels', True):
            ax.set_title(r'3D Surface with Shell Factor ($4\pi r^2 |\psi|^2$)')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Shell Density')
        if not opt.get('show_box', True):
            ax.set_axis_off()
        if opt.get('show_labels', True):
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='Shell Density (normalized)')

    def plot_angular_radial(self, fig, opt):
        ax1 = fig.add_subplot(211, projection='polar')
        theta_vals = np.linspace(0, 2*np.pi, 500)
        phi_vals = np.linspace(0, np.pi, 500)
        Theta, Phi = np.meshgrid(theta_vals, phi_vals)
        Y_lm = np.abs(self.calc.sph_harm_compat(self.m, self.l, Theta, Phi))**2
        contour = ax1.contourf(Theta, Phi, Y_lm, 50, cmap='cool')
        if opt.get('show_labels', True):
            ax1.set_title(f'Angular Part: $|Y_{{{self.l}}}^{{{self.m}}}|^2$')
        if not opt.get('show_box', True):
            ax1.set_axis_off()
        if opt.get('show_labels', True):
            fig.colorbar(contour, ax=ax1, label=r'$|Y|^2$')
        ax2 = fig.add_subplot(212)
        r_vals = np.linspace(0, self.scale, 2000)
        R_vals = self.calc.R_nl(r_vals, self.n, self.l)
        radial_density = R_vals**2 * r_vals**2
        ax2.plot(r_vals/self.a0, radial_density, 'b-', lw=2)
        ax2.fill_between(r_vals/self.a0, 0, radial_density, alpha=0.3, color='blue')
        if opt.get('show_labels', True):
            ax2.set_xlabel(r'$r / a_0$', fontsize=12)
            ax2.set_ylabel(r'$r^2 R^2$', fontsize=12)
            ax2.set_title(f'Radial Part: $r^2 R_{{{self.n}{self.l}}}^2$', fontsize=14)
            ax2.grid(True, alpha=0.3)
        if not opt.get('show_box', True):
            ax2.set_axis_off()
        fig.tight_layout()

    def plot_shell_3d_half(self, fig, opt):
        ax = fig.add_subplot(111, projection='3d')
        quality = opt.get('quality', 'ultra')
        if quality == 'default':
            stride = max(1, self.X.shape[0] // 80)
            X_data = self.X[::stride, ::stride]
            Z_data = self.Z[::stride, ::stride]
            shell_data = self.shell_density_norm[::stride, ::stride]
        elif quality == 'high':
            stride = max(1, self.X.shape[0] // 200)
            X_data = self.X[::stride, ::stride]
            Z_data = self.Z[::stride, ::stride]
            shell_data = self.shell_density_norm[::stride, ::stride]
        else:
            if not INTERP_AVAILABLE:
                stride = max(1, self.X.shape[0] // 200)
                X_data = self.X[::stride, ::stride]
                Z_data = self.Z[::stride, ::stride]
                shell_data = self.shell_density_norm[::stride, ::stride]
            else:
                x = self.X[0,:]; z = self.Z[:,0]
                interp = RectBivariateSpline(z, x, self.shell_density_norm)
                n_new = 2000
                x_new = np.linspace(x.min(), x.max(), n_new)
                z_new = np.linspace(z.min(), z.max(), n_new)
                X_new, Z_new = np.meshgrid(x_new, z_new)
                shell_new = interp(z_new, x_new)
                X_data, Z_data, shell_data = X_new, Z_new, shell_new
        mask = X_data >= 0
        X_plot = np.where(mask, X_data, np.nan)
        Z_plot = Z_data
        shell_plot = np.where(mask, shell_data, np.nan)
        surf = ax.plot_surface(X_plot, Z_plot, shell_plot, cmap='inferno', alpha=0.8, linewidth=0, antialiased=True)
        if opt.get('show_labels', True):
            ax.set_title(r'Half-Cut 3D Shell Density ($4\pi r^2 |\psi|^2$)')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Shell Density')
        if not opt.get('show_box', True):
            ax.set_axis_off()
        if opt.get('show_labels', True):
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='Shell Density (normalized)')

    def plot_density_3d_half(self, fig, opt):
        ax = fig.add_subplot(111, projection='3d')
        quality = opt.get('quality', 'ultra')
        if quality == 'default':
            stride = max(1, self.X.shape[0] // 80)
            X_data = self.X[::stride, ::stride]
            Z_data = self.Z[::stride, ::stride]
            dens_data = self.density[::stride, ::stride]
        elif quality == 'high':
            stride = max(1, self.X.shape[0] // 200)
            X_data = self.X[::stride, ::stride]
            Z_data = self.Z[::stride, ::stride]
            dens_data = self.density[::stride, ::stride]
        else:
            if not INTERP_AVAILABLE:
                stride = max(1, self.X.shape[0] // 200)
                X_data = self.X[::stride, ::stride]
                Z_data = self.Z[::stride, ::stride]
                dens_data = self.density[::stride, ::stride]
            else:
                x = self.X[0,:]; z = self.Z[:,0]
                interp = RectBivariateSpline(z, x, self.density)
                n_new = 2000
                x_new = np.linspace(x.min(), x.max(), n_new)
                z_new = np.linspace(z.min(), z.max(), n_new)
                X_new, Z_new = np.meshgrid(x_new, z_new)
                dens_new = interp(z_new, x_new)
                X_data, Z_data, dens_data = X_new, Z_new, dens_new
        mask = X_data >= 0
        X_plot = np.where(mask, X_data, np.nan)
        Z_plot = Z_data
        dens_plot = np.where(mask, dens_data, np.nan)
        valid = dens_plot[~np.isnan(dens_plot)]
        if len(valid) == 0:
            ax.text(0.5, 0.5, 0.5, 'No valid data', ha='center')
            return
        vmin = valid.min()
        vmax = valid.max()
        if vmin == vmax:
            if vmin == 0:
                vmin, vmax = 0, 1
            else:
                vmin = vmin * 0.9
                vmax = vmax * 1.1
        if self.n >= 20:
            if vmin <= 0:
                vmin = max(vmax * 1e-12, 1e-12)
            if vmax <= 0:
                norm = PowerNorm(gamma=0.3, vmin=0, vmax=1)
            else:
                norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)
        surf = ax.plot_surface(X_plot, Z_plot, dens_plot, cmap='nipy_spectral', alpha=0.8, linewidth=0, antialiased=True, norm=norm)
        if opt.get('show_labels', True):
            ax.set_title(r'Half-Cut 3D $|\psi|^2$ (adaptive color)')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Probability Density')
        if not opt.get('show_box', True):
            ax.set_axis_off()
        if opt.get('show_labels', True):
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=r'$|\psi|^2$ (normalized)')

    def plot_3d_realistic(self, fig, opt):
        resolution = opt.get('resolution', 80)
        cut_mode = opt.get('cut_mode', 'none')
        which_parts = opt.get('which_parts', None)
        num_cores = opt.get('num_cores', 4)
        show_box = opt.get('show_box', True)
        show_labels = opt.get('show_labels', True)
        background = opt.get('background', 'black')
        zoom_factor = opt.get('zoom_factor', 1.0)
        x = np.linspace(-self.scale, self.scale, resolution)
        y = np.linspace(-self.scale, self.scale, resolution)
        z = np.linspace(-self.scale, self.scale, resolution)
        X3, Y3, Z3 = np.meshgrid(x, y, z, indexing='ij')
        points = np.vstack([X3.ravel(), Y3.ravel(), Z3.ravel()]).T
        r_points = np.linalg.norm(points, axis=1)
        sphere_mask = r_points <= self.scale
        points = points[sphere_mask]
        n_points = len(points)
        if n_points == 0:
            return
        def compute_batch(points_batch):
            r = np.sqrt(points_batch[:,0]**2 + points_batch[:,1]**2 + points_batch[:,2]**2)
            theta = np.arccos(np.clip(points_batch[:,2] / (r + 1e-100), -1, 1))
            phi = np.arctan2(points_batch[:,1], points_batch[:,0])
            radial = self.calc.R_nl(r, self.n, self.l)
            angular = self.calc.sph_harm_compat(self.m, self.l, phi, theta)
            psi = radial * angular
            dens = np.abs(psi)**2
            return dens
        chunk_size = n_points // num_cores + 1
        chunks = [points[i:i+chunk_size] for i in range(0, n_points, chunk_size)]
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            dens_chunks = list(executor.map(compute_batch, chunks))
        dens3 = np.concatenate(dens_chunks)
        max_dens = np.max(dens3)
        if max_dens > 0:
            dens_norm = dens3 / max_dens
        else:
            dens_norm = dens3
        r3 = np.sqrt(points[:,0]**2 + points[:,1]**2 + points[:,2]**2)
        r_safe = np.where(r3 < 1e-100*self.a0, 1e-100*self.a0, r3)
        shell = 4 * np.pi * r_safe**2 * dens_norm
        max_shell = np.max(shell)
        if max_shell > 0:
            shell_norm = shell / max_shell
        else:
            shell_norm = shell
        r_min, r_max = self.calc.classical_turning_points(self.n, self.l)
        alpha_factor = np.clip(1.0 - (r3 - r_min) / (r_max - r_min), 0.3, 0.9)
        alpha = alpha_factor * np.clip(2.0 * dens_norm, 0.2, 1.0)
        alpha = np.clip(alpha, 0.3, 1.0)
        mask = get_cut_mask(points, cut_mode, which_parts)
        points_masked = points[mask]
        shell_masked = shell_norm[mask]
        alpha_masked = alpha[mask]
        if len(points_masked) == 0:
            return
        if self.n >= 20:
            norm = LogNorm(vmin=np.min(shell_masked), vmax=np.max(shell_masked))
        else:
            norm = PowerNorm(gamma=0.3, vmin=0, vmax=1)
        if background == 'black':
            fig.patch.set_facecolor('black')
            ax = fig.add_subplot(111, projection='3d', facecolor='black')
        else:
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111, projection='3d', facecolor='white')
        scatter = ax.scatter(points_masked[:,0], points_masked[:,1], points_masked[:,2],
                             c=shell_masked, cmap='nipy_spectral', norm=norm,
                             alpha=alpha_masked, s=1, linewidth=0, rasterized=True)
        lim = self.scale * zoom_factor
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        if show_labels:
            ax.set_title(f'3D Realistic Wavefunction (scatter, ${resolution}^3$, cut={cut_mode}, cores={num_cores})')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        if not show_box:
            ax.set_axis_off()
        if show_labels:
            fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10, label=r'$4\pi r^2 |\psi|^2$ (norm)')

if __name__ == "__main__":
    root = tk.Tk()
    app = HydrogenGUI(root)
    root.mainloop()
