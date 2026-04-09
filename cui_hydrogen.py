import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spe
import scipy.constants as const
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, PowerNorm, Normalize
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

try:
    from scipy.interpolate import RectBivariateSpline
    INTERP_AVAILABLE = True
except ImportError:
    INTERP_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = ["Times New Roman", "Arial"]

class HydrogenWavefunction:
    """Hydrogen atom wavefunction calculator."""
    
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
            result[mask_positive] = (np.exp(-2/3 * zeta_pos**1.5) / 
                                   (2 * np.sqrt(np.pi) * zeta_pos**0.25))
        if np.any(mask_negative):
            zeta_neg = zeta[mask_negative]
            result[mask_negative] = (np.sin(2/3 * (-zeta_neg)**1.5 - np.pi/4) / 
                                   (np.sqrt(np.pi) * (-zeta_neg)**0.25))
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
            log_coeff = 0.5 * (3 * np.log(2.0/(self.a0*n)) + log_fact1 - 
                              np.log(2.0*n) - log_fact2)
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

def run_cli():
    print("=== Hydrogen Wavefunction Visualization (CLI) ===")
    
    while True:
        print("\n" + "="*50)
        print("Enter quantum numbers (or type 'exit' to quit)")
        
        while True:
            try:
                n_input = input("Principal Quantum Number n (Positive Integer): ")
                if n_input.lower() == 'exit':
                    return
                n = int(n_input)
                if n > 0:
                    break
                else:
                    print("n must be a positive integer. Please try again.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        
        while True:
            try:
                l_input = input(f"Azimuthal Quantum Number l (0 ≤ l < {n}): ")
                if l_input.lower() == 'exit':
                    return
                l = int(l_input)
                if 0 <= l < n:
                    break
                else:
                    print(f"l must satisfy 0 ≤ l < {n}. Please try again.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        
        while True:
            try:
                m_input = input(f"Magnetic Quantum Number m (-{l} ≤ m ≤ {l}): ")
                if m_input.lower() == 'exit':
                    return
                m = int(m_input)
                if -l <= m <= l:
                    break
                else:
                    print(f"m must satisfy -{l} ≤ m ≤ {l}. Please try again.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        
        while True:
            try:
                g_input = input("Grid Size for 2D slices (larger = more detail but slower, e.g., 400): ")
                if g_input.lower() == 'exit':
                    return
                grid_size = int(g_input)
                if grid_size > 0:
                    break
                else:
                    print("Grid size must be a positive integer.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        
        print("Calculating 2D slice... This may take a few minutes.")
        
        calc = HydrogenWavefunction()
        a0 = calc.a0
        r_min, r_max = calc.classical_turning_points(n, l)
        
        X, Y, Z, scale = calc.create_extended_grid(n, l, grid_size=grid_size)
        r, theta, phi = cart2sph(X, Y, Z)
        
        start_time = time.time()
        radial = calc.R_nl(r, n, l)
        angular = calc.sph_harm_compat(m, l, phi, theta)
        psi = radial * angular
        density = np.abs(psi)**2
        
        max_density = np.max(density)
        if max_density > 0:
            density = density / max_density
        
        r_safe = np.where(r < 1e-100 * a0, 1e-100 * a0, r)
        shell_density = 4 * np.pi * r_safe**2 * density
        max_shell = np.max(shell_density)
        if max_shell > 0:
            shell_density_norm = shell_density / max_shell
        else:
            shell_density_norm = shell_density
        
        log_density = np.log10(density + 1e-100)
        
        elapsed = time.time() - start_time
        print(f"2D slice calculation completed in {elapsed:.2f} seconds.")
        
        theta_circle = np.linspace(0, 2*np.pi, 200)
        x_classical = r_max * np.cos(theta_circle)
        z_classical = r_max * np.sin(theta_circle)
        
        def compute_cross_section(plane, grid_size, scale, calc, n, l, m, a0):
            if plane == 'x':
                y = np.linspace(-scale, scale, grid_size)
                z = np.linspace(-scale, scale, grid_size)
                Y, Z = np.meshgrid(y, z)
                X = np.zeros_like(Y)
                points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                extent = [-scale, scale, -scale, scale]
                xlabel, ylabel = r'Y (m)', r'Z (m)'
                grid1, grid2 = Y, Z
            elif plane == 'y':
                x = np.linspace(-scale, scale, grid_size)
                z = np.linspace(-scale, scale, grid_size)
                X, Z = np.meshgrid(x, z)
                Y = np.zeros_like(X)
                points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                extent = [-scale, scale, -scale, scale]
                xlabel, ylabel = r'X (m)', r'Z (m)'
                grid1, grid2 = X, Z
            elif plane == 'z':
                x = np.linspace(-scale, scale, grid_size)
                y = np.linspace(-scale, scale, grid_size)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                extent = [-scale, scale, -scale, scale]
                xlabel, ylabel = r'X (m)', r'Y (m)'
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
                xlabel, ylabel = r'$u$ (m)', r'$v$ (m)'
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

        # 以下为所有绘图函数，保留原样（略作缩进调整）
        def plot_linear_scale(show_classical_boundary=True, show_box=True, show_labels=True, fig_num=1):
            fig, ax = plt.subplots(figsize=(8,6))
            im = ax.imshow(density, extent=[-scale,scale,-scale,scale],
                           origin='lower', cmap='viridis', aspect='equal')
            if show_labels:
                ax.set_title(r'Linear Scale')
                ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
            if show_classical_boundary:
                ax.plot(x_classical, z_classical, 'r--', lw=2, alpha=0.8, label='Classical Boundary')
                if show_labels:
                    ax.legend()
            if not show_box:
                ax.set_axis_off()
            if show_labels:
                plt.colorbar(im, ax=ax, label=r'Probability Density (normalized)')
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot1_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot1_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_log_scale(show_box=True, show_labels=True, fig_num=2):
            fig, ax = plt.subplots(figsize=(8,6))
            im = ax.imshow(log_density, extent=[-scale,scale,-scale,scale],
                           origin='lower', cmap='plasma', aspect='equal')
            if show_labels:
                ax.set_title(r'Logarithmic Scale')
                ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
            if not show_box:
                ax.set_axis_off()
            if show_labels:
                plt.colorbar(im, ax=ax, label=r'$\log_{10}|\psi|^2$')
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot2_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot2_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_contour(show_labels=True, show_box=True, show_axis_labels=True, fig_num=3):
            fig, ax = plt.subplots(figsize=(8,6))
            max_dens = np.max(density)
            if max_dens > 0:
                num_levels = 30 if n >= 5 else 15
                levels = np.logspace(np.log10(max_dens*1e-5), np.log10(max_dens), num_levels)
                try:
                    contour = ax.contour(X, Z, density, levels=levels, cmap='hot', linewidths=1.5)
                    if show_labels:
                        ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1e')
                except:
                    contour = ax.contour(X, Z, density, levels=num_levels, cmap='hot', linewidths=1.5)
                    if show_labels:
                        ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1e')
            if show_axis_labels:
                ax.set_title(r'Contour Plot')
                ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
            ax.set_xlim(-scale,scale); ax.set_ylim(-scale,scale)
            if show_axis_labels:
                ax.grid(True, alpha=0.3)
            if not show_box:
                ax.set_axis_off()
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot3_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot3_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_3d_surface(quality='ultra', show_box=True, show_labels=True, fig_num=4):
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')
            if quality == 'default':
                stride = max(1, X.shape[0] // 80)
                X_plot, Z_plot, dens_plot = X[::stride,::stride], Z[::stride,::stride], density[::stride,::stride]
            elif quality == 'high':
                stride = max(1, X.shape[0] // 200)
                X_plot, Z_plot, dens_plot = X[::stride,::stride], Z[::stride,::stride], density[::stride,::stride]
            else:
                if not INTERP_AVAILABLE:
                    stride = max(1, X.shape[0] // 200)
                    X_plot, Z_plot, dens_plot = X[::stride,::stride], Z[::stride,::stride], density[::stride,::stride]
                else:
                    x = X[0,:]; z = Z[:,0]
                    interp = RectBivariateSpline(z, x, density)
                    n_new = 2000
                    x_new = np.linspace(x.min(), x.max(), n_new)
                    z_new = np.linspace(z.min(), z.max(), n_new)
                    X_new, Z_new = np.meshgrid(x_new, z_new)
                    dens_new = interp(z_new, x_new)
                    X_plot, Z_plot, dens_plot = X_new, Z_new, dens_new
            surf = ax.plot_surface(X_plot, Z_plot, dens_plot, cmap='viridis',
                                   alpha=0.8, linewidth=0, antialiased=True)
            if show_labels:
                ax.set_title(r'3D Surface ($|\psi|^2$)')
                ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Probability Density')
            if not show_box:
                ax.set_axis_off()
            if show_labels:
                fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=r'$|\psi|^2$ (normalized)')
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot4_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot4_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_radial_density(show_classical=True, show_box=True, show_labels=True, fig_num=5):
            r_vals = np.linspace(0, scale, 2000)
            R_vals = calc.R_nl(r_vals, n, l)
            quantum = R_vals**2 * r_vals**2

            fig, ax = plt.subplots(figsize=(8,6))
            ax.semilogy(r_vals/a0, quantum + 1e-20, 'b-', lw=2, label='Quantum')

            if show_classical:
                r_min, r_max = calc.classical_turning_points(n, l)
                if l == n - 1:
                    r0 = (r_min + r_max) / 2
                    ax.axvline(r0/a0, color='red', ls='-', lw=2, alpha=0.7, label='Classical (δ)')
                else:
                    eps = 1e-9 * (r_max - r_min)
                    r_classical = np.linspace(r_min + eps, r_max - eps, 2000)
                    classical = 1.0 / (np.pi * np.sqrt((r_max - r_classical) * (r_classical - r_min)))
                    ax.semilogy(r_classical/a0, classical + 1e-20, 'r-', lw=2, alpha=0.7, label='Classical')
                ax.axvline(r_min/a0, color='red', ls='--', alpha=0.5)
                ax.axvline(r_max/a0, color='red', ls='--', alpha=0.5)

            if show_labels:
                ax.set_xlabel(r'$r / a_0$')
                ax.set_ylabel('Radial Probability Density')
                ax.set_title(f'Radial Density for $n={n}$, $l={l}$')
                ax.legend()
                ax.grid(True, alpha=0.3)
            if not show_box:
                ax.set_axis_off()
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot5_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot5_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_angular_distribution(show_box=True, show_labels=True, fig_num=6):
            theta_vals = np.linspace(0, 2*np.pi, 500)
            phi_vals = np.linspace(0, np.pi, 500)
            Theta, Phi = np.meshgrid(theta_vals, phi_vals)
            Y_lm = np.abs(calc.sph_harm_compat(m, l, Theta, Phi))**2
            fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(8,6))
            contour = ax.contourf(Theta, Phi, Y_lm, 50, cmap='cool')
            if show_labels:
                ax.set_title(f'Angular Distribution: $l={l}$, $m={m}$')
            if not show_box:
                ax.set_axis_off()
            if show_labels:
                plt.colorbar(contour, ax=ax, label=r'$|Y|^2$')
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot6_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot6_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_enhanced_2d(show_white_contour=True, show_classical_boundary=True, show_info_box=True, show_box=True, show_labels=True, fig_num=7):
            fig, ax = plt.subplots(figsize=(12,8))
            if n >= 20:
                norm = LogNorm(vmin=np.max(density)*1e-6, vmax=np.max(density))
            else:
                norm = PowerNorm(gamma=0.3, vmin=0, vmax=1)
            im = ax.imshow(density, extent=[-scale,scale,-scale,scale],
                           origin='lower', cmap='nipy_spectral', norm=norm, aspect='equal')
            if show_white_contour:
                levels = np.logspace(np.log10(np.max(density)*1e-6), np.log10(np.max(density)), 20)
                CS = ax.contour(X, Z, density, levels=levels, colors='white', alpha=0.7, linewidths=0.8)
                if show_labels:
                    ax.clabel(CS, inline=True, fontsize=8, fmt='%1.0e')
            if show_classical_boundary:
                ax.plot(x_classical, z_classical, 'w--', lw=2.5, alpha=0.9, label='Classical Boundary')
                if show_labels:
                    ax.legend()
            if show_labels:
                ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
                ax.set_title(f'Hydrogen Atom State ($n={n}$, $l={l}$, $m={m}$)', fontsize=16, fontweight='bold')
            if show_info_box and show_labels:
                info = f'$n={n}$, $l={l}$, $m={m}$\nClassical radius: {r_max/a0:.0f} $a_0$\nBohr radius: {a0:.2e} m\nCapture range: {scale/a0:.0f} $a_0$'
                ax.text(0.02, 0.98, info, transform=ax.transAxes, va='top', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            if not show_box:
                ax.set_axis_off()
            if show_labels:
                plt.colorbar(im, ax=ax, label=r'Probability Density (normalized)')
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot7_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot7_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_shell_2d(show_classical_boundary=True, show_box=True, show_labels=True, fig_num=8):
            while True:
                color_choice = input("Color style? (1=enhanced style (like plot7), 2=original inferno, default 1): ").strip()
                if color_choice == '2':
                    cmap = 'inferno'
                    use_enhanced_norm = False
                else:
                    cmap = 'nipy_spectral'
                    use_enhanced_norm = True
                
                print("Select cross-section:")
                print("  X - x=0 plane (default)")
                print("  Y - y=0 plane")
                print("  Z - z=0 plane")
                print("  XYZ - x+y+z=0 plane (diagonal cut)")
                section = input("Choice (X/Y/Z/XYZ, default X): ").strip().upper()
                if section not in ['X', 'Y', 'Z', 'XYZ']:
                    section = 'X'
                
                plane_map = {'X': 'x', 'Y': 'y', 'Z': 'z', 'XYZ': 'xyz'}
                plane = plane_map[section]
                shell_slice, extent, xlabel, ylabel, _, _ = compute_cross_section(
                    plane, grid_size, scale, calc, n, l, m, a0)
                
                if np.all(np.isnan(shell_slice)):
                    print("Warning: No valid data on this cross-section. Skipping.")
                    continue
                
                if use_enhanced_norm:
                    if n >= 20:
                        valid = shell_slice[~np.isnan(shell_slice)]
                        if len(valid) > 0 and np.any(valid > 0):
                            vmin = max(np.min(valid[valid > 0]), 1e-6)
                        else:
                            vmin = 1e-6
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
                
                fig, ax = plt.subplots(figsize=(8,6))
                im = ax.imshow(shell_slice, extent=extent, origin='lower', cmap=cmap, norm=norm, aspect='equal')
                if show_labels:
                    title = f'Shell Density on {section}=0 Plane' if section != 'XYZ' else 'Shell Density on $x+y+z=0$ Plane'
                    ax.set_title(title)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                if show_classical_boundary and section != 'XYZ':
                    if section == 'X':
                        theta = np.linspace(0, 2*np.pi, 200)
                        y_circ = r_max * np.cos(theta)
                        z_circ = r_max * np.sin(theta)
                        ax.plot(y_circ, z_circ, 'w--', lw=2, alpha=0.8, label='Classical Boundary')
                    elif section == 'Y':
                        theta = np.linspace(0, 2*np.pi, 200)
                        x_circ = r_max * np.cos(theta)
                        z_circ = r_max * np.sin(theta)
                        ax.plot(x_circ, z_circ, 'w--', lw=2, alpha=0.8, label='Classical Boundary')
                    elif section == 'Z':
                        theta = np.linspace(0, 2*np.pi, 200)
                        x_circ = r_max * np.cos(theta)
                        y_circ = r_max * np.sin(theta)
                        ax.plot(x_circ, y_circ, 'w--', lw=2, alpha=0.8, label='Classical Boundary')
                    if show_labels:
                        ax.legend()
                if not show_box:
                    ax.set_axis_off()
                if show_labels:
                    plt.colorbar(im, ax=ax, label=r'$4\pi r^2 |\psi|^2$ (normalized)')
                plt.tight_layout()
                plt.show()
                
                save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
                if save_opt == 'y':
                    default_fname = f"plot8_{section}_n{n}_l{l}_m{m}.png"
                    fname = input(f"Enter filename (default: {default_fname}): ").strip()
                    if fname == '':
                        fname = default_fname
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                    print(f"Figure saved as {fname}")
                
                cont = input("Generate another cross-section for the same quantum numbers? (y/n, default n): ").strip().lower()
                if cont != 'y':
                    break
        
        def plot_shell_3d(quality='ultra', show_box=True, show_labels=True, fig_num=9):
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')
            if quality == 'default':
                stride = max(1, X.shape[0] // 80)
                X_plot, Z_plot, shell_plot = X[::stride,::stride], Z[::stride,::stride], shell_density_norm[::stride,::stride]
            elif quality == 'high':
                stride = max(1, X.shape[0] // 200)
                X_plot, Z_plot, shell_plot = X[::stride,::stride], Z[::stride,::stride], shell_density_norm[::stride,::stride]
            else:
                if not INTERP_AVAILABLE:
                    stride = max(1, X.shape[0] // 200)
                    X_plot, Z_plot, shell_plot = X[::stride,::stride], Z[::stride,::stride], shell_density_norm[::stride,::stride]
                else:
                    x = X[0,:]; z = Z[:,0]
                    interp = RectBivariateSpline(z, x, shell_density_norm)
                    n_new = 2000
                    x_new = np.linspace(x.min(), x.max(), n_new)
                    z_new = np.linspace(z.min(), z.max(), n_new)
                    X_new, Z_new = np.meshgrid(x_new, z_new)
                    shell_new = interp(z_new, x_new)
                    X_plot, Z_plot, shell_plot = X_new, Z_new, shell_new
            surf = ax.plot_surface(X_plot, Z_plot, shell_plot, cmap='inferno',
                                   alpha=0.8, linewidth=0, antialiased=True)
            if show_labels:
                ax.set_title(r'3D Surface with Shell Factor ($4\pi r^2 |\psi|^2$)')
                ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Shell Density')
            if not show_box:
                ax.set_axis_off()
            if show_labels:
                fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=r'Shell Density (normalized)')
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot9_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot9_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_angular_radial(show_box=True, show_labels=True, fig_num=10):
            fig = plt.figure(figsize=(8,10))
            ax1 = plt.subplot(211, projection='polar')
            theta_vals = np.linspace(0, 2*np.pi, 500)
            phi_vals = np.linspace(0, np.pi, 500)
            Theta, Phi = np.meshgrid(theta_vals, phi_vals)
            Y_lm = np.abs(calc.sph_harm_compat(m, l, Theta, Phi))**2
            contour = ax1.contourf(Theta, Phi, Y_lm, 50, cmap='cool')
            if show_labels:
                ax1.set_title(fr'Angular Part: $|Y_{{{l}}}^{{{m}}}|^2$')
            if not show_box:
                ax1.set_axis_off()
            if show_labels:
                plt.colorbar(contour, ax=ax1, label=r'$|Y|^2$')
            
            ax2 = plt.subplot(212)
            r_vals = np.linspace(0, scale, 2000)
            R_vals = calc.R_nl(r_vals, n, l)
            radial_density = R_vals**2 * r_vals**2
            ax2.plot(r_vals/a0, radial_density, 'b-', lw=2)
            ax2.fill_between(r_vals/a0, 0, radial_density, alpha=0.3, color='blue')
            if show_labels:
                ax2.set_xlabel(r'$r / a_0$')
                ax2.set_ylabel(r'$r^2 R^2$')
                ax2.set_title(fr'Radial Part: $r^2 R_{{{n}{l}}}^2$')
                ax2.grid(True, alpha=0.3)
            if not show_box:
                ax2.set_axis_off()
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot10_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot10_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_shell_3d_half(quality='ultra', show_box=True, show_labels=True, fig_num=11):
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')
            if quality == 'default':
                stride = max(1, X.shape[0] // 80)
                X_data = X[::stride, ::stride]
                Z_data = Z[::stride, ::stride]
                shell_data = shell_density_norm[::stride, ::stride]
            elif quality == 'high':
                stride = max(1, X.shape[0] // 200)
                X_data = X[::stride, ::stride]
                Z_data = Z[::stride, ::stride]
                shell_data = shell_density_norm[::stride, ::stride]
            else:
                if not INTERP_AVAILABLE:
                    stride = max(1, X.shape[0] // 200)
                    X_data = X[::stride, ::stride]
                    Z_data = Z[::stride, ::stride]
                    shell_data = shell_density_norm[::stride, ::stride]
                else:
                    x = X[0,:]; z = Z[:,0]
                    interp = RectBivariateSpline(z, x, shell_density_norm)
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
            surf = ax.plot_surface(X_plot, Z_plot, shell_plot, cmap='inferno',
                                   alpha=0.8, linewidth=0, antialiased=True)
            if show_labels:
                ax.set_title(r'Half-Cut 3D Shell Density ($4\pi r^2 |\psi|^2$)')
                ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Shell Density')
            if not show_box:
                ax.set_axis_off()
            if show_labels:
                fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=r'Shell Density (normalized)')
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot11_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot11_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_density_3d_half(quality='ultra', show_box=True, show_labels=True, fig_num=12):
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111, projection='3d')
            if quality == 'default':
                stride = max(1, X.shape[0] // 80)
                X_data = X[::stride, ::stride]
                Z_data = Z[::stride, ::stride]
                dens_data = density[::stride, ::stride]
            elif quality == 'high':
                stride = max(1, X.shape[0] // 200)
                X_data = X[::stride, ::stride]
                Z_data = Z[::stride, ::stride]
                dens_data = density[::stride, ::stride]
            else:
                if not INTERP_AVAILABLE:
                    stride = max(1, X.shape[0] // 200)
                    X_data = X[::stride, ::stride]
                    Z_data = Z[::stride, ::stride]
                    dens_data = density[::stride, ::stride]
                else:
                    x = X[0,:]; z = Z[:,0]
                    interp = RectBivariateSpline(z, x, density)
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
                print("Warning: No valid data in half-cut region. Skipping plot.")
                plt.close(fig)
                return
            vmin = valid.min()
            vmax = valid.max()
            if vmin == vmax:
                if vmin == 0:
                    vmin, vmax = 0, 1
                else:
                    vmin = vmin * 0.9
                    vmax = vmax * 1.1
            if n >= 20:
                if vmin <= 0:
                    vmin = max(vmax * 1e-12, 1e-12)
                if vmax <= 0:
                    norm = PowerNorm(gamma=0.3, vmin=0, vmax=1)
                else:
                    norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)
            surf = ax.plot_surface(X_plot, Z_plot, dens_plot, cmap='nipy_spectral',
                                   alpha=0.8, linewidth=0, antialiased=True, norm=norm)
            if show_labels:
                ax.set_title(r'Half-Cut 3D $|\psi|^2$ (adaptive color)')
                ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Probability Density')
            if not show_box:
                ax.set_axis_off()
            if show_labels:
                fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=r'$|\psi|^2$ (normalized)')
            plt.tight_layout()
            plt.show()
            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot12_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot12_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")
        
        def plot_3d_realistic(resolution=80, cut_mode='none', which_parts=None, num_cores=4,
                              show_box=True, show_labels=True, background='black', zoom_factor=1.0, fig_num=13):
            print(f"Generating 3D scatter plot with {resolution}^3 = {resolution**3} points. Using {num_cores} core(s)...")
            start_3d = time.time()

            x = np.linspace(-scale, scale, resolution)
            y = np.linspace(-scale, scale, resolution)
            z = np.linspace(-scale, scale, resolution)
            X3, Y3, Z3 = np.meshgrid(x, y, z, indexing='ij')
            points = np.vstack([X3.ravel(), Y3.ravel(), Z3.ravel()]).T
            r_points = np.linalg.norm(points, axis=1)
            sphere_mask = r_points <= scale
            points = points[sphere_mask]
            n_points = len(points)
            print(f"Sphere contains {n_points} points ({(n_points/(resolution**3)*100):.1f}% of cube).")

            if n_points == 0:
                print("No points inside sphere. Aborting.")
                return

            def compute_batch(points_batch):
                r = np.sqrt(points_batch[:,0]**2 + points_batch[:,1]**2 + points_batch[:,2]**2)
                theta = np.arccos(np.clip(points_batch[:,2] / (r + 1e-100), -1, 1))
                phi = np.arctan2(points_batch[:,1], points_batch[:,0])
                radial = calc.R_nl(r, n, l)
                angular = calc.sph_harm_compat(m, l, phi, theta)
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
            r_safe = np.where(r3 < 1e-100*a0, 1e-100*a0, r3)
            shell = 4 * np.pi * r_safe**2 * dens_norm
            max_shell = np.max(shell)
            if max_shell > 0:
                shell_norm = shell / max_shell
            else:
                shell_norm = shell

            r_min, r_max = calc.classical_turning_points(n, l)
            alpha_factor = np.clip(1.0 - (r3 - r_min) / (r_max - r_min), 0.3, 0.9)
            alpha = alpha_factor * np.clip(2.0 * dens_norm, 0.2, 1.0)
            alpha = np.clip(alpha, 0.3, 1.0)

            mask = get_cut_mask(points, cut_mode, which_parts)
            points_masked = points[mask]
            shell_masked = shell_norm[mask]
            alpha_masked = alpha[mask]

            if len(points_masked) == 0:
                print("No points remain after cutting. Aborting.")
                return

            if n >= 20:
                norm = LogNorm(vmin=np.min(shell_masked), vmax=np.max(shell_masked))
            else:
                norm = PowerNorm(gamma=0.3, vmin=0, vmax=1)

            fig = plt.figure(figsize=(12, 10))
            if background == 'black':
                fig.patch.set_facecolor('black')
                ax = fig.add_subplot(111, projection='3d', facecolor='black')
            else:
                fig.patch.set_facecolor('white')
                ax = fig.add_subplot(111, projection='3d', facecolor='white')

            scatter = ax.scatter(points_masked[:,0], points_masked[:,1], points_masked[:,2],
                                 c=shell_masked, cmap='nipy_spectral', norm=norm,
                                 alpha=alpha_masked, s=1, linewidth=0, rasterized=True)

            lim = scale * zoom_factor
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

            elapsed = time.time() - start_3d
            print(f"Scatter plot generated in {elapsed:.2f} seconds.")
            plt.tight_layout()
            plt.show()

            save_opt = input("Save this figure? (y/n, default n): ").strip().lower()
            if save_opt == 'y':
                fname = input("Enter filename (default: plot13_n{}_l{}_m{}.png): ".format(n,l,m)).strip()
                if fname == '':
                    fname = f"plot13_n{n}_l{l}_m{m}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {fname}")

            save_cross_opt = input("Save three orthogonal cross-section images (x=0, y=0, z=0) with shell density? (y/n, default n): ").strip().lower()
            if save_cross_opt == 'y':
                planes = ['x', 'y', 'z']
                plane_names = ['X=0', 'Y=0', 'Z=0']
                for plane, pname in zip(planes, plane_names):
                    print(f"Computing cross-section for {pname}...")
                    shell_slice, extent, xlabel, ylabel, _, _ = compute_cross_section(
                        plane, grid_size, scale, calc, n, l, m, a0)
                    if np.all(np.isnan(shell_slice)):
                        print(f"Warning: No valid data on {pname} plane. Skipping.")
                        continue
                    if n >= 20:
                        valid = shell_slice[~np.isnan(shell_slice)]
                        vmin = max(np.min(valid[valid > 0]), 1e-6) if np.any(valid > 0) else 1e-6
                        norm_slice = LogNorm(vmin=vmin, vmax=1)
                    else:
                        valid = shell_slice[~np.isnan(shell_slice)]
                        vmin = valid.min()
                        vmax = valid.max()
                        if vmin == vmax:
                            if vmin == 0:
                                vmin, vmax = 0, 1
                            else:
                                vmin = vmin * 0.9
                                vmax = vmax * 1.1
                        norm_slice = PowerNorm(gamma=0.3, vmin=vmin, vmax=vmax)
                    fig_slice, ax_slice = plt.subplots(figsize=(8,6))
                    im = ax_slice.imshow(shell_slice, extent=extent, origin='lower',
                                         cmap='nipy_spectral', norm=norm_slice, aspect='equal')
                    ax_slice.set_title(f'Shell Density on {pname} Plane ($n={n}$, $l={l}$, $m={m}$)')
                    ax_slice.set_xlabel(xlabel)
                    ax_slice.set_ylabel(ylabel)
                    plt.colorbar(im, ax=ax_slice, label=r'$4\pi r^2 |\psi|^2$ (norm)')
                    plt.tight_layout()
                    default_fname = f"cross_section_{plane}_n{n}_l{l}_m{m}.png"
                    fname_input = input(f"Enter filename for {pname} cross-section (default: {default_fname}): ").strip()
                    if fname_input == '':
                        fname_input = default_fname
                    fig_slice.savefig(fname_input, dpi=300, bbox_inches='tight')
                    print(f"Saved as {fname_input}")
                    plt.close(fig_slice)

        # ========== 定量分析函数 ==========
        def analyze_band_integral(alpha=0.05, npoints=20000, savefig=None):
            """
            环带积分（改进版：归一化径向密度）
            """
            l_vals = np.arange(n)  # l = 0,...,n-1
            integrals = []
            for l_val in l_vals:
                r_min_l, r_max_l = calc.classical_turning_points(n, l_val)
                delta = alpha * r_max_l
                a, b = max(0, r_max_l - delta), r_max_l + delta
                if a >= b:
                    integrals.append(0.0)
                    continue

                # 计算总概率（归一化因子）
                r_full = np.linspace(0, 3 * r_max_l, npoints)  # 足够覆盖整个径向范围
                dr_full = r_full[1] - r_full[0]
                P_full = r_full**2 * calc.R_nl(r_full, n, l_val)**2
                total = np.trapz(P_full, dx=dr_full)

                # 环带积分
                r_band = np.linspace(a, b, npoints)
                dr_band = r_band[1] - r_band[0]
                P_band = r_band**2 * calc.R_nl(r_band, n, l_val)**2
                integral = np.trapz(P_band, dx=dr_band) / total   # 归一化
                integrals.append(integral)

            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(l_vals, integrals, 'bo-', markersize=4, linewidth=1.5, label='Normalized band probability')
            ax.set_xlabel('Angular quantum number $l$')
            ax.set_ylabel('Probability in band')
            ax.set_title(rf'$n={n}$, band half-width = {alpha*100:.1f}% of $r_{{\max}}$')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)  # 保证不超过1
            ax.legend()
            plt.tight_layout()
            plt.show()
            if savefig:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {savefig}")
            return fig

        # ========== 修正的模块15：圆轨道峰位偏差分析（专用公式）==========
        def analyze_peak_deviation_circular(n_list, npoints=200000, savefig=None):
            """
            圆轨道峰位偏差分析（改进版：高密度网格 + 局部精细扫描）
            绘制：
              - 原始数据折线图
              - 理论曲线 1/n
              - 上包络线（局部极大值连线）
              - 下包络线（局部极小值连线）
            """
            a0 = calc.a0
            n_vals = np.array(sorted(set(n_list)))
            deviations = []
            valid_n = []

            for n_val in n_vals:
                l_val = n_val - 1
                r_max_class = n_val**2 * a0 * (1 + np.sqrt(1 - (l_val**2)/n_val**2))
                # 全局粗网格
                r_coarse = np.linspace(0.5 * r_max_class, 1.5 * r_max_class, npoints)
                log_coeff = 1.5 * np.log(2/(n_val*a0)) - 0.5 * (np.log(2*n_val) + spe.loggamma(2*n_val))
                log_r_part = (n_val-1) * np.log(2*r_coarse/(n_val*a0))
                log_exp = -r_coarse/(n_val*a0)
                log_R = log_coeff + log_r_part + log_exp
                log_R = np.clip(log_R, -500, 500)
                R = np.exp(log_R)
                P_coarse = r_coarse**2 * R**2

                if P_coarse.max() < 1e-20:
                    print(f"Warning: n={n_val} yields negligible radial density, skipping.")
                    continue

                # 粗略峰值位置
                i_peak_coarse = np.argmax(P_coarse)
                r_peak_coarse = r_coarse[i_peak_coarse]

                # 局部精细扫描（以粗略峰位为中心，取 ±0.5% 区间，用 10000 点）
                delta = 0.005 * r_peak_coarse
                left = max(r_coarse[0], r_peak_coarse - delta)
                right = min(r_coarse[-1], r_peak_coarse + delta)
                r_fine = np.linspace(left, right, 10000)
                log_r_part_fine = (n_val-1) * np.log(2*r_fine/(n_val*a0))
                log_exp_fine = -r_fine/(n_val*a0)
                log_R_fine = log_coeff + log_r_part_fine + log_exp_fine
                log_R_fine = np.clip(log_R_fine, -500, 500)
                R_fine = np.exp(log_R_fine)
                P_fine = r_fine**2 * R_fine**2

                i_peak_fine = np.argmax(P_fine)
                r_peak = r_fine[i_peak_fine]

                r_classical = n_val**2 * a0
                deviation = abs(r_peak - r_classical) / r_classical
                deviations.append(deviation)
                valid_n.append(n_val)
                print(f"n={n_val}: r_peak={r_peak:.3e}, r_classical={r_classical:.3e}, deviation={deviation:.3e}")

            if len(deviations) == 0:
                print("No valid data points.")
                return

            deviations = np.array(deviations)
            n_vals = np.array(valid_n)

            # 提取局部极大值和极小值（用于包络线）
            def local_extrema(x, y):
                # 返回极大值点和极小值点的索引
                maxima_idx = []
                minima_idx = []
                for i in range(1, len(y)-1):
                    if y[i] > y[i-1] and y[i] > y[i+1]:
                        maxima_idx.append(i)
                    if y[i] < y[i-1] and y[i] < y[i+1]:
                        minima_idx.append(i)
                # 包含端点？这里不包含，但可手动添加首尾
                # 将端点也作为极值点处理（可选）
                return maxima_idx, minima_idx

            max_idx, min_idx = local_extrema(n_vals, deviations)
            # 添加端点以使包络线连续（可选）
            if len(max_idx) > 0:
                if 0 not in max_idx:
                    max_idx = [0] + max_idx
                if len(n_vals)-1 not in max_idx:
                    max_idx.append(len(n_vals)-1)
            else:
                max_idx = [0, len(n_vals)-1]  # 如果无极大值，直接用端点
            if len(min_idx) > 0:
                if 0 not in min_idx:
                    min_idx = [0] + min_idx
                if len(n_vals)-1 not in min_idx:
                    min_idx.append(len(n_vals)-1)
            else:
                min_idx = [0, len(n_vals)-1]

            # 按索引排序
            max_idx = sorted(set(max_idx))
            min_idx = sorted(set(min_idx))

            # 理论曲线: 1/n
            theory = 1.0 / n_vals

            # 绘图
            fig, ax = plt.subplots(figsize=(9,6))
            # 原始数据折线（浅色细线）
            ax.loglog(n_vals, deviations, 'o-', color='gray', markersize=4, linewidth=1, alpha=0.6, label='Data points')
            # 理论曲线
            ax.loglog(n_vals, theory, 'k--', linewidth=2, label=r'Theory $1/n$')
            # 上包络线（极大值连线）
            ax.loglog(n_vals[max_idx], deviations[max_idx], 'r-', linewidth=2, marker='^', markersize=6, label='Upper envelope')
            # 下包络线（极小值连线）
            ax.loglog(n_vals[min_idx], deviations[min_idx], 'b-', linewidth=2, marker='v', markersize=6, label='Lower envelope')

            ax.set_xlabel('Principal quantum number $n$')
            ax.set_ylabel(r'Relative peak deviation $\Delta r / r_0$')
            ax.set_title('Circular orbit ($l=n-1$) peak deviation')
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')
            plt.tight_layout()
            plt.show()

            if savefig:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {savefig}")
            return fig

        def analyze_angular_width(l_max=100, m_val=0, savefig=None):
            """角向分布宽度：固定 m，计算 |Y_l^m|^2 的 FWHM 随 l 的变化"""
            l_vals = np.arange(1, l_max+1)
            widths = []
            for l_val in l_vals:
                theta = np.linspace(0, np.pi, 5000)
                Y2 = np.abs(calc.sph_harm_compat(m_val, l_val, 0.0, theta))**2
                Y2 /= np.max(Y2)
                i_max = np.argmax(Y2)
                theta_max = theta[i_max]
                half = 0.5

                left_idx = i_max
                while left_idx > 0 and Y2[left_idx] > half:
                    left_idx -= 1
                if left_idx == 0:
                    left = theta[0]
                else:
                    t1, t2 = theta[left_idx], theta[left_idx+1]
                    y1, y2 = Y2[left_idx], Y2[left_idx+1]
                    if y2 == y1:
                        left = (t1 + t2) / 2
                    else:
                        left = t1 + (half - y1) * (t2 - t1) / (y2 - y1)

                right_idx = i_max
                while right_idx < len(theta)-1 and Y2[right_idx] > half:
                    right_idx += 1
                if right_idx == len(theta)-1:
                    right = theta[-1]
                else:
                    t1, t2 = theta[right_idx-1], theta[right_idx]
                    y1, y2 = Y2[right_idx-1], Y2[right_idx]
                    if y2 == y1:
                        right = (t1 + t2) / 2
                    else:
                        right = t1 + (half - y1) * (t2 - t1) / (y2 - y1)

                width = right - left
                widths.append(width)

            if len(widths) == 0:
                print("No angular width data.")
                return

            widths = np.array(widths)

            def power_law(x, a, b):
                return a * x**b

            try:
                popt, _ = curve_fit(power_law, l_vals, widths)
                a_fit, b_fit = popt
                fit_possible = True
            except Exception as e:
                print(f"Fitting failed: {e}")
                fit_possible = False

            fig, ax = plt.subplots(figsize=(8,5))
            ax.loglog(l_vals, widths, 'bo', label='Data')
            if fit_possible:
                ax.loglog(l_vals, power_law(l_vals, *popt), 'r-', label=rf'Fit: $\propto l^{{{b_fit:.2f}}}$')
            ax.set_xlabel('Angular quantum number $l$')
            ax.set_ylabel('FWHM (radians)')
            ax.set_title(rf'Angular width of $|Y_l^{{{m_val}}}|^2$')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            if savefig:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {savefig}")
            return fig

        # ========== 新增分析函数（嵌套在 run_cli 内以捕获 calc） ==========
        def plot_classical_vs_quantum_radial(n, l, npoints=5000, savefig=None):
            """
            绘制量子径向概率密度与经典径向分布的对比
            """
            a0 = calc.a0
            r_min, r_max = calc.classical_turning_points(n, l)
            r = np.linspace(r_min, r_max, npoints)
            
            # 原子单位
            E_hartree = -1/(2*n**2)
            L2 = l*(l+1)
            U_eff = -1/r + L2/(2*r**2)
            T_r = E_hartree - U_eff
            v_r = np.sqrt(2 * np.abs(T_r))
            P_classical = 1 / v_r
            norm = np.trapz(P_classical, r)
            if norm > 0:
                P_classical /= norm
            else:
                P_classical = np.zeros_like(r)
            
            R_vals = calc.R_nl(r, n, l)
            P_quantum = r**2 * R_vals**2
            norm_q = np.trapz(P_quantum, r)
            if norm_q > 0:
                P_quantum /= norm_q
            
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(r/a0, P_quantum, 'b-', lw=1.5, label='Quantum')
            ax.plot(r/a0, P_classical, 'r--', lw=2, label='Classical')
            ax.set_xlabel(r'$r / a_0$')
            ax.set_ylabel('Radial probability density')
            ax.set_title(f'$n={n}$, $l={l}$')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            if savefig:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {savefig}")
            return fig

        def analyze_radial_moments(n, l, rmax_factor=10, npoints=100000, savefig=None):
            """
            计算并显示径向分布的均值 <r> 和方差，与理论值对比
            """
            a0 = calc.a0
            r_max = calc.classical_turning_points(n, l)[1] * rmax_factor
            r = np.linspace(0, r_max, npoints)
            dr = r[1] - r[0]
            R = calc.R_nl(r, n, l)
            P = r**2 * R**2
            norm = np.trapz(P, dx=dr)
            P_norm = P / norm
            
            mean_r = np.trapz(r * P_norm, dx=dr)
            mean_r2 = np.trapz(r**2 * P_norm, dx=dr)
            variance = mean_r2 - mean_r**2
            
            # 理论公式（原子单位）
            mean_r_theory = (3*n**2 - l*(l+1))/2 * a0
            r2_theory = n**2/2 * (5*n**2 + 1 - 3*l*(l+1)) * a0**2
            var_theory = r2_theory - mean_r_theory**2
            
            print(f"n={n}, l={l}")
            print(f"  <r> (num)  = {mean_r/a0:.6f} a0, theory = {mean_r_theory/a0:.6f} a0, rel. error = {abs(mean_r-mean_r_theory)/mean_r_theory:.2e}")
            print(f"  Var (num)  = {variance/a0**2:.6f} a0^2, theory = {var_theory/a0**2:.6f} a0^2, rel. error = {abs(variance-var_theory)/var_theory:.2e}")
            
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(r/a0, P_norm * a0, 'b-', label='Radial density')
            ax.axvline(mean_r/a0, color='r', linestyle='--', label=f'<r> = {mean_r/a0:.2f} a0')
            ax.axvline(mean_r_theory/a0, color='g', linestyle=':', label='Theory')
            ax.set_xlabel(r'$r / a_0$')
            ax.set_ylabel(r'$r^2 R^2$ (norm)')
            ax.set_title(f'Radial distribution moments (n={n}, l={l})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            if savefig:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {savefig}")
            return mean_r, variance

        def analyze_node_spacing(n, l, npoints=100000, savefig=None):
            """
            寻找径向波函数 R_nl 的节点，计算节点间距，并与经典周期关联
            """
            a0 = calc.a0
            r_max = calc.classical_turning_points(n, l)[1] * 2
            r = np.linspace(1e-6 * a0, r_max, npoints)
            R = calc.R_nl(r, n, l)
            sign_changes = np.where(np.diff(np.sign(R)))[0]
            fine_nodes = []
            for idx in sign_changes:
                r1, r2 = r[idx], r[idx+1]
                R1, R2 = R[idx], R[idx+1]
                if R2 != R1:
                    r0 = r1 - R1 * (r2 - r1) / (R2 - R1)
                    fine_nodes.append(r0)
                else:
                    fine_nodes.append(r1)
            nodes = np.array(fine_nodes)
            
            if len(nodes) < 2:
                print(f"n={n}, l={l} has only {len(nodes)} nodes, skipping.")
                return
            
            spacings = np.diff(nodes)
            avg_spacing = np.mean(spacings)
            
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(nodes[:-1]/a0, spacings/a0, 'bo-', markersize=3)
            ax.axhline(avg_spacing/a0, color='r', linestyle='--', label=f'Average = {avg_spacing/a0:.3f} a0')
            ax.set_xlabel(r'Node position $r_i / a_0$')
            ax.set_ylabel(r'Node spacing $\Delta r / a_0$')
            ax.set_title(f'Node spacing for $n={n}$, $l={l}$')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            if savefig:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {savefig}")
            return nodes, spacings

        def analyze_angular_width_vs_m(l_max=50, m_values=None, savefig=None):
            """
            研究角向分布宽度随 m 的变化，对于多个固定的 m
            """
            if m_values is None:
                m_values = [0, 1, 5, 10]
            l_vals = np.arange(1, l_max+1)
            fig, ax = plt.subplots(figsize=(8,5))
            for m_val in m_values:
                widths = []
                for l_val in l_vals:
                    if abs(m_val) > l_val:
                        continue
                    theta = np.linspace(0, np.pi, 5000)
                    Y2 = np.abs(calc.sph_harm_compat(m_val, l_val, 0.0, theta))**2
                    Y2 /= np.max(Y2)
                    # 寻找半高宽
                    i_max = np.argmax(Y2)
                    # 向左找半高
                    left_idx = i_max
                    while left_idx > 0 and Y2[left_idx] > 0.5:
                        left_idx -= 1
                    if left_idx == 0:
                        left = theta[0]
                    else:
                        t1, t2 = theta[left_idx], theta[left_idx+1]
                        y1, y2 = Y2[left_idx], Y2[left_idx+1]
                        left = t1 + (0.5 - y1) * (t2 - t1) / (y2 - y1)
                    # 向右找半高
                    right_idx = i_max
                    while right_idx < len(theta)-1 and Y2[right_idx] > 0.5:
                        right_idx += 1
                    if right_idx == len(theta)-1:
                        right = theta[-1]
                    else:
                        t1, t2 = theta[right_idx-1], theta[right_idx]
                        y1, y2 = Y2[right_idx-1], Y2[right_idx]
                        right = t1 + (0.5 - y1) * (t2 - t1) / (y2 - y1)
                    width = right - left
                    widths.append(width)
                ax.plot(l_vals[:len(widths)], widths, 'o-', label=f'm={m_val}')
            ax.set_xlabel('$l$')
            ax.set_ylabel('FWHM (rad)')
            ax.set_title('Angular width vs $l$ for different $m$')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            if savefig:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {savefig}")
            return fig

        def plot_wkb_vs_exact_radial(n, l, npoints=5000, savefig=None):
            """
            比较精确径向概率密度与 WKB 近似结果
            """
            a0 = calc.a0
            r_min, r_max = calc.classical_turning_points(n, l)
            r = np.linspace(r_min, r_max, npoints)
            # 精确
            R_exact = calc.R_nl(r, n, l)
            P_exact = r**2 * R_exact**2
            # WKB 近似
            if n - l - 1 > 500:
                R_wkb_full = calc.R_nl(r, n, l)  # 内部已使用 WKB
            else:
                log_coeff = 0.5 * (3 * np.log(2.0/(a0*n)) + calc.log_factorial(n-l-1) - np.log(2.0*n) - calc.log_factorial(n+l))
                rho = 2.0 * r / (a0 * n)
                lag_wkb = calc.laguerre_wkb_optimized(rho, n-l-1, 2*l+1)
                R_wkb_full = np.exp(log_coeff) * np.exp(-r/(a0*n)) * (rho**l) * lag_wkb
            P_wkb = r**2 * R_wkb_full**2
            
            # 归一化
            norm_exact = np.trapz(P_exact, r)
            if norm_exact > 0:
                P_exact /= norm_exact
            norm_wkb = np.trapz(P_wkb, r)
            if norm_wkb > 0:
                P_wkb /= norm_wkb
            
            fig, ax = plt.subplots(figsize=(8,5))
            ax.plot(r/a0, P_exact, 'b-', label='Exact')
            ax.plot(r/a0, P_wkb, 'r--', label='WKB')
            ax.set_xlabel(r'$r / a_0$')
            ax.set_ylabel('Radial probability density')
            ax.set_title(f'WKB vs Exact, $n={n}$, $l={l}$')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            if savefig:
                fig.savefig(savefig, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {savefig}")
            return fig

        # ========== 主菜单循环 ==========
        while True:
            print("\n" + "-"*50)
            print(f"Current quantum numbers: n={n}, l={l}, m={m}")
            print("Available plots:")
            print("1.  Linear Scale (|ψ|², with optional classical boundary)")
            print("2.  Logarithmic Scale (log10|ψ|²)")
            print("3.  Contour Plot (with optional numerical labels)")
            print("4.  3D Surface (|ψ|²)")
            print("5.  Radial Density (r²R²)")
            print("6.  Angular Distribution (|Y|², full color range)")
            print("7.  Enhanced 2D Map (adaptive color scale)")
            print("8.  2D Shell Density (4πr²|ψ|²) with cross-section selection")
            print("9.  3D Shell Density (4πr²|ψ|²)")
            print("10. Angular + Radial (combined subplots, full color range)")
            print("11. Half-Cut 3D Shell Density (4πr²|ψ|²)")
            print("12. Half-Cut 3D |ψ|² (adaptive color)")
            print("13. Realistic 3D Wavefunction (scatter, 4πr²|ψ|² coloring, multi-core)")
            print("\nOptions (numeric shortcuts in parentheses):")
            print("  - Enter plot numbers separated by spaces (e.g., '1 3 5')")
            print("  - 'all' (0) : generate all plots with default annotations (3D realistic uses 80³, no cut)")
            print("  - 'none' (-1): skip plotting and return to menu")
            print("  - 'new' (-2) : input new quantum numbers")
            print("  - 'exit' (-3): quit program")
            
            choice = input("\nYour choice: ").strip().lower()
            if choice == '0':
                choice = 'all'
            elif choice == '-1':
                choice = 'none'
            elif choice == '-2':
                choice = 'new'
            elif choice == '-3':
                choice = 'exit'
            if choice == 'exit':
                return
            if choice == 'new':
                break
            if choice == 'none':
                continue
            if choice == 'all':
                plot_linear_scale(show_classical_boundary=True, fig_num=1)
                plot_log_scale(fig_num=2)
                plot_contour(show_labels=True, fig_num=3)
                plot_3d_surface(quality='ultra', fig_num=4)
                plot_radial_density(show_classical=True, fig_num=5)
                plot_angular_distribution(fig_num=6)
                plot_enhanced_2d(show_white_contour=True, show_classical_boundary=True, show_info_box=True, fig_num=7)
                plot_shell_2d(show_classical_boundary=True, fig_num=8)
                plot_shell_3d(quality='ultra', fig_num=9)
                plot_angular_radial(fig_num=10)
                plot_shell_3d_half(quality='ultra', fig_num=11)
                plot_density_3d_half(quality='ultra', fig_num=12)
                plot_3d_realistic(resolution=80, cut_mode='none', num_cores=4, background='black', zoom_factor=1.0, fig_num=13)
                continue
            
            try:
                fig_numbers = [int(x) for x in choice.split()]
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces, or a keyword.")
                continue
            
            for fig_num in fig_numbers:
                if fig_num == 1:
                    ans = input("Show classical boundary? (y/n, default y): ").strip().lower()
                    show_boundary = ans != 'n'
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (title, axis labels, colorbar)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_linear_scale(show_classical_boundary=show_boundary, show_box=show_box, show_labels=show_labels, fig_num=1)
                elif fig_num == 2:
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (title, axis labels, colorbar)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_log_scale(show_box=show_box, show_labels=show_labels, fig_num=2)
                elif fig_num == 3:
                    ans = input("Show contour labels? (y/n, default y): ").strip().lower()
                    show_contour_labels = ans != 'n'
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show axis labels and title? (y/n, default y): ").strip().lower()
                    show_axis_labels = label_ans != 'n'
                    plot_contour(show_labels=show_contour_labels, show_box=show_box, show_axis_labels=show_axis_labels, fig_num=3)
                elif fig_num == 4:
                    print("Select 3D plot quality: (d)efault, (h)igh, (u)ltra (default u)")
                    q = input("Choice (d/h/u, default u): ").strip().lower()
                    if q == 'd':
                        qual = 'default'
                    elif q == 'h':
                        qual = 'high'
                    else:
                        qual = 'ultra'
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (title, axis labels, colorbar)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_3d_surface(quality=qual, show_box=show_box, show_labels=show_labels, fig_num=4)
                elif fig_num == 5:
                    ans = input("Show classical distribution and boundaries? (y/n, default y): ").strip().lower()
                    show_classical = ans != 'n'
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (title, axis labels)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_radial_density(show_classical=show_classical, show_box=show_box, show_labels=show_labels, fig_num=5)
                elif fig_num == 6:
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show title and colorbar? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_angular_distribution(show_box=show_box, show_labels=show_labels, fig_num=6)
                elif fig_num == 7:
                    ans1 = input("Show white contour lines? (y/n, default y): ").strip().lower()
                    w = ans1 != 'n'
                    ans2 = input("Show classical boundary? (y/n, default y): ").strip().lower()
                    b = ans2 != 'n'
                    ans3 = input("Show information box? (y/n, default y): ").strip().lower()
                    i = ans3 != 'n'
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (title, axis labels, colorbar)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_enhanced_2d(show_white_contour=w, show_classical_boundary=b, show_info_box=i,
                                     show_box=show_box, show_labels=show_labels, fig_num=7)
                elif fig_num == 8:
                    ans = input("Show classical boundary? (y/n, default y): ").strip().lower()
                    show_boundary = ans != 'n'
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (title, axis labels, colorbar)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_shell_2d(show_classical_boundary=show_boundary, show_box=show_box, show_labels=show_labels, fig_num=8)
                elif fig_num == 9:
                    print("Select 3D shell quality: (d)efault, (h)igh, (u)ltra (default u)")
                    q = input("Choice (d/h/u, default u): ").strip().lower()
                    if q == 'd':
                        qual = 'default'
                    elif q == 'h':
                        qual = 'high'
                    else:
                        qual = 'ultra'
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (title, axis labels, colorbar)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_shell_3d(quality=qual, show_box=show_box, show_labels=show_labels, fig_num=9)
                elif fig_num == 10:
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (titles, colorbars, axis labels)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_angular_radial(show_box=show_box, show_labels=show_labels, fig_num=10)
                elif fig_num == 11:
                    print("Select half-cut 3D shell quality: (d)efault, (h)igh, (u)ltra (default u)")
                    q = input("Choice (d/h/u, default u): ").strip().lower()
                    if q == 'd':
                        qual = 'default'
                    elif q == 'h':
                        qual = 'high'
                    else:
                        qual = 'ultra'
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (title, axis labels, colorbar)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_shell_3d_half(quality=qual, show_box=show_box, show_labels=show_labels, fig_num=11)
                elif fig_num == 12:
                    print("Select half-cut 3D |ψ|² quality: (d)efault, (h)igh, (u)ltra (default u)")
                    q = input("Choice (d/h/u, default u): ").strip().lower()
                    if q == 'd':
                        qual = 'default'
                    elif q == 'h':
                        qual = 'high'
                    else:
                        qual = 'ultra'
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (title, axis labels, colorbar)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    plot_density_3d_half(quality=qual, show_box=show_box, show_labels=show_labels, fig_num=12)
                elif fig_num == 13:
                    try:
                        res_input = input("Enter resolution for 3D scatter (e.g., 60, 80, 100; default 80): ").strip()
                        if res_input == '':
                            res = 80
                        else:
                            res = int(res_input)
                            if res <= 0:
                                res = 80
                    except:
                        res = 80
                    cores_input = input("Number of CPU cores to use (1/2/4/8, default 4): ").strip()
                    if cores_input in ['1','2','4','8']:
                        cores = int(cores_input)
                    else:
                        cores = 4
                    print("Select cut mode:")
                    print("  none - no cut")
                    print("  half_x - keep x>=0")
                    print("  half_y - keep y>=0")
                    print("  half_z - keep z>=0")
                    print("  quadrant - keep selected xy-plane quadrants")
                    print("  octant - keep selected octants (based on x,y,z signs)")
                    print("  hexadecant - keep selected 16 sectors (phi,theta)")
                    cut_mode = input("Cut mode (none/half_x/half_y/half_z/quadrant/octant/hexadecant, default none): ").strip().lower()
                    if cut_mode not in ['none','half_x','half_y','half_z','quadrant','octant','hexadecant']:
                        cut_mode = 'none'
                    which_parts = None
                    if cut_mode in ['quadrant','octant','hexadecant']:
                        parts_input = input(f"Enter parts to keep (0-{ {'quadrant':3, 'octant':7, 'hexadecant':15}[cut_mode] }, separated by commas, default all): ").strip()
                        if parts_input:
                            try:
                                which_parts = [int(p.strip()) for p in parts_input.split(',')]
                            except:
                                print("Invalid parts, using all.")
                                which_parts = None
                    box_ans = input("Show axes box? (y/n, default y): ").strip().lower()
                    show_box = box_ans != 'n'
                    label_ans = input("Show labels (title, axis labels, colorbar)? (y/n, default y): ").strip().lower()
                    show_labels = label_ans != 'n'
                    bg_input = input("Background color? (1=black, 2=white, default 1): ").strip()
                    background = 'black' if bg_input != '2' else 'white'
                    zoom_input = input("Zoom factor (e.g., 0.5 for zoom in, 2 for zoom out; default 1.0): ").strip()
                    try:
                        zoom_factor = float(zoom_input) if zoom_input != '' else 1.0
                    except ValueError:
                        zoom_factor = 1.0
                    plot_3d_realistic(resolution=res, cut_mode=cut_mode, which_parts=which_parts,
                                      num_cores=cores, show_box=show_box, show_labels=show_labels,
                                      background=background, zoom_factor=zoom_factor, fig_num=13)
if __name__ == "__main__":
    run_cli()
