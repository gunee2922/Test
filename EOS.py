import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.optimize import root_scalar, curve_fit
import matplotlib.gridspec as gridspec
import warnings
from collections import defaultdict

# EOS functions

def smart_pressure_sort(P_data, *args):
    """
    Smart pressure sorting for non-sequential pressure data.
    Sorts pressure data and corresponding arrays while maintaining relationships.
    
    Args:
        P_data: pressure array
        *args: additional arrays (V, sigma_P, sigma_V, etc.) to sort in same order
    
    Returns:
        tuple: (sorted_P, sorted_arg1, sorted_arg2, ...)
    """
    if len(P_data) == 0:
        return tuple([np.array([]) for _ in range(len(args) + 1)])
    
    # Get sorting indices
    sort_indices = np.argsort(P_data)
    
    # Sort pressure and all associated arrays
    sorted_arrays = [P_data[sort_indices]]
    for arr in args:
        if arr is not None and len(arr) == len(P_data):
            sorted_arrays.append(arr[sort_indices])
        else:
            sorted_arrays.append(arr)
    
    return tuple(sorted_arrays)

def apply_physical_constraints(params, eos_type='Birch-Murnaghan'):
    """
    Apply physical constraints to EOS parameters.
    
    Args:
        params: list/array of [V0, K0, Kp] parameters
        eos_type: type of EOS equation
    
    Returns:
        constrained parameters
    """
    V0, K0, Kp = params
    
    # Physical constraints
    V0 = max(V0, 1.0)      # V0 should be positive and reasonable
    K0 = max(K0, 1.0)      # K0 should be positive
    K0 = min(K0, 1000.0)   # K0 shouldn't be unreasonably large
    
    if eos_type == 'Birch-Murnaghan':
        Kp = max(Kp, 0.5)  # K' should be positive for BM
        Kp = min(Kp, 10.0) # K' typically between 0.5-10
    else:  # Vinet
        Kp = max(Kp, 0.5)
        Kp = min(Kp, 12.0)
    
    return [V0, K0, Kp]

def adaptive_bracket_selection(eos_func, pressure, V0_guess, K0_guess, Kp_guess, bracket_factor=0.1):
    """
    Adaptive bracket selection for safe volume calculation.
    
    Args:
        eos_func: EOS function
        pressure: target pressure
        V0_guess, K0_guess, Kp_guess: EOS parameters
        bracket_factor: factor for bracket adjustment
    
    Returns:
        tuple: (lower_bound, upper_bound) for root finding
    """
    try:
        # Start with standard brackets
        lower = V0_guess * (0.3 - bracket_factor)
        upper = V0_guess * (1.2 + bracket_factor)
        
        # Check if brackets have different signs
        f_lower = eos_func(lower, V0_guess, K0_guess, Kp_guess) - pressure
        f_upper = eos_func(upper, V0_guess, K0_guess, Kp_guess) - pressure
        
        # If same sign, expand brackets
        iteration = 0
        max_iterations = 10
        
        while f_lower * f_upper > 0 and iteration < max_iterations:
            if abs(f_lower) < abs(f_upper):
                lower *= 0.8
            else:
                upper *= 1.2
            
            f_lower = eos_func(lower, V0_guess, K0_guess, Kp_guess) - pressure
            f_upper = eos_func(upper, V0_guess, K0_guess, Kp_guess) - pressure
            iteration += 1
        
        # Ensure reasonable bounds
        lower = max(lower, V0_guess * 0.1)  # Not too small
        upper = min(upper, V0_guess * 2.0)  # Not too large
        
        return lower, upper
    
    except:
        # Fallback to conservative brackets
        return V0_guess * 0.3, V0_guess * 1.2

def safe_volume_calculation(eos_func, pressure, V0, K0, Kp, max_attempts=3):
    """
    Safe volume calculation with multiple fallback strategies.
    
    Args:
        eos_func: EOS function (birch_murnaghan_P or vinet_P)
        pressure: target pressure
        V0, K0, Kp: EOS parameters
        max_attempts: maximum number of attempts with different strategies
    
    Returns:
        dict: {'volume': calculated_volume, 'success': True/False, 'method': method_used}
    """
    for attempt in range(max_attempts):
        try:
            # Apply physical constraints
            V0_c, K0_c, Kp_c = apply_physical_constraints([V0, K0, Kp])
            
            # Get adaptive brackets
            lower, upper = adaptive_bracket_selection(eos_func, pressure, V0_c, K0_c, Kp_c, 
                                                    bracket_factor=attempt * 0.05)
            
            # Try root finding
            sol = root_scalar(
                lambda V: eos_func(V, V0_c, K0_c, Kp_c) - pressure,
                bracket=[lower, upper],
                method='brentq',  # More robust than bisect
                xtol=1e-12,
                rtol=1e-10
            )
            
            if sol.converged and sol.root > 0:
                return {
                    'volume': sol.root,
                    'success': True,
                    'method': f'brentq_attempt_{attempt+1}'
                }
        
        except Exception as e:
            # Try with different method or parameters
            if attempt < max_attempts - 1:
                continue
            else:
                # Last attempt: try with very conservative brackets
                try:
                    conservative_lower = V0 * 0.5
                    conservative_upper = V0 * 1.1
                    
                    sol = root_scalar(
                        lambda V: eos_func(V, V0, K0, Kp) - pressure,
                        bracket=[conservative_lower, conservative_upper],
                        method='bisect'
                    )
                    
                    return {
                        'volume': sol.root,
                        'success': True,
                        'method': 'conservative_bisect'
                    }
                except:
                    # Final fallback: linear approximation
                    try:
                        V_approx = V0 * (1 - pressure / (3 * K0))  # Simple linear approximation
                        return {
                            'volume': max(V_approx, V0 * 0.1),
                            'success': False,
                            'method': 'linear_approximation'
                        }
                    except:
                        return {
                            'volume': V0 * 0.9,
                            'success': False,
                            'method': 'fallback_default'
                        }

def robust_eos_fitting(V_data, P_data, sigma_P=None, eos_type='Birch-Murnaghan', 
                      max_iterations=10, tolerance=1e-8):
    """
    Robust EOS fitting with multiple initial guesses and optimization strategies.
    
    Args:
        V_data, P_data: volume and pressure data
        sigma_P: pressure uncertainties
        eos_type: 'Birch-Murnaghan' or 'Vinet'
        max_iterations: maximum number of fitting attempts
        tolerance: convergence tolerance
    
    Returns:
        dict: fitting results with best parameters, errors, and quality metrics
    """
    # Sort data by pressure
    sorted_data = smart_pressure_sort(P_data, V_data, sigma_P if sigma_P is not None else None)
    P_sorted, V_sorted = sorted_data[0], sorted_data[1]
    sigma_P_sorted = sorted_data[2] if sigma_P is not None else None
    
    # Choose EOS function
    if eos_type == 'Birch-Murnaghan':
        eos_func = birch_murnaghan_P
        fit_func = birch_murnaghan_fit_func
    else:
        eos_func = vinet_P
        fit_func = vinet_fit_func
    
    # Generate multiple initial guesses
    V0_guesses = [V_sorted.max(), V_sorted.max() * 1.05, V_sorted.max() * 0.95]
    K0_guesses = [100, 150, 200, 250]  # Range of typical bulk moduli
    Kp_guesses = [3.5, 4.0, 4.5, 5.0]  # Range of typical K'
    
    best_result = None
    best_chisq = np.inf
    all_attempts = []
    
    for V0_guess in V0_guesses:
        for K0_guess in K0_guesses:
            for Kp_guess in Kp_guesses:
                try:
                    # Apply constraints to initial guess
                    initial_guess = apply_physical_constraints([V0_guess, K0_guess, Kp_guess], eos_type)
                    
                    # Perform fitting
                    if sigma_P_sorted is not None:
                        popt, pcov = curve_fit(
                            fit_func, V_sorted, P_sorted,
                            p0=initial_guess,
                            sigma=sigma_P_sorted,
                            absolute_sigma=True,
                            maxfev=5000
                        )
                    else:
                        popt, pcov = curve_fit(
                            fit_func, V_sorted, P_sorted,
                            p0=initial_guess,
                            maxfev=5000
                        )
                    
                    # Apply constraints to fitted parameters
                    popt = apply_physical_constraints(popt, eos_type)
                    
                    # Calculate parameter errors
                    perr = np.sqrt(np.diag(pcov))
                    
                    # Calculate quality metrics
                    P_pred = eos_func(V_sorted, *popt)
                    residuals = P_sorted - P_pred
                    
                    # R-squared
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((P_sorted - np.mean(P_sorted))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Chi-squared
                    if sigma_P_sorted is not None:
                        chi_squared = np.sum((residuals / sigma_P_sorted)**2)
                        reduced_chi_squared = chi_squared / (len(P_sorted) - 3)  # 3 parameters
                    else:
                        chi_squared = ss_res
                        reduced_chi_squared = chi_squared / (len(P_sorted) - 3)
                    
                    # RMS error
                    rms_error = np.sqrt(np.mean(residuals**2))
                    
                    result = {
                        'params': popt,
                        'errors': perr,
                        'covariance': pcov,
                        'r_squared': r_squared,
                        'chi_squared': chi_squared,
                        'reduced_chi_squared': reduced_chi_squared,
                        'rms_error': rms_error,
                        'residuals': residuals,
                        'initial_guess': initial_guess,
                        'n_points': len(P_sorted)
                    }
                    
                    all_attempts.append(result)
                    
                    # Check if this is the best fit so far
                    if reduced_chi_squared < best_chisq:
                        best_chisq = reduced_chi_squared
                        best_result = result
                
                except Exception as e:
                    # Log failed attempt but continue
                    all_attempts.append({
                        'failed': True,
                        'error': str(e),
                        'initial_guess': [V0_guess, K0_guess, Kp_guess]
                    })
                    continue
    
    if best_result is None:
        raise RuntimeError("All fitting attempts failed")
    
    # Add metadata about fitting process
    best_result['fitting_attempts'] = len(all_attempts)
    best_result['successful_attempts'] = len([a for a in all_attempts if 'failed' not in a])
    best_result['eos_type'] = eos_type
    
    return best_result

def extrapolate_to_zero_pressure(eos_func, params, max_volume_factor=1.5):
    """
    Extrapolate EOS to zero pressure to find V0.
    
    Args:
        eos_func: EOS function
        params: [V0, K0, Kp] parameters
        max_volume_factor: maximum factor for volume search
    
    Returns:
        V0 at P=0
    """
    V0, K0, Kp = params
    
    try:
        # Find volume at P=0
        result = safe_volume_calculation(eos_func, 0.0, V0, K0, Kp)
        if result['success']:
            return result['volume']
        else:
            # Fallback: use fitted V0 parameter
            return V0
    except:
        return V0

def calculate_quality_metrics(P_data, P_predicted, sigma_P=None, n_params=3):
    """
    Calculate quality metrics for EOS fitting.
    
    Args:
        P_data: observed pressure data
        P_predicted: predicted pressure from EOS
        sigma_P: pressure uncertainties
        n_params: number of fitted parameters
    
    Returns:
        dict: quality metrics (R², χ², RMS, etc.)
    """
    residuals = P_data - P_predicted
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((P_data - np.mean(P_data))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Chi-squared
    if sigma_P is not None:
        chi_squared = np.sum((residuals / sigma_P)**2)
        reduced_chi_squared = chi_squared / (len(P_data) - n_params)
    else:
        chi_squared = ss_res
        reduced_chi_squared = chi_squared / (len(P_data) - n_params)
    
    # RMS error
    rms_error = np.sqrt(np.mean(residuals**2))
    
    # Mean absolute error
    mae = np.mean(np.abs(residuals))
    
    # Maximum absolute error
    max_abs_error = np.max(np.abs(residuals))
    
    return {
        'r_squared': r_squared,
        'chi_squared': chi_squared,
        'reduced_chi_squared': reduced_chi_squared,
        'rms_error': rms_error,
        'mae': mae,
        'max_abs_error': max_abs_error,
        'residuals': residuals
    }

def birch_murnaghan_axial(P, L0, M0, Mp, Mpp=0.0, use_mpp=False):
    x = P / M0
    if use_mpp:
        return L0 * (1 - x + ((Mp - 1) / 2) * x**2 + (Mpp / 6) * x**3)
    return L0 * (1 - x + ((Mp - 1) / 2) * x**2)

# Volume-based EOS P(V) and Vinet

def birch_murnaghan_P(V, V0, K0, Kp):
    eta = (V0 / V)**(1/3)
    return (3/2) * K0 * (eta**7 - eta**5) * (1 + (3/4)*(Kp - 4)*(eta**2 - 1))

def vinet_P(V, V0, K0, Kp):
    eta = (V / V0)**(1/3)
    return 3 * K0 * (1 - eta) * np.exp((3/2)*(Kp - 1)*(1 - eta))

# EOS fitting functions for error band calculation
def birch_murnaghan_fit_func(V, V0, K0, Kp):
    return birch_murnaghan_P(V, V0, K0, Kp)

def vinet_fit_func(V, V0, K0, Kp):
    return vinet_P(V, V0, K0, Kp)

# EOS equation dictionary
EOS_EQUATIONS = {
    "Birch-Murnaghan": "P = (3/2)K₀(η⁷ - η⁵)[1 + (3/4)(K′ - 4)(η² - 1)],\nη = (V₀/V)^(1/3)",
    "Vinet": "P = 3K₀(1 - η)·exp[(3/2)(K′-1)(1 - η)],\nη = (V/V₀)^(1/3)"
}
AXIAL_BM_EQUATION = "L = L₀·[1 - x + ((M′ - 1)/2)x² + (M″/6)x³], x = P/M₀"  # for axial BM
AXIAL_BM_EQUATION_NO_MPP = "L = L₀·[1 - x + ((M′ - 1)/2)x²], x = P/M₀"      # without Mpp

# Crystal system definitions
CRYSTAL_SYSTEMS = {
    'Cubic': ['a'],
    'Tetragonal': ['a', 'c'], 
    'Trigonal': ['a', 'c'],
    'Hexagonal': ['a', 'c'],
    'Orthorhombic': ['a', 'b', 'c'],
    'Monoclinic': ['a', 'b', 'c'],
    'Triclinic': ['a', 'b', 'c']
}

# For lattice parameter and d-spacing fitting
def fit_multi_lattice_eos(P_data, lattice_data, sigma_data, axes, eos_type='Birch-Murnaghan'):
    """Fit EOS to multiple lattice parameter data"""
    results = {}
    
    for axis in axes:
        if axis not in lattice_data:
            continue
            
        L_data = lattice_data[axis]
        sigma_L = sigma_data.get(f'sigma_{axis}', np.ones_like(L_data) * 0.001)
        sigma_P = sigma_data.get('sigma_P', np.ones_like(P_data) * 0.1)
        
        try:
            if eos_type == 'Birch-Murnaghan':
                # Initial guess for axial BM
                L0_guess = L_data.max()
                M0_guess = 120.0  # GPa
                Mp_guess = 4.0
                
                def axial_bm_func(P, L0, M0, Mp):
                    return birch_murnaghan_axial(P, L0, M0, Mp, use_mpp=False)
                
                popt, pcov = curve_fit(axial_bm_func, P_data, L_data, 
                                     p0=[L0_guess, M0_guess, Mp_guess],
                                     sigma=sigma_P if len(sigma_P) == len(P_data) else None,
                                     absolute_sigma=True)
                
                perr = np.sqrt(np.diag(pcov))
                
                results[axis] = {
                    'params': popt,
                    'errors': perr,
                    'covariance': pcov,
                    'function': axial_bm_func,
                    'param_names': ['L0', 'M0', 'Mp']
                }
                
            else:  # Vinet - convert to volume-based fitting for each axis
                # For non-cubic, treat each axis independently using axial compression
                L0_guess = L_data.max()
                M0_guess = 120.0
                Mp_guess = 4.0
                
                def axial_bm_func(P, L0, M0, Mp):
                    return birch_murnaghan_axial(P, L0, M0, Mp, use_mpp=False)
                
                popt, pcov = curve_fit(axial_bm_func, P_data, L_data, 
                                     p0=[L0_guess, M0_guess, Mp_guess],
                                     sigma=sigma_P if len(sigma_P) == len(P_data) else None,
                                     absolute_sigma=True)
                
                perr = np.sqrt(np.diag(pcov))
                
                results[axis] = {
                    'params': popt,
                    'errors': perr,
                    'covariance': pcov,
                    'function': axial_bm_func,
                    'param_names': ['L0', 'M0', 'Mp']
                }
                
        except Exception as e:
            raise ValueError(f"Fitting failed for axis {axis}: {str(e)}")
    
    return results

def calculate_dspacing(h, k, l, lattice_params, crystal_system):
    """Calculate d-spacing from Miller indices and lattice parameters"""
    if crystal_system == 'Cubic':
        a = lattice_params['a']
        return a / np.sqrt(h**2 + k**2 + l**2)
    
    elif crystal_system in ['Tetragonal', 'Trigonal']:
        a = lattice_params['a']
        c = lattice_params['c']
        return 1 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)
    
    elif crystal_system == 'Hexagonal':
        a = lattice_params['a']
        c = lattice_params['c']
        return 1 / np.sqrt((4/3) * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)
    
    elif crystal_system in ['Orthorhombic', 'Monoclinic', 'Triclinic']:
        a = lattice_params['a']
        b = lattice_params['b']
        c = lattice_params['c']
        # Simplified orthogonal approximation
        return 1 / np.sqrt(h**2 / a**2 + k**2 / b**2 + l**2 / c**2)
    
    else:
        raise ValueError(f"Unsupported crystal system: {crystal_system}")

class EOSPlotApp:
    def __init__(self, master):
        self.master = master
        master.title("EOS Plotting Tool - Enhanced Version")
        
        # Set initial window size and make it resizable
        master.geometry("1400x900")  # 충분히 큰 초기 크기
        master.minsize(1200, 700)   # 최소 크기 설정
        master.state('normal')      # 정상 창 모드
        
        # Enhanced data storage for separated import system
        self.eos_data = {}  # Main curve fitting data
        self.error_data = {}  # Error band calculation data
        self.fitting_status = {}  # Real-time fitting status
        self.quality_metrics = {}  # Quality metrics storage
        
        # Monte Carlo 결과 저장용
        self.mc_results = {}
        
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True)

        # Crystal System tab
        self.crystal_frame = ttk.Frame(self.notebook)
        self.setup_crystal_ui()
        self.notebook.add(self.crystal_frame, text='Crystal System EOS')

        # Volume/Density tab
        self.volume_frame = ttk.Frame(self.notebook)
        self.setup_volume_ui()
        self.notebook.add(self.volume_frame, text='Volume/Density EOS')

    def update_fitting_status(self, curve_name, status_text, metrics=None, color="black"):
        """
        Update fitting status with real-time quality metrics display.
        
        Args:
            curve_name: name of the curve being fitted
            status_text: status message
            metrics: dict of quality metrics (R², χ², etc.)
            color: text color for status
        """
        # Store status
        self.fitting_status[curve_name] = {
            'status': status_text,
            'timestamp': pd.Timestamp.now(),
            'color': color
        }
        
        # Store metrics if provided
        if metrics:
            self.quality_metrics[curve_name] = metrics
            
            # Append metrics to status text
            metrics_text = f"\nR² = {metrics.get('r_squared', 0):.4f}"
            if 'reduced_chi_squared' in metrics:
                metrics_text += f", χ²_red = {metrics['reduced_chi_squared']:.4f}"
            if 'rms_error' in metrics:
                metrics_text += f", RMS = {metrics['rms_error']:.4f} GPa"
            
            status_text += metrics_text
        
        # Update UI if status label exists
        if hasattr(self, 'curve_entries'):
            for entry in self.curve_entries:
                if entry['name'].get() == curve_name and 'status_label' in entry:
                    entry['status_label'].config(text=status_text, foreground=color)
                    break

    def import_eos_data_for_curve(self, curve_idx):
        """Import main EOS data for curve fitting"""
        curve_name = self.curve_entries[curve_idx]['name'].get()
        
        fname = filedialog.askopenfilename(
            title=f"Import EOS Data for {curve_name}",
            filetypes=[("Excel files", "*.xlsx"), ("Excel files", "*.xls")]
        )
        if not fname:
            return
            
        try:
            df = pd.read_excel(fname)
            
            # Check required columns
            required_cols = {'P', 'V'}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                messagebox.showerror(
                    "Column Error", 
                    f"Excel file must contain columns: {', '.join(required_cols)}\n"
                    f"Missing columns: {', '.join(missing_cols)}\n\n"
                    f"Optional columns: sigma_P, sigma_V (for uncertainties)"
                )
                return
            
            # Add default uncertainties if not present
            if 'sigma_P' not in df.columns:
                df['sigma_P'] = df['P'] * 0.01  # 1% default uncertainty
            if 'sigma_V' not in df.columns:
                df['sigma_V'] = df['V'] * 0.01  # 1% default uncertainty
            
            # Sort by pressure using smart_pressure_sort
            sorted_data = smart_pressure_sort(df['P'].values, df['V'].values, 
                                            df['sigma_P'].values, df['sigma_V'].values)
            df_sorted = pd.DataFrame({
                'P': sorted_data[0],
                'V': sorted_data[1], 
                'sigma_P': sorted_data[2],
                'sigma_V': sorted_data[3]
            })
            
            # Store in enhanced data system
            self.eos_data[curve_name] = df_sorted
            
            # Update status
            self.update_fitting_status(
                curve_name, 
                f"EOS Data: {len(df_sorted)} points loaded (P: {df_sorted['P'].min():.1f}-{df_sorted['P'].max():.1f} GPa)",
                color="green"
            )
            
            # Also store in old system for compatibility
            self.curve_entries[curve_idx]['data_df'] = df_sorted
            self.curve_entries[curve_idx]['data_status_label'].config(
                text=f"EOS: {len(df_sorted)} points", 
                foreground="green"
            )
            
            messagebox.showinfo("Import Successful", 
                              f"Loaded {len(df_sorted)} EOS data points for {curve_name}\n"
                              f"Pressure range: {df_sorted['P'].min():.1f} - {df_sorted['P'].max():.1f} GPa\n"
                              f"Data automatically sorted by pressure.")
            
        except Exception as e:
            self.update_fitting_status(curve_name, f"EOS Data import failed: {str(e)}", color="red")
            messagebox.showerror("Import Error", f"Failed to load EOS data:\n{str(e)}")

    def import_error_data_for_curve(self, curve_idx):
        """Import separate error data for error band calculations"""
        curve_name = self.curve_entries[curve_idx]['name'].get()
        
        fname = filedialog.askopenfilename(
            title=f"Import Error Data for {curve_name}",
            filetypes=[("Excel files", "*.xlsx"), ("Excel files", "*.xls")]
        )
        if not fname:
            return
            
        try:
            df = pd.read_excel(fname)
            
            # Check required columns for error data
            required_cols = {'P', 'V', 'sigma_P', 'sigma_V'}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                messagebox.showerror(
                    "Column Error", 
                    f"Error data file must contain columns: {', '.join(required_cols)}\n"
                    f"Missing columns: {', '.join(missing_cols)}"
                )
                return
            
            # Sort by pressure
            sorted_data = smart_pressure_sort(df['P'].values, df['V'].values, 
                                            df['sigma_P'].values, df['sigma_V'].values)
            df_sorted = pd.DataFrame({
                'P': sorted_data[0],
                'V': sorted_data[1], 
                'sigma_P': sorted_data[2],
                'sigma_V': sorted_data[3]
            })
            
            # Store error data separately
            self.error_data[curve_name] = df_sorted
            
            # Update status
            self.update_fitting_status(
                curve_name, 
                f"Error Data: {len(df_sorted)} points loaded",
                color="blue"
            )
            
            # Update UI
            if 'error_status_label' in self.curve_entries[curve_idx]:
                self.curve_entries[curve_idx]['error_status_label'].config(
                    text=f"Error: {len(df_sorted)} points", 
                    foreground="blue"
                )
            
            messagebox.showinfo("Import Successful", 
                              f"Loaded {len(df_sorted)} error data points for {curve_name}")
            
        except Exception as e:
            self.update_fitting_status(curve_name, f"Error data import failed: {str(e)}", color="red")
            messagebox.showerror("Import Error", f"Failed to load error data:\n{str(e)}")

    def import_data_for_curve(self, curve_idx):
        """Legacy data import function - redirects to enhanced EOS data import for compatibility"""
        return self.import_eos_data_for_curve(curve_idx)

    # ---- Crystal UI (Updated for multi-lattice experimental data fitting) ----
    def setup_crystal_ui(self):
        frame = self.crystal_frame
        
        # Create main container with scrollbar for Crystal tab
        main_container = ttk.Frame(frame)
        main_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Canvas and scrollbar for scrolling
        canvas = tk.Canvas(main_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 마우스 휠 스크롤 활성화
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Use scrollable_frame for the rest of the UI
        frame = scrollable_frame
        
        # Title section
        title_section = ttk.Frame(frame)
        title_section.grid(row=0, column=0, columnspan=6, sticky='ew', padx=15, pady=15)
        
        title_label = tk.Label(title_section, text="Crystal System EOS - Multi-Lattice Parameter Fitting", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Crystal system selection
        system_section = ttk.LabelFrame(frame, text="Crystal System Selection")
        system_section.grid(row=1, column=0, columnspan=6, sticky='ew', padx=15, pady=10)
        
        system_title = tk.Label(system_section, text="Select Crystal System", font=("Arial", 14, "bold"))
        system_title.grid(row=0, column=0, columnspan=6, pady=10)
        
        ttk.Label(system_section, text="Crystal System:", font=("Arial", 12, "bold")).grid(row=1, column=0, padx=15, pady=8, sticky='w')
        
        self.crystal_system_var = tk.StringVar(value='Cubic')
        crystal_system_menu = ttk.Combobox(system_section, values=list(CRYSTAL_SYSTEMS.keys()), 
                                         textvariable=self.crystal_system_var, state='readonly', width=20,
                                         font=("Arial", 12))
        crystal_system_menu.grid(row=1, column=1, padx=15, pady=8, sticky='w')
        crystal_system_menu.bind("<<ComboboxSelected>>", self.update_required_columns_info)
        
        # Required columns info
        self.columns_info_label = ttk.Label(system_section, text="", foreground="blue", 
                                          font=("Arial", 11), wraplength=800)
        self.columns_info_label.grid(row=2, column=0, columnspan=6, padx=15, pady=8)
        
        # Data import section
        import_section = ttk.LabelFrame(frame, text="Data Import")
        import_section.grid(row=2, column=0, columnspan=6, sticky='ew', padx=15, pady=10)
        
        import_title = tk.Label(import_section, text="Import Experimental Data", font=("Arial", 14, "bold"))
        import_title.grid(row=0, column=0, columnspan=6, pady=10)
        
        ttk.Button(import_section, text="Import Excel Data", command=self.import_crystal_data,
                  width=20).grid(row=1, column=0, padx=15, pady=10)
        
        self.crystal_data_status = ttk.Label(import_section, text="No data loaded", 
                                           foreground="gray", font=("Arial", 12))
        self.crystal_data_status.grid(row=1, column=1, padx=15, pady=10, sticky='w')
        
        # EOS selection and settings
        settings_section = ttk.LabelFrame(frame, text="EOS Settings")
        settings_section.grid(row=3, column=0, columnspan=6, sticky='ew', padx=15, pady=10)
        settings_section.columnconfigure(1, weight=1)
        settings_section.columnconfigure(3, weight=1)
        settings_section.columnconfigure(5, weight=1)
        
        settings_title = tk.Label(settings_section, text="Analysis Settings", font=("Arial", 14, "bold"))
        settings_title.grid(row=0, column=0, columnspan=6, pady=10)
        
        # EOS type selection
        ttk.Label(settings_section, text="EOS Type:", font=("Arial", 12, "bold")).grid(row=1, column=0, padx=15, pady=8, sticky='w')
        self.crystal_eos_var = tk.StringVar(value='Birch-Murnaghan')
        crystal_eos_menu = ttk.Combobox(settings_section, values=['Birch-Murnaghan', 'Vinet'], 
                                       textvariable=self.crystal_eos_var, state='readonly', width=20,
                                       font=("Arial", 12))
        crystal_eos_menu.grid(row=1, column=1, padx=15, pady=8, sticky='w')
        crystal_eos_menu.bind("<<ComboboxSelected>>", self.update_crystal_equation_label)
        
        # Pressure range for extrapolation
        ttk.Label(settings_section, text="Min Pressure (GPa):", font=("Arial", 12)).grid(row=2, column=0, padx=15, pady=8, sticky='w')
        self.crystal_Pmin_e = ttk.Entry(settings_section, width=12, font=("Arial", 12))
        self.crystal_Pmin_e.insert(0,'0')
        self.crystal_Pmin_e.grid(row=2, column=1, padx=15, pady=8, sticky='w')
        
        ttk.Label(settings_section, text="Max Pressure (GPa):", font=("Arial", 12)).grid(row=2, column=2, padx=15, pady=8, sticky='w')
        self.crystal_Pmax_e = ttk.Entry(settings_section, width=12, font=("Arial", 12))
        self.crystal_Pmax_e.insert(0,'50')
        self.crystal_Pmax_e.grid(row=2, column=3, padx=15, pady=8, sticky='w')
        
        ttk.Label(settings_section, text="Number of Points:", font=("Arial", 12)).grid(row=2, column=4, padx=15, pady=8, sticky='w')
        self.crystal_npts_e = ttk.Entry(settings_section, width=12, font=("Arial", 12))
        self.crystal_npts_e.insert(0,'200')
        self.crystal_npts_e.grid(row=2, column=5, padx=15, pady=8, sticky='w')
        
        # Options
        options_frame = ttk.Frame(settings_section)
        options_frame.grid(row=3, column=0, columnspan=6, pady=10)
        
        self.crystal_swap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Swap Axes (X ↔ Y)", variable=self.crystal_swap_var,
                       ).grid(row=0, column=0, padx=20)
        
        self.show_data_points_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show Data Points", variable=self.show_data_points_var,
                       ).grid(row=0, column=1, padx=20)
        
        # Miller indices for d-spacing (if applicable)
        miller_section = ttk.LabelFrame(frame, text="d-spacing Calculation")
        miller_section.grid(row=4, column=0, columnspan=6, sticky='ew', padx=15, pady=10)
        
        miller_title = tk.Label(miller_section, text="Miller Indices (for d-spacing calculation)", font=("Arial", 14, "bold"))
        miller_title.grid(row=0, column=0, columnspan=6, pady=10)
        
        ttk.Label(miller_section, text="h k l:", font=("Arial", 12)).grid(row=1, column=0, padx=15, pady=8, sticky='w')
        miller_frame = ttk.Frame(miller_section)
        miller_frame.grid(row=1, column=1, padx=15, pady=8, sticky='w')
        
        self.crystal_h_e = ttk.Entry(miller_frame, width=8, font=("Arial", 12))
        self.crystal_h_e.insert(0,'1')
        self.crystal_h_e.grid(row=0, column=0, padx=3)
        
        self.crystal_k_e = ttk.Entry(miller_frame, width=8, font=("Arial", 12))
        self.crystal_k_e.insert(0,'0')
        self.crystal_k_e.grid(row=0, column=1, padx=3)
        
        self.crystal_l_e = ttk.Entry(miller_frame, width=8, font=("Arial", 12))
        self.crystal_l_e.insert(0,'0')
        self.crystal_l_e.grid(row=0, column=2, padx=3)
        
        self.calc_dspacing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(miller_section, text="Calculate d-spacing", variable=self.calc_dspacing_var,
                       ).grid(row=1, column=2, padx=15, pady=8)
        
        # Buttons section
        button_section = ttk.Frame(frame)
        button_section.grid(row=5, column=0, columnspan=6, pady=25)
        
        ttk.Button(button_section, text="Fit & Plot EOS", command=self.plot_crystal_eos, 
                  width=20).grid(row=0, column=0, padx=20)
        ttk.Button(button_section, text="Export Results", command=self.export_crystal_data, 
                  width=20).grid(row=0, column=1, padx=20)

        # EOS Equation display
        equation_section = ttk.LabelFrame(frame, text="Current EOS Equation")
        equation_section.grid(row=6, column=0, columnspan=6, sticky='ew', padx=15, pady=10)
        
        eq_title = tk.Label(equation_section, text="Current EOS Equation", font=("Arial", 14, "bold"))
        eq_title.pack(pady=10)
        
        self.crystal_eq_label = ttk.Label(equation_section, text="", foreground="blue", 
                                        font=("Arial", 12, "bold"), wraplength=900, justify='left')
        self.crystal_eq_label.pack(padx=15, pady=15)
        
        # Initialize
        self.crystal_data_df = None
        self.crystal_fit_results = None
        self.update_required_columns_info()
        self.update_crystal_equation_label()

    def update_required_columns_info(self, event=None):
        """Update the required columns information based on crystal system"""
        crystal_system = self.crystal_system_var.get()
        axes = CRYSTAL_SYSTEMS[crystal_system]
        
        required_cols = ['P'] + axes + ['sigma_P'] + [f'sigma_{axis}' for axis in axes]
        
        info_text = f"Required Excel columns for {crystal_system}:\n"
        info_text += f"• Basic: P (pressure in GPa), sigma_P (pressure uncertainty)\n"
        info_text += f"• Lattice parameters: {', '.join(axes)} (in Å)\n"
        info_text += f"• Uncertainties: {', '.join([f'sigma_{axis}' for axis in axes])} (in Å)\n"
        info_text += f"Total columns needed: {', '.join(required_cols)}"
        
        self.columns_info_label.config(text=info_text)

    def import_crystal_data(self):
        """Import experimental crystal data from Excel"""
        fname = filedialog.askopenfilename(
            title="Select Excel file with crystal data",
            filetypes=[("Excel files", "*.xlsx"), ("Excel files", "*.xls")]
        )
        if not fname:
            return
            
        try:
            df = pd.read_excel(fname)
            
            # Get required columns based on crystal system
            crystal_system = self.crystal_system_var.get()
            axes = CRYSTAL_SYSTEMS[crystal_system]
            
            required_cols = set(['P', 'sigma_P'] + axes + [f'sigma_{axis}' for axis in axes])
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                messagebox.showerror(
                    "Column Error", 
                    f"Excel file must contain columns for {crystal_system} system:\n"
                    f"Missing columns: {', '.join(missing_cols)}\n\n"
                    f"Required format:\n"
                    f"P: Pressure (GPa)\n"
                    f"Lattice parameters: {', '.join(axes)} (Å)\n"
                    f"sigma_P: Pressure uncertainty (GPa)\n"
                    f"Uncertainties: {', '.join([f'sigma_{axis}' for axis in axes])} (Å)"
                )
                return
            
            # Validate data
            for col in required_cols:
                if df[col].isna().any():
                    messagebox.showwarning("Data Warning", f"Column '{col}' contains missing values")
            
            # Store data
            self.crystal_data_df = df
            self.crystal_data_status.config(
                text=f"Loaded {len(df)} points ({crystal_system})", 
                foreground="green"
            )
            
            # Show data summary
            summary_text = f"Loaded {len(df)} crystal data points ({crystal_system})\n"
            summary_text += f"Pressure range: {df['P'].min():.1f} - {df['P'].max():.1f} GPa\n"
            for axis in axes:
                summary_text += f"Lattice {axis}: {df[axis].min():.3f} - {df[axis].max():.3f} Å\n"
            
            messagebox.showinfo("Import Successful", summary_text)
            
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to load data:\n{str(e)}")

    def update_crystal_equation_label(self, event=None):
        eos_type = self.crystal_eos_var.get()
        if eos_type == 'Birch-Murnaghan':
            self.crystal_eq_label.config(text=AXIAL_BM_EQUATION_NO_MPP)
        else:
            self.crystal_eq_label.config(text=AXIAL_BM_EQUATION_NO_MPP + "\n(Using axial compression model)")

    def plot_crystal_eos(self):
        if self.crystal_data_df is None:
            messagebox.showwarning("No Data", "Please import experimental data first.")
            return
        
        try:
            # Get settings
            crystal_system = self.crystal_system_var.get()
            axes = CRYSTAL_SYSTEMS[crystal_system]
            eos_type = self.crystal_eos_var.get()
            Pmin = float(self.crystal_Pmin_e.get())
            Pmax = float(self.crystal_Pmax_e.get())
            npts = int(self.crystal_npts_e.get())
            swap_axes = self.crystal_swap_var.get()
            show_data = self.show_data_points_var.get()
            
            # Prepare data
            df = self.crystal_data_df
            P_data = df['P'].values
            
            # Prepare lattice data
            lattice_data = {}
            sigma_data = {'sigma_P': df['sigma_P'].values}
            
            for axis in axes:
                lattice_data[axis] = df[axis].values
                sigma_data[f'sigma_{axis}'] = df[f'sigma_{axis}'].values
            
            # Fit EOS for each lattice parameter
            self.crystal_fit_results = fit_multi_lattice_eos(P_data, lattice_data, sigma_data, axes, eos_type)
            
            # Generate extrapolated data
            P_extrap = np.linspace(Pmin, Pmax, npts)
            
            # Calculate extrapolated lattice parameters
            lattice_extrap = {}
            for axis in axes:
                if axis in self.crystal_fit_results:
                    fit_result = self.crystal_fit_results[axis]
                    fitted_func = fit_result['function']
                    params = fit_result['params']
                    lattice_extrap[axis] = fitted_func(P_extrap, *params)
            
            # Calculate d-spacing if requested
            d_spacing_data = None
            d_spacing_extrap = None
            if self.calc_dspacing_var.get():
                try:
                    h = int(self.crystal_h_e.get())
                    k = int(self.crystal_k_e.get())
                    l = int(self.crystal_l_e.get())
                    
                    # Calculate d-spacing for experimental data
                    lattice_params_data = {axis: lattice_data[axis] for axis in axes}
                    d_spacing_data = calculate_dspacing(h, k, l, lattice_params_data, crystal_system)
                    
                    # Calculate d-spacing for extrapolated data
                    d_spacing_extrap = calculate_dspacing(h, k, l, lattice_extrap, crystal_system)
                    
                except Exception as e:
                    messagebox.showwarning("d-spacing Error", f"Failed to calculate d-spacing: {str(e)}")
            
            # Plotting
            n_plots = len(axes) + (1 if d_spacing_extrap is not None else 0)
            fig, axes_plots = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots), sharex=not swap_axes)
            
            if n_plots == 1:
                axes_plots = [axes_plots]
            
            # Plot each lattice parameter
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            
            for i, axis in enumerate(axes):
                ax = axes_plots[i]
                
                # Plot experimental data
                if show_data:
                    if swap_axes:
                        ax.errorbar(lattice_data[axis], P_data, 
                                   xerr=sigma_data[f'sigma_{axis}'], yerr=sigma_data['sigma_P'],
                                   fmt='o', color=colors[i % len(colors)], markersize=6, capsize=4, 
                                   label=f'{axis} Experimental Data')
                    else:
                        ax.errorbar(P_data, lattice_data[axis], 
                                   xerr=sigma_data['sigma_P'], yerr=sigma_data[f'sigma_{axis}'],
                                   fmt='o', color=colors[i % len(colors)], markersize=6, capsize=4, 
                                   label=f'{axis} Experimental Data')
                
                # Plot fitted curve
                if axis in lattice_extrap:
                    if swap_axes:
                        ax.plot(lattice_extrap[axis], P_extrap, '-', color=colors[i % len(colors)], 
                               linewidth=2, label=f'{axis} {eos_type} Fit')
                        ax.set_xlabel(f'Lattice Parameter {axis} (Å)', fontsize=12)
                        ax.set_ylabel('Pressure (GPa)', fontsize=12)
                    else:
                        ax.plot(P_extrap, lattice_extrap[axis], '-', color=colors[i % len(colors)], 
                               linewidth=2, label=f'{axis} {eos_type} Fit')
                        ax.set_xlabel('Pressure (GPa)', fontsize=12)
                        ax.set_ylabel(f'Lattice Parameter {axis} (Å)', fontsize=12)
                
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Lattice Parameter {axis}', fontsize=12)
            
            # Plot d-spacing if calculated
            if d_spacing_extrap is not None:
                ax = axes_plots[-1]
                h, k, l = int(self.crystal_h_e.get()), int(self.crystal_k_e.get()), int(self.crystal_l_e.get())
                
                if show_data and d_spacing_data is not None:
                    if swap_axes:
                        ax.errorbar(d_spacing_data, P_data, yerr=sigma_data['sigma_P'],
                                   fmt='s', color='purple', markersize=4, capsize=3, 
                                   label=f'd({h}{k}{l}) Data')
                    else:
                        ax.errorbar(P_data, d_spacing_data, xerr=sigma_data['sigma_P'],
                                   fmt='s', color='purple', markersize=4, capsize=3, 
                                   label=f'd({h}{k}{l}) Data')
                
                if swap_axes:
                    ax.plot(d_spacing_extrap, P_extrap, 'purple', linestyle='--', linewidth=2, 
                           label=f'd({h}{k}{l}) Fit')
                    ax.set_xlabel('d-spacing (Å)', fontsize=12)
                    ax.set_ylabel('Pressure (GPa)', fontsize=12)
                else:
                    ax.plot(P_extrap, d_spacing_extrap, 'purple', linestyle='--', linewidth=2, 
                           label=f'd({h}{k}{l}) Fit')
                    ax.set_xlabel('Pressure (GPa)', fontsize=12)
                    ax.set_ylabel('d-spacing (Å)', fontsize=12)
                
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_title(f'd-spacing ({h}{k}{l})', fontsize=12)
            
            plt.suptitle(f'{crystal_system} Crystal System: {eos_type} EOS Fit', fontsize=16)
            plt.tight_layout()
            plt.show()
            
            # Store results for export
            export_data = {'Pressure (GPa)': P_extrap}
            
            # Add fitted lattice parameters
            for axis in axes:
                if axis in lattice_extrap:
                    export_data[f'Lattice_{axis} (Å)'] = lattice_extrap[axis]
            
            # Add d-spacing if calculated
            if d_spacing_extrap is not None:
                h, k, l = int(self.crystal_h_e.get()), int(self.crystal_k_e.get()), int(self.crystal_l_e.get())
                export_data[f'd-spacing ({h}{k}{l}) (Å)'] = d_spacing_extrap
            
            # Add experimental data
            max_len = len(P_extrap)
            exp_len = len(P_data)
            
            for axis in axes:
                export_data[f'Exp_Lattice_{axis} (Å)'] = list(lattice_data[axis]) + [np.nan] * (max_len - exp_len)
                export_data[f'Exp_sigma_{axis} (Å)'] = list(sigma_data[f'sigma_{axis}']) + [np.nan] * (max_len - exp_len)
            
            export_data['Exp_Pressure (GPa)'] = list(P_data) + [np.nan] * (max_len - exp_len)
            export_data['Exp_sigma_P (GPa)'] = list(sigma_data['sigma_P']) + [np.nan] * (max_len - exp_len)
            
            if d_spacing_data is not None:
                export_data[f'Exp_d-spacing ({h}{k}{l}) (Å)'] = list(d_spacing_data) + [np.nan] * (max_len - exp_len)
            
            # Add fitting parameters
            for axis in axes:
                if axis in self.crystal_fit_results:
                    fit_result = self.crystal_fit_results[axis]
                    params = fit_result['params']
                    errors = fit_result['errors']
                    param_names = fit_result['param_names']
                    
                    for j, (param_name, param_val, param_err) in enumerate(zip(param_names, params, errors)):
                        export_data[f'{axis}_fitted_{param_name}'] = [param_val] * max_len
                        export_data[f'{axis}_error_{param_name}'] = [param_err] * max_len
            
            self.last_crystal_df = pd.DataFrame(export_data)
            
            # Show fitting results
            result_text = f"{crystal_system} {eos_type} Fit Results:\n\n"
            for axis in axes:
                if axis in self.crystal_fit_results:
                    fit_result = self.crystal_fit_results[axis]
                    params = fit_result['params']
                    errors = fit_result['errors']
                    param_names = fit_result['param_names']
                    
                    result_text += f"Axis {axis}:\n"
                    for param_name, param_val, param_err in zip(param_names, params, errors):
                        if param_name == 'L0':
                            result_text += f"  {param_name} = {param_val:.4f} ± {param_err:.4f} Å\n"
                        elif param_name == 'M0':
                            result_text += f"  {param_name} = {param_val:.2f} ± {param_err:.2f} GPa\n"
                        else:
                            result_text += f"  {param_name} = {param_val:.3f} ± {param_err:.3f}\n"
                    result_text += "\n"
            
            messagebox.showinfo("Fitting Results", result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fit/plot EOS:\n{str(e)}")

    def export_crystal_data(self):
        if hasattr(self, 'last_crystal_df'):
            fname = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel', '*.xlsx')])
            if fname:
                self.last_crystal_df.to_excel(fname, index=False)
                messagebox.showinfo("Export Successful", f"Crystal EOS data saved to {fname}")
        else:
            messagebox.showwarning("No Data", "Fit EOS first.")

    # [Volume/Density UI methods remain the same as in previous version]
    # ... (include all the volume/density methods from the previous code)

        # ---- Volume/Density UI (계속) ----
    def setup_volume_ui(self):
        frame = self.volume_frame
        padX, padY = 8, 6
        
        # Create main container with scrollbar
        main_container = ttk.Frame(frame)
        main_container.pack(fill='both', expand=True, padx=8, pady=8)
        
        # Canvas and scrollbar for scrolling
        canvas = tk.Canvas(main_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 마우스 휠 스크롤만 활성화
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Use scrollable_frame instead of frame for the rest of the UI
        working_frame = self.scrollable_frame
        
        # Top controls with swap axis option
        top_controls = ttk.LabelFrame(working_frame, text="General Settings")
        top_controls.grid(row=0, column=0, columnspan=6, sticky='ew', padx=padX, pady=padY)
        
        # Add title for general settings
        general_title = tk.Label(top_controls, text="General Settings", font=("Arial", 14, "bold"))
        general_title.grid(row=0, column=0, columnspan=3, pady=8)
        
        ttk.Label(top_controls, text="Number of EOS Curves:", font=("Arial", 12)).grid(row=1, column=0, padx=padX, pady=padY, sticky='w')
        self.num_curves = tk.IntVar(value=1)
        ttk.Spinbox(top_controls, from_=1, to=10, textvariable=self.num_curves, width=6, 
                   command=self.update_volume_curves, font=("Arial", 12)).grid(row=1, column=1, padx=padX, pady=padY)
        
        # Add Swap Axes checkbox for Volume/Density tab
        self.volume_swap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top_controls, text="Swap Axes (X ↔ Y)", variable=self.volume_swap_var).grid(row=1, column=2, padx=25, pady=padY)
        
        self.curves_container = ttk.Frame(working_frame)
        self.curves_container.grid(row=1, column=0, columnspan=6, padx=padX, pady=padY, sticky='w')
        
        self.curve_entries = []
        self.update_volume_curves()
        
        # Buttons at the bottom
        button_frame = ttk.Frame(working_frame)
        button_frame.grid(row=2, column=0, columnspan=6, padx=padX, pady=15)
        ttk.Button(button_frame, text="Plot Curves", command=self.plot_volume_density, width=18).grid(row=0, column=0, padx=8)
        ttk.Button(button_frame, text="Export to Excel", command=self.export_volume_data, width=18).grid(row=0, column=1, padx=8)
        ttk.Button(button_frame, text="Show MC Statistics", command=self.show_mc_statistics, width=18).grid(row=0, column=2, padx=8)

    def import_data_for_curve(self, curve_idx):
        """Import experimental data for a specific curve"""
        fname = filedialog.askopenfilename(
            title=f"Select Excel file for Curve {curve_idx+1}",
            filetypes=[("Excel files", "*.xlsx"), ("Excel files", "*.xls")]
        )
        if not fname:
            return
            
        try:
            df = pd.read_excel(fname)
            
            # Check required columns
            required_cols = {'P', 'V', 'sigma_P', 'sigma_V'}
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                messagebox.showerror(
                    "Column Error", 
                    f"Excel file must contain columns: {', '.join(required_cols)}\n"
                    f"Missing columns: {', '.join(missing_cols)}"
                )
                return
            
            # Check for optional density columns
            optional_cols = {'Z', 'M', 'sigma_Z', 'sigma_M'}
            has_density = optional_cols.issubset(set(df.columns))
            
            # Store data in the curve entry
            self.curve_entries[curve_idx]['data_df'] = df
            self.curve_entries[curve_idx]['data_status_label'].config(
                text=f"Loaded {len(df)} points" + (" (density)" if has_density else ""),
                foreground="green"
            )
            messagebox.showinfo("Import Successful", f"Loaded {len(df)} data points for Curve {curve_idx+1}")
            
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to load data:\n{str(e)}")

    def fit_eos_to_data(self, eos_type, data_df, curve_name=None):
        """Enhanced EOS fitting with robust algorithms and quality metrics"""
        
        # Use robust fitting algorithm
        try:
            # Update status
            if curve_name:
                self.update_fitting_status(curve_name, "Fitting EOS with robust algorithm...", color="orange")
            
            # Use robust fitting
            result = robust_eos_fitting(
                data_df['V'].values,
                data_df['P'].values,
                sigma_P=data_df['sigma_P'].values if 'sigma_P' in data_df.columns else None,
                eos_type=eos_type
            )
            
            # Update status with quality metrics
            if curve_name:
                self.update_fitting_status(
                    curve_name, 
                    f"Fitting successful ({result['successful_attempts']}/{result['fitting_attempts']} attempts)",
                    metrics=result,
                    color="green"
                )
            
            # Return in old format for compatibility
            return result['params'], result['covariance'], result['errors']
            
        except Exception as e:
            # Fallback to original fitting method
            if curve_name:
                self.update_fitting_status(curve_name, f"Robust fitting failed, using fallback: {str(e)}", color="orange")
            
            return self._fit_eos_fallback(eos_type, data_df)

    def _fit_eos_fallback(self, eos_type, data_df):
        """Fallback fitting method (original algorithm)"""
        # Sort data first
        sorted_data = smart_pressure_sort(data_df['P'].values, data_df['V'].values, 
                                        data_df['sigma_P'].values if 'sigma_P' in data_df.columns else None)
        V = sorted_data[1]
        P = sorted_data[0]
        sigma_P = sorted_data[2] if len(sorted_data) > 2 and sorted_data[2] is not None else None
        
        # Choose EOS function
        if eos_type == 'Birch-Murnaghan':
            eos_func = birch_murnaghan_fit_func
        else:  # Vinet
            eos_func = vinet_fit_func
            
        # Initial guess for parameters
        V0_guess = V.max()
        K0_guess = 160.0
        Kp_guess = 4.0
        
        try:
            # Fit with experimental uncertainties
            if sigma_P is not None:
                popt, pcov = curve_fit(
                    eos_func, V, P, 
                    p0=[V0_guess, K0_guess, Kp_guess],
                    sigma=sigma_P, 
                    absolute_sigma=True
                )
            else:
                popt, pcov = curve_fit(
                    eos_func, V, P, 
                    p0=[V0_guess, K0_guess, Kp_guess]
                )
            
            # Apply physical constraints
            popt = apply_physical_constraints(popt, eos_type)
            
            # Calculate parameter uncertainties
            perr = np.sqrt(np.diag(pcov))
            
            return popt, pcov, perr
            
        except Exception as e:
            raise ValueError(f"Fitting failed: {str(e)}")

    def monte_carlo_error_analysis(self, P_range, entry, n_samples=1000):
        """
        Enhanced Monte Carlo error analysis considering all uncertainty sources:
        1. EOS fitting parameter uncertainties (V0, K0, K')
        2. Experimental measurement uncertainties (σP, σV)
        3. Density calculation parameter uncertainties (σZ, σM)
        """
        name = entry['name'].get()
        eos_type = entry['eos'].get()
        
        # Get EOS parameters and their uncertainties
        if entry['use_fitting'].get() and entry['data_df'] is not None:
            # Use fitted parameters
            try:
                popt, pcov, perr = self.fit_eos_to_data(eos_type, entry['data_df'])
                V0, K0, Kp = popt
                
                # Use experimental data for noise simulation
                data_df = entry['data_df']
                has_exp_data = True
            except:
                messagebox.showerror("Error", f"Failed to fit data for {name}")
                return None, None, None
        else:
            # Use manual parameters
            V0 = float(entry['V0'].get())
            K0 = float(entry['K0'].get())
            Kp = float(entry['Kp'].get())
            V0_err = float(entry['V0_err'].get())
            K0_err = float(entry['K0_err'].get())
            Kp_err = float(entry['Kp_err'].get())
            
            # Create covariance matrix from manual uncertainties
            pcov = np.diag([V0_err**2, K0_err**2, Kp_err**2])
            data_df = entry['data_df']
            has_exp_data = False
        
        # Choose EOS function
        if eos_type == 'Birch-Murnaghan':
            eos_func = birch_murnaghan_P
        else:  # Vinet
            eos_func = vinet_P
        
        # Monte Carlo sampling
        volume_samples = []
        density_samples = [] if entry['density'].get() else None
        param_samples = []  # 파라미터 샘플링 저장
        
        successful_samples = 0
        
        for i in range(n_samples):
            try:
                # 1. Sample EOS parameters from multivariate normal distribution
                if entry['use_fitting'].get() and entry['data_df'] is not None:
                    V0_s, K0_s, Kp_s = np.random.multivariate_normal([V0, K0, Kp], pcov)
                else:
                    V0_s, K0_s, Kp_s = np.random.multivariate_normal([V0, K0, Kp], pcov)
                
                # Ensure physical constraints
                V0_s = max(V0_s, V0 * 0.5)  # V0 shouldn't be too small
                K0_s = max(K0_s, 10.0)      # K0 should be positive and reasonable
                Kp_s = max(Kp_s, 1.0)       # K' should be > 1
                
                param_samples.append([V0_s, K0_s, Kp_s])
                
                # 2. Calculate volumes for each pressure with experimental noise
                vols_i = []
                for j, p in enumerate(P_range):
                    try:
                        # Add experimental pressure noise if available
                        if has_exp_data and data_df is not None:
                            # Use pressure-dependent noise if available
                            if len(data_df) > j:
                                p_noise = np.random.normal(0, data_df['sigma_P'].iloc[j])
                            else:
                                p_noise = np.random.normal(0, data_df['sigma_P'].mean())
                            p_noisy = p + p_noise
                        else:
                            p_noisy = p
                        
                        # Calculate volume using safe inverse EOS
                        calc_result = safe_volume_calculation(eos_func, p_noisy, V0_s, K0_s, Kp_s)
                        vol = calc_result['volume']
                        
                        # Add experimental volume noise if available
                        if has_exp_data and data_df is not None:
                            if len(data_df) > j:
                                vol_noise = np.random.normal(0, data_df['sigma_V'].iloc[j])
                            else:
                                vol_noise = np.random.normal(0, data_df['sigma_V'].mean())
                            vol += vol_noise
                        
                        vols_i.append(max(vol, V0_s * 0.1))  # Ensure positive volume
                    except:
                        vols_i.append(np.nan)
                
                volume_samples.append(vols_i)
                
                # 3. Calculate density with parameter uncertainties if requested
                if entry['density'].get():
                    try:
                        # Get density parameters
                        Z = float(entry['Z'].get())
                        M = float(entry['M'].get())
                        
                        # Sample density parameters with uncertainties
                        if (has_exp_data and data_df is not None and 
                            'sigma_Z' in data_df.columns and 'sigma_M' in data_df.columns):
                            Z_err = data_df['sigma_Z'].iloc[0] if not data_df['sigma_Z'].isna().all() else 0.01
                            M_err = data_df['sigma_M'].iloc[0] if not data_df['sigma_M'].isna().all() else 0.01
                        else:
                            # Default uncertainties
                            Z_err = 0.01  # Small uncertainty for atomic number
                            M_err = 0.01  # Small uncertainty for molar mass
                        
                        Z_s = np.random.normal(Z, Z_err)
                        M_s = np.random.normal(M, M_err)
                        
                        # Ensure positive values
                        Z_s = max(Z_s, 1.0)
                        M_s = max(M_s, 1.0)
                        
                        # Calculate density
                        NA = 6.022e23
                        dens_i = [(Z_s * M_s) / (NA * vol * 1e-24) for vol in vols_i]
                        
                        if density_samples is None:
                            density_samples = []
                        density_samples.append(dens_i)
                        
                    except:
                        if density_samples is None:
                            density_samples = []
                        density_samples.append([np.nan] * len(vols_i))
                
                successful_samples += 1
                        
            except Exception as e:
                # Skip this sample if calculation fails
                continue
        
        if len(volume_samples) == 0:
            messagebox.showerror("Error", f"Monte Carlo simulation failed for {name}")
            return None, None, None
        
        # Calculate statistics
        volume_samples = np.array(volume_samples)
        param_samples = np.array(param_samples)
        
        volume_mean = np.nanmean(volume_samples, axis=0)
        volume_std = np.nanstd(volume_samples, axis=0)
        volume_min = np.nanpercentile(volume_samples, 2.5, axis=0)
        volume_max = np.nanpercentile(volume_samples, 97.5, axis=0)
        
        # Parameter statistics
        param_means = np.mean(param_samples, axis=0)
        param_stds = np.std(param_samples, axis=0)
        
        density_results = None
        if density_samples is not None and len(density_samples) > 0:
            density_samples = np.array(density_samples)
            density_mean = np.nanmean(density_samples, axis=0)
            density_std = np.nanstd(density_samples, axis=0)
            density_min = np.nanpercentile(density_samples, 2.5, axis=0)
            density_max = np.nanpercentile(density_samples, 97.5, axis=0)
            density_results = {
                'mean': density_mean,
                'std': density_std,
                'min': density_min,
                'max': density_max
            }
        
        volume_results = {
            'mean': volume_mean,
            'std': volume_std,
            'min': volume_min,
            'max': volume_max
        }
        
        # Store MC results for visualization
        mc_stats = {
            'name': name,
            'eos_type': eos_type,
            'n_requested': n_samples,
            'n_successful': successful_samples,
            'success_rate': successful_samples / n_samples * 100,
            'param_samples': param_samples,
            'param_means': param_means,
            'param_stds': param_stds,
            'volume_samples': volume_samples,
            'density_samples': density_samples,
            'P_range': P_range,
            'has_exp_data': has_exp_data,
            'data_df': data_df if has_exp_data else None
        }
        
        self.mc_results[name] = mc_stats
        
        return volume_results, density_results, successful_samples

    def show_mc_statistics(self):
        """Show Monte Carlo simulation statistics"""
        if not self.mc_results:
            messagebox.showinfo("No Data", "No Monte Carlo results available. Run simulation first.")
            return
        
        # Create figure with subplots
        n_curves = len(self.mc_results)
        fig = plt.figure(figsize=(16, 4 * n_curves))
        gs = gridspec.GridSpec(n_curves, 4, figure=fig)
        
        for i, (name, stats) in enumerate(self.mc_results.items()):
            # Parameter distribution histograms
            ax1 = fig.add_subplot(gs[i, 0])
            param_samples = stats['param_samples']
            ax1.hist(param_samples[:, 0], bins=50, alpha=0.7, color='blue', label='V₀')
            ax1.axvline(stats['param_means'][0], color='red', linestyle='--', linewidth=2)
            ax1.set_xlabel('V₀ (Å³)')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{name}: V₀ Distribution\nμ={stats["param_means"][0]:.2f}±{stats["param_stds"][0]:.2f}')
            ax1.grid(alpha=0.3)
            
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.hist(param_samples[:, 1], bins=50, alpha=0.7, color='green', label='K₀')
            ax2.axvline(stats['param_means'][1], color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('K₀ (GPa)')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'K₀ Distribution\nμ={stats["param_means"][1]:.2f}±{stats["param_stds"][1]:.2f}')
            ax2.grid(alpha=0.3)
            
            ax3 = fig.add_subplot(gs[i, 2])
            ax3.hist(param_samples[:, 2], bins=50, alpha=0.7, color='orange', label="K'")
            ax3.axvline(stats['param_means'][2], color='red', linestyle='--', linewidth=2)
            ax3.set_xlabel("K'")
            ax3.set_ylabel('Frequency')
            ax3.set_title(f"K' Distribution\nμ={stats['param_means'][2]:.3f}±{stats['param_stds'][2]:.3f}")
            ax3.grid(alpha=0.3)
            
            # Confidence interval visualization
            ax4 = fig.add_subplot(gs[i, 3])
            P_range = stats['P_range']
            volume_samples = stats['volume_samples']
            
            # Calculate percentiles
            vol_mean = np.nanmean(volume_samples, axis=0)
            vol_5 = np.nanpercentile(volume_samples, 5, axis=0)
            vol_25 = np.nanpercentile(volume_samples, 25, axis=0)
            vol_75 = np.nanpercentile(volume_samples, 75, axis=0)
            vol_95 = np.nanpercentile(volume_samples, 95, axis=0)
            
            # Plot confidence intervals
            ax4.fill_between(P_range, vol_5, vol_95, alpha=0.2, color='blue', label='90% CI')
            ax4.fill_between(P_range, vol_25, vol_75, alpha=0.4, color='blue', label='50% CI')
            ax4.plot(P_range, vol_mean, 'r-', linewidth=2, label='Mean')
            
            # Plot experimental data if available
            if stats['has_exp_data'] and stats['data_df'] is not None:
                data_df = stats['data_df']
                ax4.errorbar(data_df['P'], data_df['V'], 
                           xerr=data_df['sigma_P'], yerr=data_df['sigma_V'],
                           fmt='ko', markersize=4, capsize=3, label='Exp Data')
            
            ax4.set_xlabel('Pressure (GPa)')
            ax4.set_ylabel('Volume (Å³)')
            ax4.set_title(f'MC Confidence Intervals\nSuccess Rate: {stats["success_rate"]:.1f}%')
            ax4.legend()
            ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Show text statistics
        stats_text = "Monte Carlo Simulation Statistics\n" + "="*50 + "\n\n"
        
        for name, stats in self.mc_results.items():
            stats_text += f"Curve: {name} ({stats['eos_type']})\n"
            stats_text += f"Samples: {stats['n_successful']}/{stats['n_requested']} (Success Rate: {stats['success_rate']:.1f}%)\n"
            stats_text += f"Parameters (mean ± std):\n"
            stats_text += f"  V₀ = {stats['param_means'][0]:.2f} ± {stats['param_stds'][0]:.2f} Å³\n"
            stats_text += f"  K₀ = {stats['param_means'][1]:.2f} ± {stats['param_stds'][1]:.2f} GPa\n"
            stats_text += f"  K' = {stats['param_means'][2]:.3f} ± {stats['param_stds'][2]:.3f}\n"
            
            # Correlation coefficients
            corr_matrix = np.corrcoef(stats['param_samples'], rowvar=False)
            stats_text += f"Parameter Correlations:\n"
            stats_text += f"  V₀-K₀: {corr_matrix[0,1]:.3f}\n"
            stats_text += f"  V₀-K': {corr_matrix[0,2]:.3f}\n"
            stats_text += f"  K₀-K': {corr_matrix[1,2]:.3f}\n"
            stats_text += "\n" + "-"*30 + "\n\n"
        
        # Show in message box (for basic info) and create text window for detailed stats
        messagebox.showinfo("MC Statistics", f"Monte Carlo statistics displayed.\nTotal curves analyzed: {len(self.mc_results)}")
        
        # Create detailed statistics window
        stats_window = tk.Toplevel(self.master)
        stats_window.title("Monte Carlo Detailed Statistics")
        stats_window.geometry("600x500")
        
        text_widget = tk.Text(stats_window, wrap=tk.WORD, font=('Courier', 10))
        scrollbar_text = ttk.Scrollbar(stats_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar_text.set)
        
        text_widget.insert(tk.END, stats_text)
        text_widget.config(state=tk.DISABLED)
        
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar_text.pack(side="right", fill="y")

    def update_volume_curves(self):
        target = self.num_curves.get()
        current = len(self.curve_entries)
        
        for i in range(current, target):
            grp = ttk.LabelFrame(self.curves_container, text=f"EOS Curve {i+1}")
            grp.grid(row=i, column=0, padx=8, pady=5, sticky='ew')
            grp.columnconfigure(1, weight=1)
            grp.columnconfigure(3, weight=1)
            grp.columnconfigure(5, weight=1)
            
            # Row 0: Basic curve settings
            ttk.Label(grp, text="Name:", font=("Arial", 12)).grid(row=0, column=0, padx=5, sticky='w')
            name_var = tk.StringVar(value=f"Curve{i+1}")
            name_e = ttk.Entry(grp, textvariable=name_var, width=15, font=("Arial", 12))
            name_e.grid(row=0, column=1, padx=5, sticky='w')
            
            ttk.Label(grp, text="EOS:", font=("Arial", 12)).grid(row=0, column=2, padx=5, sticky='w')
            eos_var = tk.StringVar(value='Birch-Murnaghan')
            eos_cb = ttk.Combobox(grp, textvariable=eos_var, values=['Birch-Murnaghan','Vinet'], 
                                 state='readonly', width=18, font=("Arial", 12))
            eos_cb.grid(row=0, column=3, padx=5, sticky='w')
            
            # Row 1: Analysis options
            error_var = tk.BooleanVar(value=False)
            error_cb = ttk.Checkbutton(grp, text="Show Error Band", variable=error_var,
                                     command=lambda ev=error_var, idx=i: self.toggle_error_fields(ev, idx))
            error_cb.grid(row=1, column=0, columnspan=2, padx=5, sticky='w')
            
            monte_carlo_var = tk.BooleanVar(value=False)
            monte_carlo_cb = ttk.Checkbutton(grp, text="Monte Carlo Simulation", variable=monte_carlo_var)
            monte_carlo_cb.grid(row=1, column=2, columnspan=2, padx=5, sticky='w')
            
            # Row 2: EOS Equation Label
            eq_lbl = ttk.Label(grp, text=EOS_EQUATIONS['Birch-Murnaghan'], foreground="blue", 
                              font=("Arial", 11, "bold"))
            eq_lbl.grid(row=2, column=0, columnspan=6, sticky='w', padx=5)
            
            # When EOS type changes, update label
            def make_eos_cb_callback(lbl, var):
                return lambda e=None: lbl.config(text=EOS_EQUATIONS[var.get()])
            eos_cb.bind("<<ComboboxSelected>>", make_eos_cb_callback(eq_lbl, eos_var))

            # Row 3: Enhanced Data import section
            data_frame = ttk.LabelFrame(grp, text="Enhanced Data Import System")
            data_frame.grid(row=3, column=0, columnspan=6, sticky='ew', padx=5, pady=3)
            data_frame.columnconfigure(1, weight=1)
            data_frame.columnconfigure(3, weight=1)
            
            # EOS Data Import
            ttk.Button(data_frame, text="Import EOS Data", 
                      command=lambda idx=i: self.import_eos_data_for_curve(idx)).grid(row=0, column=0, padx=3, pady=3)
            
            data_status_label = ttk.Label(data_frame, text="No EOS data", foreground="gray", font=("Arial", 10))
            data_status_label.grid(row=0, column=1, padx=5, pady=3, sticky='w')
            
            # Error Data Import
            ttk.Button(data_frame, text="Import Error Data", 
                      command=lambda idx=i: self.import_error_data_for_curve(idx)).grid(row=0, column=2, padx=3, pady=3)
            
            error_status_label = ttk.Label(data_frame, text="No error data", foreground="gray", font=("Arial", 10))
            error_status_label.grid(row=0, column=3, padx=5, pady=3, sticky='w')
            
            # Fitting options
            use_fitting_var = tk.BooleanVar(value=False)
            use_fitting_cb = ttk.Checkbutton(data_frame, text="Use for fitting", variable=use_fitting_var)
            use_fitting_cb.grid(row=1, column=0, padx=5, pady=3)
            
            # Status display for real-time metrics
            status_label = ttk.Label(data_frame, text="Ready for data import", foreground="blue", 
                                   font=("Arial", 10), wraplength=400)
            status_label.grid(row=1, column=1, columnspan=3, padx=5, pady=3, sticky='w')
            
            # Row 4: EOS Parameters
            ttk.Label(grp, text="V₀ (Å³):", font=("Arial", 12)).grid(row=4, column=0, padx=5, sticky='w')
            v0 = ttk.Entry(grp, width=12, font=("Arial", 12)); v0.insert(0,'100'); v0.grid(row=4, column=1, padx=5, sticky='w')
            ttk.Label(grp, text="K₀ (GPa):", font=("Arial", 12)).grid(row=4, column=2, padx=5, sticky='w')
            k0 = ttk.Entry(grp, width=12, font=("Arial", 12)); k0.insert(0,'160'); k0.grid(row=4, column=3, padx=5, sticky='w')
            ttk.Label(grp, text="K′:", font=("Arial", 12)).grid(row=4, column=4, padx=5, sticky='w')
            kp = ttk.Entry(grp, width=12, font=("Arial", 12)); kp.insert(0,'4.0'); kp.grid(row=4, column=5, padx=5, sticky='w')
            
            # Row 5: Error analysis parameters (initially hidden)
            error_frame = ttk.Frame(grp)
            error_frame.grid(row=5, column=0, columnspan=6, sticky='ew', padx=5)
            error_frame.columnconfigure(1, weight=1)
            error_frame.columnconfigure(3, weight=1)
            error_frame.columnconfigure(5, weight=1)
            
            # Parameter uncertainties
            ttk.Label(error_frame, text="σV₀:", font=("Arial", 11)).grid(row=0, column=0, padx=5, sticky='w')
            v0_err = ttk.Entry(error_frame, width=10, font=("Arial", 11)); v0_err.insert(0,'1.0'); v0_err.grid(row=0, column=1, padx=5, sticky='w')
            ttk.Label(error_frame, text="σK₀:", font=("Arial", 11)).grid(row=0, column=2, padx=5, sticky='w')
            k0_err = ttk.Entry(error_frame, width=10, font=("Arial", 11)); k0_err.insert(0,'5.0'); k0_err.grid(row=0, column=3, padx=5, sticky='w')
            ttk.Label(error_frame, text="σK′:", font=("Arial", 11)).grid(row=0, column=4, padx=5, sticky='w')
            kp_err = ttk.Entry(error_frame, width=10, font=("Arial", 11)); kp_err.insert(0,'0.1'); kp_err.grid(row=0, column=5, padx=5, sticky='w')
            
            # Monte Carlo settings
            ttk.Label(error_frame, text="MC Samples:", font=("Arial", 11)).grid(row=1, column=0, padx=5, sticky='w')
            mc_samples = ttk.Entry(error_frame, width=10, font=("Arial", 11)); mc_samples.insert(0,'1000'); mc_samples.grid(row=1, column=1, padx=5, sticky='w')
            ttk.Label(error_frame, text="Confidence (%):", font=("Arial", 11)).grid(row=1, column=2, padx=5, sticky='w')
            confidence = ttk.Entry(error_frame, width=10, font=("Arial", 11)); confidence.insert(0,'95'); confidence.grid(row=1, column=3, padx=5, sticky='w')
            
            error_frame.grid_remove()
            
            # Row 6: Pressure range and points
            ttk.Label(grp, text="P min (GPa):", font=("Arial", 12)).grid(row=6, column=0, padx=5, sticky='w')
            pmin = ttk.Entry(grp, width=12, font=("Arial", 12)); pmin.insert(0,'0'); pmin.grid(row=6, column=1, padx=5, sticky='w')
            ttk.Label(grp, text="P max (GPa):", font=("Arial", 12)).grid(row=6, column=2, padx=5, sticky='w')
            pmax = ttk.Entry(grp, width=12, font=("Arial", 12)); pmax.insert(0,'20'); pmax.grid(row=6, column=3, padx=5, sticky='w')
            ttk.Label(grp, text="Points:", font=("Arial", 12)).grid(row=6, column=4, padx=5, sticky='w')
            npts = ttk.Entry(grp, width=12, font=("Arial", 12)); npts.insert(0,'100'); npts.grid(row=6, column=5, padx=5, sticky='w')
            
            # Row 7: Density settings
            density_var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(grp, text="Plot Density", variable=density_var,
                                 command=lambda dv=density_var, idx=i: self.toggle_density_fields(dv, idx))
            cb.grid(row=7, column=0, columnspan=2, padx=5, pady=3, sticky='w')
            
            # Row 8: Density parameters (initially hidden)
            z_label = ttk.Label(grp, text="Z:", font=("Arial", 12))
            z_entry = ttk.Entry(grp, width=12, font=("Arial", 12))
            m_label = ttk.Label(grp, text="M (g/mol):", font=("Arial", 12))
            m_entry = ttk.Entry(grp, width=12, font=("Arial", 12))
            z_label.grid(row=8, column=0, padx=5, sticky='w'); z_entry.grid(row=8, column=1, padx=5, sticky='w')
            m_label.grid(row=8, column=2, padx=5, sticky='w'); m_entry.grid(row=8, column=3, padx=5, sticky='w')
            z_label.grid_remove(); z_entry.grid_remove(); m_label.grid_remove(); m_entry.grid_remove()
            
            self.curve_entries.append({
                'frame': grp,
                'name': name_var, 'eos': eos_var,
                'V0': v0, 'K0': k0, 'Kp': kp,
                'Pmin': pmin, 'Pmax': pmax, 'Npts': npts,
                'density': density_var,
                'Z_label': z_label, 'Z': z_entry,
                'M_label': m_label, 'M': m_entry,
                'eos_lbl': eq_lbl,
                'error_band': error_var,
                'monte_carlo': monte_carlo_var,
                'error_frame': error_frame,
                'V0_err': v0_err, 'K0_err': k0_err, 'Kp_err': kp_err,
                'mc_samples': mc_samples, 'confidence': confidence,
                'data_df': None,
                'data_status_label': data_status_label,
                'error_status_label': error_status_label,
                'status_label': status_label,
                'use_fitting': use_fitting_var
            })
            
        for i in range(current-1, target-1, -1):
            entry = self.curve_entries.pop()
            entry['frame'].destroy()

    def toggle_error_fields(self, ev, idx):
        entry = self.curve_entries[idx]
        if ev.get():
            entry['error_frame'].grid()
        else:
            entry['error_frame'].grid_remove()

    def toggle_density_fields(self, dv, idx):
        entry = self.curve_entries[idx]
        for key in ['Z_label','Z','M_label','M']:
            widget = entry[key]
            if dv.get(): widget.grid()
            else: widget.grid_remove()

    def calculate_error_band(self, P, V0, K0, Kp, V0_err, K0_err, Kp_err, eos_type):
        """Enhanced error band calculation using safe volume calculation"""
        # Generate multiple parameter sets using normal distribution
        n_samples = 100
        V0_samples = np.random.normal(V0, V0_err, n_samples)
        K0_samples = np.random.normal(K0, K0_err, n_samples)
        Kp_samples = np.random.normal(Kp, Kp_err, n_samples)
        
        # Choose EOS function
        if eos_type == 'Birch-Murnaghan':
            eos_func = birch_murnaghan_P
        else:
            eos_func = vinet_P
        
        # Calculate volumes for each parameter set
        volume_samples = []
        for i in range(n_samples):
            vols_i = []
            for p in P:
                # Apply physical constraints to sampled parameters
                constrained_params = apply_physical_constraints([V0_samples[i], K0_samples[i], Kp_samples[i]], eos_type)
                V0_c, K0_c, Kp_c = constrained_params
                
                # Use safe volume calculation
                calc_result = safe_volume_calculation(eos_func, p, V0_c, K0_c, Kp_c)
                vols_i.append(calc_result['volume'])
            
            volume_samples.append(vols_i)
        
        # Calculate percentiles for error band
        volume_samples = np.array(volume_samples)
        volume_min = np.nanpercentile(volume_samples, 2.5, axis=0)  # 2.5th percentile
        volume_max = np.nanpercentile(volume_samples, 97.5, axis=0)  # 97.5th percentile
        
        return volume_min, volume_max

    def plot_volume_density(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        data = {}
        
        # Check if axes should be swapped
        swap_axes = self.volume_swap_var.get()
        
        for entry in self.curve_entries:
            name = entry['name'].get()
            Pmin = float(entry['Pmin'].get())
            Pmax = float(entry['Pmax'].get())
            N = int(entry['Npts'].get())
            P = np.linspace(Pmin, Pmax, N)
            data[f'{name} Pressure (GPa)'] = P
            
            plot_density = entry['density'].get()
            
            # Check if we should use imported data for fitting
            if entry['use_fitting'].get() and entry['data_df'] is not None:
                try:
                    # Plot experimental data first
                    data_df = entry['data_df']
                    V_exp = data_df['V'].values
                    P_exp = data_df['P'].values
                    sigma_V_exp = data_df['sigma_V'].values
                    sigma_P_exp = data_df['sigma_P'].values
                    
                    # Apply axis swapping for experimental data
                    if swap_axes:
                        ax.errorbar(V_exp, P_exp, xerr=sigma_V_exp, yerr=sigma_P_exp, 
                                   fmt='o', markersize=4, capsize=3, label=f'{name} Exp Data')
                    else:
                        ax.errorbar(P_exp, V_exp, xerr=sigma_P_exp, yerr=sigma_V_exp, 
                                   fmt='o', markersize=4, capsize=3, label=f'{name} Exp Data')
                    
                    # Store experimental data
                    data[f'{name}_Exp_Volume (Å³)'] = V_exp
                    data[f'{name}_Exp_Pressure (GPa)'] = P_exp
                    data[f'{name}_Exp_sigma_V (Å³)'] = sigma_V_exp
                    data[f'{name}_Exp_sigma_P (GPa)'] = sigma_P_exp
                    
                    # Fit EOS to experimental data
                    popt, pcov, perr = self.fit_eos_to_data(entry['eos'].get(), data_df)
                    V0_fit, K0_fit, Kp_fit = popt
                    
                    # Update entry fields with fitted values
                    entry['V0'].delete(0, tk.END); entry['V0'].insert(0, f"{V0_fit:.2f}")
                    entry['K0'].delete(0, tk.END); entry['K0'].insert(0, f"{K0_fit:.2f}")
                    entry['Kp'].delete(0, tk.END); entry['Kp'].insert(0, f"{Kp_fit:.3f}")
                    
                    # Update uncertainty fields with fitted uncertainties
                    if entry['error_band'].get():
                        entry['V0_err'].delete(0, tk.END); entry['V0_err'].insert(0, f"{perr[0]:.2f}")
                        entry['K0_err'].delete(0, tk.END); entry['K0_err'].insert(0, f"{perr[1]:.2f}")
                        entry['Kp_err'].delete(0, tk.END); entry['Kp_err'].insert(0, f"{perr[2]:.3f}")
                    
                    # Calculate fitted curve using safe volume calculation
                    vols = []
                    failed_calcs = 0
                    for p in P:
                        calc_result = safe_volume_calculation(
                            birch_murnaghan_P if entry['eos'].get() == 'Birch-Murnaghan' else vinet_P,
                            p, V0_fit, K0_fit, Kp_fit
                        )
                        vols.append(calc_result['volume'])
                        if not calc_result['success']:
                            failed_calcs += 1
                    
                    vols = np.array(vols)
                    
                    if failed_calcs > 0:
                        self.update_fitting_status(
                            name, 
                            f"Fitting completed with {failed_calcs}/{len(P)} volume calculation warnings",
                            color="orange"
                        )
                    
                    # Store fitting results
                    data[f'{name}_fitted V0 (Å³)'] = [V0_fit] * len(P)
                    data[f'{name}_fitted K0 (GPa)'] = [K0_fit] * len(P)
                    data[f'{name}_fitted Kp'] = [Kp_fit] * len(P)
                    data[f'{name}_fitted σV0 (Å³)'] = [perr[0]] * len(P)
                    data[f'{name}_fitted σK0 (GPa)'] = [perr[1]] * len(P)
                    data[f'{name}_fitted σKp'] = [perr[2]] * len(P)
                    
                except Exception as e:
                    messagebox.showerror("Fitting Error", f"Failed to fit {name}: {str(e)}")
                    # Fall back to manual parameters with safe calculation
                    V0,K0,Kp=float(entry['V0'].get()),float(entry['K0'].get()),float(entry['Kp'].get())
                    vols = []
                    for p in P:
                        calc_result = safe_volume_calculation(
                            birch_murnaghan_P if entry['eos'].get() == 'Birch-Murnaghan' else vinet_P,
                            p, V0, K0, Kp
                        )
                        vols.append(calc_result['volume'])
                    vols = np.array(vols)
            else:
                # Use manual parameters with safe calculation
                V0,K0,Kp=float(entry['V0'].get()),float(entry['K0'].get()),float(entry['Kp'].get())
                vols = []
                for p in P:
                    calc_result = safe_volume_calculation(
                        birch_murnaghan_P if entry['eos'].get() == 'Birch-Murnaghan' else vinet_P,
                        p, V0, K0, Kp
                    )
                    vols.append(calc_result['volume'])
                vols = np.array(vols)
            
            data[f'{name} Volume (Å³)'] = vols
            
            # Calculate density if requested
            dens = None
            if plot_density:
                try:
                    Z = float(entry['Z'].get())
                    M = float(entry['M'].get())
                    NA = 6.022e23
                    dens = (Z * M) / (NA * vols * 1e-24)
                    data[f'{name} Density (g/cm³)'] = dens
                except ValueError:
                    messagebox.showerror("Input Error", f"Invalid Z/M for {name}")
                    return
            
            # Calculate error bands - ONLY for the selected plot type
            if entry['error_band'].get():
                try:
                    if entry['monte_carlo'].get():
                        # Use Monte Carlo simulation
                        n_samples = int(entry['mc_samples'].get())
                        volume_results, density_results, actual_samples = self.monte_carlo_error_analysis(P, entry, n_samples)
                        
                        if volume_results is not None:
                            vol_min = volume_results['min']
                            vol_max = volume_results['max']
                            data[f'{name}_MC_error_min Volume (Å³)'] = vol_min
                            data[f'{name}_MC_error_max Volume (Å³)'] = vol_max
                            data[f'{name}_MC_std Volume (Å³)'] = volume_results['std']
                            data[f'{name}_MC_samples'] = [actual_samples] * len(P)
                            
                            if plot_density and density_results is not None:
                                # For density: show density error bands
                                dens_min = density_results['min']
                                dens_max = density_results['max']
                                data[f'{name}_MC_error_min Density (g/cm³)'] = dens_min
                                data[f'{name}_MC_error_max Density (g/cm³)'] = dens_max
                                data[f'{name}_MC_std Density (g/cm³)'] = density_results['std']
                                
                                # Plot density error bands
                                if swap_axes:
                                    ax.fill_between(dens_min, P, dens_max, alpha=0.3, label=f'{name} MC Error Band')
                                else:
                                    ax.fill_between(P, dens_min, dens_max, alpha=0.3, label=f'{name} MC Error Band')
                            elif not plot_density:
                                # For volume: show volume error bands
                                if swap_axes:
                                    ax.fill_between(vol_min, P, vol_max, alpha=0.3, label=f'{name} MC Error Band')
                                else:
                                    ax.fill_between(P, vol_min, vol_max, alpha=0.3, label=f'{name} MC Error Band')
                    else:
                        # Use simple error propagation
                        V0_err = float(entry['V0_err'].get())
                        K0_err = float(entry['K0_err'].get())
                        Kp_err = float(entry['Kp_err'].get())
                        
                        vol_min, vol_max = self.calculate_error_band(P, V0, K0, Kp, V0_err, K0_err, Kp_err, entry['eos'].get())
                        data[f'{name}_error_min Volume (Å³)'] = vol_min
                        data[f'{name}_error_max Volume (Å³)'] = vol_max
                        
                        if plot_density and dens is not None:
                            # Calculate density error bands from volume error bands
                            Z = float(entry['Z'].get())
                            M = float(entry['M'].get())
                            NA = 6.022e23
                            dens_min = (Z * M) / (NA * vol_max * 1e-24)  # vol_max gives dens_min
                            dens_max = (Z * M) / (NA * vol_min * 1e-24)  # vol_min gives dens_max
                            data[f'{name}_error_min Density (g/cm³)'] = dens_min
                            data[f'{name}_error_max Density (g/cm³)'] = dens_max
                            
                            # Plot density error bands
                            if swap_axes:
                                ax.fill_between(dens_min, P, dens_max, alpha=0.3, label=f'{name} Error Band')
                            else:
                                ax.fill_between(P, dens_min, dens_max, alpha=0.3, label=f'{name} Error Band')
                        elif not plot_density:
                            # Plot volume error bands
                            if swap_axes:
                                ax.fill_between(vol_min, P, vol_max, alpha=0.3, label=f'{name} Error Band')
                            else:
                                ax.fill_between(P, vol_min, vol_max, alpha=0.3, label=f'{name} Error Band')
                        
                except ValueError as e:
                    messagebox.showerror("Error Analysis Failed", f"Error for {name}: {str(e)}")
                    continue
            
            # Plot main curve with axis swapping - ONLY plot the selected type
            if plot_density and dens is not None:
                # Plot density curve
                if swap_axes:
                    ax.plot(dens, P, '--', label=f'{name} Density', linewidth=2)
                else:
                    ax.plot(P, dens, '--', label=f'{name} Density', linewidth=2)
            else:
                # Plot volume curve
                if swap_axes:
                    ax.plot(vols, P, label=f'{name} Volume', linewidth=2)
                else:
                    ax.plot(P, vols, label=f'{name} Volume', linewidth=2)

        # Set axis labels with swapping consideration
        any_density = any(e['density'].get() for e in self.curve_entries)
        if swap_axes:
            ylabel = 'Density (g/cm³)' if any_density else 'Volume (Å³)'
            ax.set_xlabel(ylabel, fontsize=12)
            ax.set_ylabel('Pressure (GPa)', fontsize=12)
        else:
            ax.set_xlabel('Pressure (GPa)', fontsize=12)
            ylabel = 'Density (g/cm³)' if any_density else 'Volume (Å³)'
            ax.set_ylabel(ylabel, fontsize=12)
            
        ax.set_title('Equation of State', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        self.last_volume_df = pd.DataFrame(data)

    def export_volume_data(self):
        if not hasattr(self, 'last_volume_df'):
            messagebox.showwarning("No Data", "Plot first.")
            return
        fname = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel', '*.xlsx')])
        if fname:
            self.last_volume_df.to_excel(fname, index=False)
            messagebox.showinfo("Export Successful", f"Data saved to {fname}")


if __name__ == '__main__':
    root = tk.Tk()
    app = EOSPlotApp(root)
    root.mainloop()