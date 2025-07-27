import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.utilities.lambdify import lambdify
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.cm as cm
from utils import get_figure_size, apply_image_container

def parse_complex_function(func_str):
    """Parse a complex function string into a sympy expression"""
    # Add support for complex operations
    z = sp.Symbol('z', complex=True)
    try:
        transformations = (standard_transformations + 
                          (implicit_multiplication_application,))
        expr = parse_expr(func_str, local_dict={'z': z}, transformations=transformations)
        return expr, z
    except Exception as e:
        st.error(f"Error parsing function: {str(e)}")
        st.stop()

def evaluate_function(func, z_sym, grid):
    """Evaluate a complex function on a grid"""
    try:
        # Create a numpy-compatible function
        func_numpy = lambdify(z_sym, func, modules=["numpy"])
        return func_numpy(grid)
    except Exception as e:
        st.error(f"Error evaluating function: {str(e)}")
        st.stop()

def compute_residues(func, z_sym):
    """Attempt to compute residues of the function"""
    try:
        # Try to find poles symbolically
        poles = []
        residues = []
        
        # Check common denominators for poles
        denom = sp.denom(func)
        factors = sp.factor_list(denom)[1]
        
        for factor, multiplicity in factors:
            # Try to solve for the roots of the factor
            if factor.has(z_sym):
                roots = sp.solve(factor, z_sym)
                for root in roots:
                    # Calculate residue at the pole
                    residue = sp.limit((z_sym - root) ** multiplicity * func, z_sym, root)
                    poles.append(root)
                    residues.append(residue)
        
        return poles, residues
    except Exception as e:
        st.warning(f"Could not compute residues automatically: {str(e)}")
        return [], []

def create_contour_path(center, radius, num_points=100):
    """Create a circular contour path"""
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center.real + radius * np.cos(theta)
    y = center.imag + radius * np.sin(theta)
    return x + 1j * y

def compute_contour_integral(func_numpy, contour_points):
    """Compute the contour integral numerically"""
    # Calculate the function values along the contour
    f_values = func_numpy(contour_points)
    
    # Calculate dz for each segment
    dz = np.diff(contour_points)
    
    # Midpoint rule: evaluate f at midpoints
    midpoints = (contour_points[:-1] + contour_points[1:]) / 2
    f_mid = func_numpy(midpoints)
    
    # Compute the integral as sum of f(z) * dz
    integral = np.sum(f_mid * dz)
    
    return integral

def visualize_integral_contour(func_str, domain_min, domain_max, resolution):
    """Visualize contour integrals in the complex plane"""
    # Set the current visualization type
    st.session_state["current_visualization"] = "integral_contour"
    
    st.markdown("### Contour Integration Visualization")
    
    # Parse the function
    func, z_sym = parse_complex_function(func_str)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Show the mathematical formula
    status_text.text("Parsing mathematical expression...")
    progress_bar.progress(10)
    
    try:
        latex_formula = sp.latex(func)
        st.markdown(f"**Function**: $f(z) = {latex_formula}$")
    except:
        st.markdown(f"**Function**: f(z) = {func_str}")
    
    # Find poles and residues
    status_text.text("Finding poles and residues...")
    progress_bar.progress(20)
    
    poles, residues = compute_residues(func, z_sym)
    
    # Create the grid for visualization
    status_text.text("Creating visualization grid...")
    progress_bar.progress(30)
    
    x = np.linspace(domain_min, domain_max, resolution)
    y = np.linspace(domain_min, domain_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Evaluate function on grid
    status_text.text("Evaluating function...")
    progress_bar.progress(40)
    
    func_numpy = lambdify(z_sym, func, modules=["numpy"])
    
    # Try to evaluate the function on the grid
    try:
        W = func_numpy(Z)
        # Use the absolute value for visualization
        mag = np.abs(W)
        
        # Handle infinities and NaNs
        mag = np.where(np.isnan(mag) | np.isinf(mag), 0, mag)
        
        # Log scaling for better visualization
        log_mag = np.log1p(mag)
    except Exception as e:
        st.error(f"Error evaluating function on grid: {str(e)}")
        # Create a dummy grid
        log_mag = np.zeros_like(X)
    
    # Allow user to select contour using columns for compact layout
    with st.container():
        st.markdown("### Choose Contour")
        
        # Let user choose contour center and radius
        col1, col2, col3 = st.columns(3)
        with col1:
            center_x = st.slider("Center (Real)", domain_min, domain_max, 0.0)
        with col2:
            center_y = st.slider("Center (Imag)", domain_min, domain_max, 0.0)
        with col3:
            radius = st.slider("Radius", 0.1, (domain_max - domain_min)/2, min(2.0, (domain_max - domain_min)/4))
    
    center = complex(center_x, center_y)
    
    # Create the contour
    status_text.text("Creating contour...")
    progress_bar.progress(60)
    
    contour_points = create_contour_path(center, radius)
    
    # Compute the contour integral
    status_text.text("Computing contour integral...")
    progress_bar.progress(70)
    
    try:
        integral_value = compute_contour_integral(func_numpy, contour_points)
        integral_str = f"{integral_value.real:.4f} + {integral_value.imag:.4f}j"
    except Exception as e:
        st.error(f"Error computing integral: {str(e)}")
        integral_value = None
        integral_str = "Error"
    
    # Create visualization
    status_text.text("Creating visualization...")
    progress_bar.progress(90)
    
    # Get figure size from user preferences
    fig_size = get_figure_size()
    
    # Create the figure with a dynamic size
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot the function magnitude
    contour = ax.contourf(X, Y, log_mag, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, ax=ax, label='log(|f(z)|)')
    
    # Plot poles if any were found
    if poles:
        pole_x = [complex(pole).real for pole in poles]
        pole_y = [complex(pole).imag for pole in poles]
        ax.scatter(pole_x, pole_y, color='red', s=100, marker='x', label='Poles')
    
    # Plot the contour
    contour_x = np.real(contour_points)
    contour_y = np.imag(contour_points)
    ax.plot(contour_x, contour_y, 'r-', linewidth=2, label='Integration Contour')
    
    # Add arrows to show direction
    num_arrows = 8
    arrow_indices = np.linspace(0, len(contour_x)-1, num_arrows, dtype=int)
    for i in arrow_indices:
        idx = i % len(contour_x)
        next_idx = (idx + 5) % len(contour_x)
        dx = contour_x[next_idx] - contour_x[idx]
        dy = contour_y[next_idx] - contour_y[idx]
        ax.arrow(contour_x[idx], contour_y[idx], dx, dy, 
                head_width=0.1, head_length=0.15, fc='r', ec='r')
    
    # Add coordinate axes
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add labels and title
    ax.set_xlabel('Re(z)')
    ax.set_ylabel('Im(z)')
    if integral_value is not None:
        ax.set_title(f"Contour Integral: $\\oint_C f(z) dz = {integral_str}$")
    else:
        ax.set_title("Contour Integral Visualization")
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Complete progress
    progress_bar.progress(100)
    status_text.text("Visualization complete!")
    
    # Display the plot with the resizable container
    apply_image_container(fig, caption=f"Contour integral visualization for f(z) = {func_str}")
    
    # Display results and explanation in expandable sections
    with st.expander("Contour Integral Results"):
        st.markdown("### Contour Integral Results")
        
        # Show poles and residues
        if poles:
            st.markdown("#### Poles and Residues")
            for i, (pole, residue) in enumerate(zip(poles, residues)):
                try:
                    pole_str = f"{float(pole.real):.4f} + {float(pole.imag):.4f}j"
                    residue_str = f"{float(residue.real):.4f} + {float(residue.imag):.4f}j"
                    st.markdown(f"**Pole {i+1}**: $z = {pole_str}$ with residue $= {residue_str}$")
                except:
                    st.markdown(f"**Pole {i+1}**: $z = {pole}$ with residue $= {residue}$")
        else:
            st.markdown("No poles were automatically detected. The function may be entire (no poles) or the poles may be complex to find symbolically.")
        
        # Show the computed integral
        st.markdown(f"**Integral Value**: $\\oint_C f(z) dz = {integral_str}$")
    
    # Explain the Residue Theorem
    with st.expander("Residue Theorem Explanation"):
        st.markdown("""
        ### The Residue Theorem
        
        The residue theorem states that for a complex function $f(z)$ that is analytic except at isolated singularities inside a closed contour $C$:
        
        $$\\oint_C f(z) dz = 2\\pi i \\sum_{k} \\text{Res}(f, a_k)$$
        
        where the sum is over all poles $a_k$ of $f(z)$ inside the contour, and $\\text{Res}(f, a_k)$ is the residue of $f$ at $a_k$.
        
        ### Interpreting the Results
        
        - If the contour encloses no poles, the integral should be close to zero
        - If poles are enclosed, the integral equals $2\\pi i$ times the sum of the residues
        - Numerical errors can occur near poles or with complex functions
        
        Try moving the contour to include or exclude different poles to see how the integral value changes!
        """) 