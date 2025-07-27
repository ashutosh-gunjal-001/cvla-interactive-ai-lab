import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.utilities.lambdify import lambdify
from utils import get_figure_size, apply_image_container

def visualize_harmonic_flow(func_str, domain_min, domain_max, resolution):
    """
    Visualize harmonic flow of a complex function.
    
    Parameters:
    -----------
    func_str : str
        The complex function to visualize as a string (e.g., "log(z)")
    domain_min : float
        Minimum value for both real and imaginary axes
    domain_max : float
        Maximum value for both real and imaginary axes
    resolution : int
        Grid resolution for visualization
    """
    try:
        # Set the current visualization type (wide aspect ratio)
        st.session_state["current_visualization"] = "harmonic_flow_wide"
        
        # Parse the function string
        z = sp.Symbol('z', complex=True)
        transformations = (standard_transformations + 
                          (implicit_multiplication_application,))
        expr = parse_expr(func_str, local_dict={'z': z}, transformations=transformations)
        
        # Convert to a numpy function
        func = lambdify(z, expr, modules=['numpy'])
        
        # Create grid for plotting
        resolution = min(resolution, 40)  # Limit for flow visualization clarity
        real = np.linspace(domain_min, domain_max, resolution)
        imag = np.linspace(domain_min, domain_max, resolution)
        real_grid, imag_grid = np.meshgrid(real, imag)
        z_grid = real_grid + 1j * imag_grid
        
        # Evaluate the function on the grid
        w = func(z_grid)
        
        # Separate real and imaginary parts for harmonic components
        u = np.real(w)
        v = np.imag(w)
        
        # Create gradients for visualization of harmonic fields
        # Compute approximate partial derivatives for field visualization
        eps = 1e-6
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)
        
        # Central difference approximation for derivatives
        for i in range(1, resolution-1):
            for j in range(1, resolution-1):
                du_dx[j, i] = (u[j, i+1] - u[j, i-1]) / (2 * (real[i+1] - real[i]))
                du_dy[j, i] = (u[j+1, i] - u[j-1, i]) / (2 * (imag[j+1] - imag[j]))
                dv_dx[j, i] = (v[j, i+1] - v[j, i-1]) / (2 * (real[i+1] - real[i]))
                dv_dy[j, i] = (v[j+1, i] - v[j-1, i]) / (2 * (imag[j+1] - imag[j]))
        
        # Get figure size from user preferences
        fig_size = get_figure_size()
        
        # Create two subplots - harmonic function and its flow
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        
        # Plot 1: Real part contour + Imaginary part flow
        contour_real = ax1.contourf(real_grid, imag_grid, u, 15, cmap='viridis', alpha=0.7)
        colorbar1 = plt.colorbar(contour_real, ax=ax1)
        colorbar1.set_label('Real Part Value')
        
        # Add streamlines for the real part's gradient
        ax1.streamplot(real_grid, imag_grid, du_dx, du_dy, density=1, color='white', 
                      arrowsize=1, linewidth=1)
        
        ax1.set_title('Real Part with Gradient Flow')
        ax1.set_xlabel('Re(z)')
        ax1.set_ylabel('Im(z)')
        
        # Plot 2: Imaginary part contour + flow
        contour_imag = ax2.contourf(real_grid, imag_grid, v, 15, cmap='plasma', alpha=0.7)
        colorbar2 = plt.colorbar(contour_imag, ax=ax2)
        colorbar2.set_label('Imaginary Part Value')
        
        # Add streamlines for the imaginary part's gradient
        ax2.streamplot(real_grid, imag_grid, dv_dx, dv_dy, density=1, color='white', 
                      arrowsize=1, linewidth=1)
        
        ax2.set_title('Imaginary Part with Gradient Flow')
        ax2.set_xlabel('Re(z)')
        ax2.set_ylabel('Im(z)')
        
        plt.tight_layout()
        
        # Display function information
        st.markdown(f"### Harmonic Flow for f(z) = {func_str}")
        st.markdown(f"**Domain**: [{domain_min}, {domain_max}] × [{domain_min}, {domain_max}]")
        
        # Use the image container helper to display the plot
        apply_image_container(fig, caption=f"Harmonic flow visualization for f(z) = {func_str}")
        
        # Display harmonic properties
        # Check if the Cauchy-Riemann equations approximately hold
        cr_check = np.abs(du_dx - dv_dy) + np.abs(du_dy + dv_dx)
        cr_satisfied = np.mean(cr_check[1:-1, 1:-1]) < 0.1
        
        with st.expander("Harmonic Analysis"):
            st.markdown("### Harmonic Analysis")
            if cr_satisfied:
                st.success("The function appears to be analytic in this domain (Cauchy-Riemann equations satisfied).")
                st.markdown("""
                The real and imaginary parts form a **harmonic conjugate pair**. The following properties hold:
                - The level curves of the real part and imaginary part intersect at right angles
                - The function preserves angles locally (conformal mapping)
                - The function has complex derivatives at every point in the domain
                """)
            else:
                st.warning("The function may not be analytic in parts of this domain.")
            
        # Add educational content
        with st.expander("Learn about Harmonic Functions"):
            st.markdown("""
            ### Harmonic Functions and Complex Analysis
            
            A function u(x,y) is **harmonic** if it satisfies Laplace's equation: ∇²u = 0
            
            For a complex analytic function f(z) = u(x,y) + iv(x,y):
            - Both u and v are harmonic functions
            - u and v satisfy the Cauchy-Riemann equations: ∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x
            - The level curves of u and v form orthogonal families of curves
            
            The streamlines shown in the visualization represent the gradient flow of each component,
            showing how the function "flows" across the complex plane.
            """)
        
    except Exception as e:
        st.error(f"Error visualizing harmonic flow: {str(e)}")
        st.info(f"Function input: f(z) = {func_str}")
        
        # Suggest fixes for common input errors
        if "sympify" in str(e):
            st.markdown("""
            ### Common syntax issues:
            - Use ** for powers: `z**2` instead of `z^2`
            - Use multiplication explicitly: `2*z` instead of `2z`
            - Available functions: sin, cos, tan, exp, log, sqrt, etc.
            """)
        elif "singular" in str(e).lower() or "domain" in str(e).lower():
            st.warning(f"The function may have singularities in the domain [{domain_min}, {domain_max}].")
            st.markdown("Try adjusting the domain range to avoid singularities like division by zero.") 