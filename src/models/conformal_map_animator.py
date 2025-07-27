import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.utilities.lambdify import lambdify
import io
from PIL import Image
import base64
from utils import get_figure_size, apply_image_container

def visualize_conformal_map(func_str, domain_min, domain_max, resolution):
    """
    Visualize a conformal mapping by showing how it transforms grid lines.
    
    Parameters:
    -----------
    func_str : str
        The complex function to visualize as a string (e.g., "1/z")
    domain_min : float
        Minimum value for both real and imaginary axes
    domain_max : float
        Maximum value for both real and imaginary axes
    resolution : int
        Grid resolution for visualization
    """
    try:
        # Set the current visualization type (wide aspect ratio)
        st.session_state["current_visualization"] = "conformal_map_wide"
        
        # Parse the function string
        z = sp.Symbol('z')
        transformations = (standard_transformations + 
                          (implicit_multiplication_application,))
        expr = parse_expr(func_str, local_dict={'z': z}, transformations=transformations)
        
        # Convert to a numpy function
        func = lambdify(z, expr, modules=['numpy'])
        
        # Define the grid for visualization
        grid_density = min(30, resolution // 15)  # Keep grid lines manageable
        
        # Create grid lines
        real_lines = np.linspace(domain_min, domain_max, grid_density)
        imag_lines = np.linspace(domain_min, domain_max, grid_density)
        
        # Get figure size from user preferences
        fig_size = get_figure_size()
        
        # Create figure for visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        
        # Plot 1: Original domain with grid
        ax1.set_title('Original Domain (z-plane)')
        ax1.set_xlabel('Re(z)')
        ax1.set_ylabel('Im(z)')
        ax1.set_xlim(domain_min, domain_max)
        ax1.set_ylim(domain_min, domain_max)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Draw the grid lines with different colors
        for i, x in enumerate(real_lines):
            h = i / len(real_lines)  # Hue value for coloring
            color = hsv_to_rgb([h, 0.7, 0.9])
            ax1.axvline(x=x, color=color, linestyle='-', linewidth=1, alpha=0.8)
            
        for i, y in enumerate(imag_lines):
            h = i / len(imag_lines)  # Hue value for coloring
            color = hsv_to_rgb([h, 0.7, 0.9])
            ax1.axhline(y=y, color=color, linestyle='-', linewidth=1, alpha=0.8)
        
        # Plot 2: Transformed domain
        ax2.set_title(f'Transformed Domain (w-plane)\nw = f(z) = {func_str}')
        ax2.set_xlabel('Re(w)')
        ax2.set_ylabel('Im(w)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Create dense sampling points along each grid line
        t = np.linspace(domain_min, domain_max, resolution)
        
        # Store w-values to determine appropriate axis limits
        all_w_real = []
        all_w_imag = []
        
        # Map vertical grid lines (constant real part)
        for i, x in enumerate(real_lines):
            z_line = x + 1j * t
            w_line = func(z_line)
            
            # Filter out infinities and NaNs
            valid_idx = np.isfinite(w_line)
            w_line = w_line[valid_idx]
            
            if len(w_line) > 0:
                h = i / len(real_lines)  # Hue value for coloring
                color = hsv_to_rgb([h, 0.7, 0.9])
                ax2.plot(w_line.real, w_line.imag, color=color, linewidth=1, alpha=0.8)
                
                # Store for axis limits calculation
                all_w_real.extend(w_line.real)
                all_w_imag.extend(w_line.imag)
        
        # Map horizontal grid lines (constant imaginary part)
        for i, y in enumerate(imag_lines):
            z_line = t + 1j * y
            w_line = func(z_line)
            
            # Filter out infinities and NaNs
            valid_idx = np.isfinite(w_line)
            w_line = w_line[valid_idx]
            
            if len(w_line) > 0:
                h = i / len(imag_lines)  # Hue value for coloring
                color = hsv_to_rgb([h, 0.7, 0.9])
                ax2.plot(w_line.real, w_line.imag, color=color, linewidth=1, alpha=0.8)
                
                # Store for axis limits calculation
                all_w_real.extend(w_line.real)
                all_w_imag.extend(w_line.imag)
        
        # Set appropriate axis limits for the transformed plot
        if all_w_real and all_w_imag:
            # Filter out extreme values
            filtered_real = np.array(all_w_real)
            filtered_imag = np.array(all_w_imag)
            
            # Remove outliers (values beyond 5 times the IQR)
            q1_real, q3_real = np.percentile(filtered_real, [25, 75])
            q1_imag, q3_imag = np.percentile(filtered_imag, [25, 75])
            iqr_real = q3_real - q1_real
            iqr_imag = q3_imag - q1_imag
            
            lower_real = q1_real - 2 * iqr_real
            upper_real = q3_real + 2 * iqr_real
            lower_imag = q1_imag - 2 * iqr_imag
            upper_imag = q3_imag + 2 * iqr_imag
            
            # Set the axis limits with some padding
            padding = 0.1
            ax2.set_xlim(lower_real - padding * (upper_real - lower_real),
                        upper_real + padding * (upper_real - lower_real))
            ax2.set_ylim(lower_imag - padding * (upper_imag - lower_imag),
                        upper_imag + padding * (upper_imag - lower_imag))
        
        plt.tight_layout()
        
        # Display the static visualization with resizable container
        apply_image_container(fig, caption=f"Static conformal mapping for f(z) = {func_str}")
        
        # Now create a simple animation showing the transformation
        st.markdown("### Transformation Animation")
        
        # Get animation figure size (slightly smaller)
        anim_size = (fig_size[0] * 0.8, fig_size[1] * 0.8)
        
        # Generate frames for animation
        frames = []
        num_frames = 21
        
        # Calculate limits for animation
        if all_w_real and all_w_imag:
            # Use the same limits as the static plot
            real_lim = ax2.get_xlim()
            imag_lim = ax2.get_ylim()
        else:
            real_lim = [domain_min, domain_max]
            imag_lim = [domain_min, domain_max]
        
        # Generate all frames manually instead of using FuncAnimation
        for frame in range(num_frames):
            # Create a new figure for each frame
            frame_fig, frame_ax = plt.subplots(figsize=anim_size)
            frame_ax.set_aspect('equal')
            frame_ax.grid(True, alpha=0.3)
            frame_ax.set_title(f'Transformation: w = f(z) = {func_str}')
            frame_ax.set_xlabel('Re')
            frame_ax.set_ylabel('Im')
            
            # Interpolation parameter (0 = original domain, 1 = fully transformed)
            t = frame / (num_frames - 1)
            
            # Update axis limits based on interpolation
            frame_ax.set_xlim((1-t)*domain_min + t*real_lim[0], (1-t)*domain_max + t*real_lim[1])
            frame_ax.set_ylim((1-t)*domain_min + t*imag_lim[0], (1-t)*domain_max + t*imag_lim[1])
            
            # Update grid lines
            for i, x in enumerate(real_lines):
                z_line = x + 1j * np.linspace(domain_min, domain_max, min(resolution, 100))
                w_line = func(z_line)
                
                # Filter out infinities and NaNs
                valid_idx = np.isfinite(w_line)
                z_valid = z_line[valid_idx]
                w_valid = w_line[valid_idx]
                
                if len(w_valid) > 0:
                    # Interpolate between z and w
                    h = i / len(real_lines)  # Hue value for coloring
                    color = hsv_to_rgb([h, 0.7, 0.9])
                    interpolated = (1 - t) * z_valid + t * w_valid
                    frame_ax.plot(interpolated.real, interpolated.imag, color=color, 
                                linewidth=1, alpha=0.8)
            
            for i, y in enumerate(imag_lines):
                z_line = np.linspace(domain_min, domain_max, min(resolution, 100)) + 1j * y
                w_line = func(z_line)
                
                # Filter out infinities and NaNs
                valid_idx = np.isfinite(w_line)
                z_valid = z_line[valid_idx]
                w_valid = w_line[valid_idx]
                
                if len(w_valid) > 0:
                    # Interpolate between z and w
                    h = i / len(imag_lines)  # Hue value for coloring
                    color = hsv_to_rgb([h, 0.7, 0.9])
                    interpolated = (1 - t) * z_valid + t * w_valid
                    frame_ax.plot(interpolated.real, interpolated.imag, color=color, 
                                linewidth=1, alpha=0.8)
            
            # Save frame to buffer and convert to image
            buf = io.BytesIO()
            plt.tight_layout()
            frame_fig.savefig(buf, format='png', dpi=70)
            buf.seek(0)
            frames.append(buf.read())
            plt.close(frame_fig)
        
        # Create animated GIF display with resizable container
        with st.container():
            if frames:
                # Create HTML for animation display with responsive styling
                html = """
                <div class="resizable-image-container" style="max-width: 100%;">
                    <img src="data:image/gif;base64,{}" style="max-width: 100%; height: auto;" alt="Animation">
                </div>
                """
                
                # Convert the frames to an animated GIF
                from PIL import Image
                images = [Image.open(io.BytesIO(frame)) for frame in frames]
                gif_buffer = io.BytesIO()
                images[0].save(
                    gif_buffer, 
                    format='GIF',
                    save_all=True,
                    append_images=images[1:],
                    optimize=True,
                    duration=100,
                    loop=0
                )
                gif_buffer.seek(0)
                
                # Display the animation
                gif_base64 = base64.b64encode(gif_buffer.read()).decode('utf-8')
                st.markdown(html.format(gif_base64), unsafe_allow_html=True)
                st.markdown(f"<div class='image-caption'>Animation of w = f(z) = {func_str}</div>", unsafe_allow_html=True)
            else:
                st.warning("Could not generate animation frames.")
        
        # Add educational explanation
        with st.expander("Understanding Conformal Mappings"):
            st.markdown("""
            ### Understanding Conformal Mappings
            
            Conformal mappings preserve angles between curves at their intersections. 
            This property makes them useful in many fields:
            
            - **Fluid Dynamics**: Transform complex flow problems to simpler geometries
            - **Electrostatics**: Map electric field configurations between different domains
            - **Cartography**: Create maps with angle-preserving projections
            - **Complex Analysis**: Study contour integrals and analytic continuations
            
            The grid lines that were perpendicular in the original domain remain perpendicular 
            after transformation, demonstrating the angle-preserving property.
            """)
        
        # Show properties of this specific map
        with st.expander("Properties of this Conformal Map"):
            # Check if z=0 is a special point
            try:
                w0 = complex(func(0))
                if np.isfinite(w0):
                    st.markdown(f"f(0) = {w0:.3g}")
                else:
                    st.markdown("z = 0 is a singularity of this function")
            except:
                st.markdown("z = 0 may be a singularity of this function")
            
            # Check if infinity is mapped to a finite point
            if "1/z" in func_str or "/z" in func_str:
                st.markdown("This function maps infinity to a finite point or vice versa.")
        
    except Exception as e:
        st.error(f"Error visualizing conformal map: {str(e)}")
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
            st.markdown("Try adjusting the domain to avoid points where the function is not defined.") 