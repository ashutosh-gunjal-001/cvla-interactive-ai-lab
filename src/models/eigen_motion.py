import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import eig

def simulate_eigen_motion(matrix, duration=5, steps=100):
    """
    Simulate and visualize how vectors are transformed by a matrix using its eigenvectors.
    
    Parameters:
    -----------
    matrix : np.ndarray
        2x2 or 3x3 matrix
    duration : float
        Duration of animation in seconds
    steps : int
        Number of steps in the animation
    """
    try:
        # Input validation
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix, dtype=float)
        
        # Ensure matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            st.error("Matrix must be square")
            return
        
        # Ensure matrix is 2D or 3D
        if matrix.shape[0] not in [2, 3]:
            st.error("Matrix must be 2x2 or 3x3")
            return
        
        # Check for NaN or Inf values
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            st.error("Matrix contains NaN or infinite values")
            return
        
        # Scale matrix for better numerical stability
        max_val = np.max(np.abs(matrix))
        if max_val > 0:
            matrix = matrix / max_val
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eig(matrix)
        
        # Display eigenvalues and eigenvectors
        st.markdown("### Eigenvalues and Eigenvectors")
        for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            # Format complex numbers properly
            if np.iscomplex(val):
                val_str = f"{np.real(val):.2f} {'+' if np.imag(val) >= 0 else '-'} {abs(np.imag(val)):.2f}i"
            else:
                val_str = f"{val:.2f}"
            
            # Format eigenvectors
            vec_str = []
            for x in vec:
                if np.iscomplex(x):
                    vec_str.append(f"{np.real(x):.2f} {'+' if np.imag(x) >= 0 else '-'} {abs(np.imag(x)):.2f}i")
                else:
                    vec_str.append(f"{x:.2f}")
            
            st.latex(f"\\lambda_{i+1} = {val_str}")
            st.latex(f"\\mathbf{{v}}_{i+1} = \\begin{{bmatrix}} {' \\\\ '.join(vec_str)} \\end{{bmatrix}}")
        
        # Create animation frames
        frames = []
        t = np.linspace(0, 2*np.pi, steps)
        
        if matrix.shape[0] == 2:
            # Create unit circle points
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            unit_circle = np.vstack((circle_x, circle_y))
            
            # Transform unit circle
            transformed_circle = matrix @ unit_circle
            
            # Create figure
            fig = go.Figure()
            
            # Add unit circle
            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                name='Unit Circle',
                line=dict(color='blue', width=2)
            ))
            
            # Add transformed circle
            fig.add_trace(go.Scatter(
                x=transformed_circle[0],
                y=transformed_circle[1],
                mode='lines',
                name='Transformed Circle',
                line=dict(color='red', width=2)
            ))
            
            # Add eigenvectors
            for i, vec in enumerate(eigenvectors.T):
                # Scale eigenvectors for visibility
                scale = 2.0
                scaled_vec = scale * np.real(vec)
                fig.add_trace(go.Scatter(
                    x=[0, scaled_vec[0]],
                    y=[0, scaled_vec[1]],
                    mode='lines+markers',
                    name=f'Eigenvector {i+1}',
                    line=dict(color='green', width=2)
                ))
            
            # Update layout
            fig.update_layout(
                title='Matrix Transformation Visualization',
                xaxis_title='X',
                yaxis_title='Y',
                showlegend=True
            )
            
        else:  # 3D case
            # Create unit sphere points
            phi = np.linspace(0, np.pi, 20)
            theta = np.linspace(0, 2*np.pi, 20)
            phi, theta = np.meshgrid(phi, theta)
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            # Flatten for transformation
            points = np.vstack((x.flatten(), y.flatten(), z.flatten()))
            transformed_points = matrix @ points
            
            # Reshape back
            x_trans = transformed_points[0].reshape(x.shape)
            y_trans = transformed_points[1].reshape(y.shape)
            z_trans = transformed_points[2].reshape(z.shape)
            
            # Create figure
            fig = go.Figure()
            
            # Add unit sphere
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.3,
                colorscale='Blues',
                name='Unit Sphere'
            ))
            
            # Add transformed sphere
            fig.add_trace(go.Surface(
                x=x_trans, y=y_trans, z=z_trans,
                opacity=0.3,
                colorscale='Reds',
                name='Transformed Sphere'
            ))
            
            # Add eigenvectors
            for i, vec in enumerate(eigenvectors.T):
                # Scale eigenvectors for visibility
                scale = 2.0
                scaled_vec = scale * np.real(vec)
                fig.add_trace(go.Scatter3d(
                    x=[0, scaled_vec[0]],
                    y=[0, scaled_vec[1]],
                    z=[0, scaled_vec[2]],
                    mode='lines+markers',
                    name=f'Eigenvector {i+1}',
                    line=dict(color='green', width=4)
                ))
            
            # Update layout
            fig.update_layout(
                title='3D Matrix Transformation Visualization',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='cube'
                ),
                showlegend=True
            )
        
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error in eigen motion simulation: {str(e)}")
        st.error("Please check your matrix and try again.") 