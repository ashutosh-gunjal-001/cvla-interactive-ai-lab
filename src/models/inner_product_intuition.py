import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import norm

def inner_product_intuition(v1, v2, space_type='real'):
    """
    Visualize and explain inner products in different spaces.
    
    Parameters:
    -----------
    v1, v2 : np.ndarray
        Input vectors
    space_type : str
        Type of space ('real' or 'complex')
    """
    # Ensure vectors have same dimension
    if v1.shape != v2.shape:
        st.error("Vectors must have the same dimension")
        return
    
    # Convert to complex if needed
    if space_type == 'complex':
        v1 = v1.astype(complex)
        v2 = v2.astype(complex)
    
    # Compute inner product
    if space_type == 'real':
        inner_prod = np.dot(v1, v2)
    else:  # complex
        inner_prod = np.vdot(v1, v2)  # conjugate of first argument
    
    # Compute norms
    norm_v1 = norm(v1)
    norm_v2 = norm(v2)
    
    # Compute angle (only for real vectors)
    if space_type == 'real':
        cos_theta = inner_prod / (norm_v1 * norm_v2)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # Display mathematical explanation
    st.subheader("Mathematical Explanation")
    if space_type == 'real':
        st.latex(r"\langle \mathbf{v}_1, \mathbf{v}_2 \rangle = \mathbf{v}_1^T \mathbf{v}_2")
        st.latex(f"= {inner_prod:.4f}")
        st.latex(r"\cos(\theta) = \frac{\langle \mathbf{v}_1, \mathbf{v}_2 \rangle}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}")
        st.latex(f"= {cos_theta:.4f}")
        st.latex(f"\\theta = {np.degrees(theta):.2f}^\\circ")
    else:
        st.latex(r"\langle \mathbf{v}_1, \mathbf{v}_2 \rangle = \mathbf{v}_1^* \mathbf{v}_2")
        st.latex(f"= {inner_prod:.4f}")
    
    # Visualize vectors
    if len(v1) <= 3:  # Only visualize for 2D or 3D vectors
        fig = go.Figure()
        
        # Convert complex numbers to real parts for visualization
        v1_real = np.real(v1)
        v2_real = np.real(v2)
        
        # Add vectors
        if len(v1) == 2:
            fig.add_trace(go.Scatter(
                x=[0, v1_real[0]], y=[0, v1_real[1]],
                mode='lines+markers',
                name='v₁',
                line=dict(color='red', width=4),
                marker=dict(size=4)
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, v2_real[0]], y=[0, v2_real[1]],
                mode='lines+markers',
                name='v₂',
                line=dict(color='blue', width=4),
                marker=dict(size=4)
            ))
            
            # Add projection if real space
            if space_type == 'real':
                proj = (inner_prod / (norm_v1**2)) * v1_real
                fig.add_trace(go.Scatter(
                    x=[0, proj[0]], y=[0, proj[1]],
                    mode='lines',
                    name='Projection of v₂ onto v₁',
                    line=dict(color='green', width=2, dash='dash')
                ))
        else:  # 3D
            fig.add_trace(go.Scatter3d(
                x=[0, v1_real[0]], y=[0, v1_real[1]], z=[0, v1_real[2]],
                mode='lines+markers',
                name='v₁',
                line=dict(color='red', width=4),
                marker=dict(size=4)
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[0, v2_real[0]], y=[0, v2_real[1]], z=[0, v2_real[2]],
                mode='lines+markers',
                name='v₂',
                line=dict(color='blue', width=4),
                marker=dict(size=4)
            ))
            
            # Add projection if real space
            if space_type == 'real':
                proj = (inner_prod / (norm_v1**2)) * v1_real
                fig.add_trace(go.Scatter3d(
                    x=[0, proj[0]], y=[0, proj[1]], z=[0, proj[2]],
                    mode='lines',
                    name='Projection of v₂ onto v₁',
                    line=dict(color='green', width=2, dash='dash')
                ))
        
        # Update layout
        fig.update_layout(
            title='Vector Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z' if len(v1) == 3 else '',
                aspectmode='data'
            ),
            showlegend=True
        )
        st.plotly_chart(fig)
    
    # Display properties
    st.subheader("Properties")
    st.latex(f"\\|\mathbf{{v}}_1\\| = {norm_v1:.4f}")
    st.latex(f"\\|\mathbf{{v}}_2\\| = {norm_v2:.4f}")
    
    # Check orthogonality
    if abs(inner_prod) < 1e-10:
        st.success("Vectors are orthogonal")
    else:
        st.info("Vectors are not orthogonal")
    
    # Check if vectors are normalized
    if abs(norm_v1 - 1) < 1e-10:
        st.success("v₁ is normalized")
    else:
        st.info("v₁ is not normalized")
    
    if abs(norm_v2 - 1) < 1e-10:
        st.success("v₂ is normalized")
    else:
        st.info("v₂ is not normalized") 