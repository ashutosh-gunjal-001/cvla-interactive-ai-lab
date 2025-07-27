import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import norm

def gram_schmidt_animator(vectors, space_type='real'):
    """
    Visualize the Gram-Schmidt process step by step.
    
    Parameters:
    -----------
    vectors : list of np.ndarray
        Input vectors to orthogonalize
    space_type : str
        Type of space ('real' or 'complex')
    """
    # Input validation
    if not vectors:
        st.error("No vectors provided")
        return
    
    # Convert to numpy array and ensure proper shape
    try:
        vectors = [np.array(v, dtype=complex if space_type == 'complex' else float) for v in vectors]
    except Exception as e:
        st.error(f"Error converting vectors: {str(e)}")
        return
    
    # Check vector dimensions
    dim = len(vectors[0])
    if not all(len(v) == dim for v in vectors):
        st.error("All vectors must have the same dimension")
        return
    
    # Check for zero vectors
    if any(np.all(np.abs(v) < 1e-10) for v in vectors):
        st.error("Zero vectors are not allowed")
        return
    
    # Scale vectors for better numerical stability
    max_norm = max(norm(v) for v in vectors)
    if max_norm > 0:
        vectors = [v / max_norm for v in vectors]
    
    # Check if vectors are linearly independent
    matrix = np.column_stack(vectors)
    if np.linalg.matrix_rank(matrix) < len(vectors):
        st.error("Vectors must be linearly independent")
        return
    
    # Initialize lists for orthogonal and orthonormal vectors
    orthogonal = []
    orthonormal = []
    
    # Perform Gram-Schmidt process
    for i, v in enumerate(vectors):
        # Start with original vector
        u = v.copy()
        
        # Subtract projections onto previous vectors
        for j in range(i):
            if space_type == 'real':
                proj = np.dot(u, orthogonal[j]) / np.dot(orthogonal[j], orthogonal[j]) * orthogonal[j]
            else:  # complex
                proj = np.vdot(u, orthogonal[j]) / np.vdot(orthogonal[j], orthogonal[j]) * orthogonal[j]
            u = u - proj
        
        # Add to orthogonal list
        orthogonal.append(u)
        
        # Normalize
        u_norm = norm(u)
        if u_norm > 1e-10:  # Avoid division by zero
            orthonormal.append(u / u_norm)
        else:
            st.error("Zero vector produced in Gram-Schmidt process")
            return
    
    # Display process steps
    st.subheader("Gram-Schmidt Process Steps")
    
    for i, (v, u, e) in enumerate(zip(vectors, orthogonal, orthonormal)):
        st.markdown(f"### Step {i+1}")
        
        # Display original vector
        st.latex(f"\\mathbf{{v}}_{i+1} = {np.array2string(v, precision=2)}")
        
        # Display orthogonal vector
        st.latex(f"\\mathbf{{u}}_{i+1} = {np.array2string(u, precision=2)}")
        
        # Display orthonormal vector
        st.latex(f"\\mathbf{{e}}_{i+1} = {np.array2string(e, precision=2)}")
        
        # Visualize vectors (for 2D or 3D)
        if len(v) <= 3:
            fig = go.Figure()
            
            # Convert complex numbers to real for visualization
            v_real = np.real(v)
            u_real = np.real(u)
            e_real = np.real(e)
            
            # Add original vector
            fig.add_trace(go.Scatter3d(
                x=[0, v_real[0]], y=[0, v_real[1]], z=[0, v_real[2]] if len(v) == 3 else [0, 0],
                mode='lines+markers',
                name=f'v{i+1} (real part)',
                line=dict(color='red', width=4),
                marker=dict(size=4)
            ))
            
            # Add orthogonal vector
            fig.add_trace(go.Scatter3d(
                x=[0, u_real[0]], y=[0, u_real[1]], z=[0, u_real[2]] if len(u) == 3 else [0, 0],
                mode='lines+markers',
                name=f'u{i+1} (real part)',
                line=dict(color='blue', width=4),
                marker=dict(size=4)
            ))
            
            # Add orthonormal vector
            fig.add_trace(go.Scatter3d(
                x=[0, e_real[0]], y=[0, e_real[1]], z=[0, e_real[2]] if len(e) == 3 else [0, 0],
                mode='lines+markers',
                name=f'e{i+1} (real part)',
                line=dict(color='green', width=4),
                marker=dict(size=4)
            ))
            
            # Add previous orthogonal vectors
            for j in range(i):
                prev_u = np.real(orthogonal[j])
                fig.add_trace(go.Scatter3d(
                    x=[0, prev_u[0]], y=[0, prev_u[1]], z=[0, prev_u[2]] if len(prev_u) == 3 else [0, 0],
                    mode='lines',
                    name=f'u{j+1} (real part)',
                    line=dict(color='gray', width=2, dash='dash')
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Step {i+1} Visualization (Real Parts Only)',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z' if len(v) == 3 else '',
                    aspectmode='data'
                ),
                showlegend=True
            )
            st.plotly_chart(fig)
            
            # If complex, show imaginary parts separately
            if space_type == 'complex':
                v_imag = np.imag(v)
                u_imag = np.imag(u)
                e_imag = np.imag(e)
                
                fig_imag = go.Figure()
                
                # Add original vector
                fig_imag.add_trace(go.Scatter3d(
                    x=[0, v_imag[0]], y=[0, v_imag[1]], z=[0, v_imag[2]] if len(v) == 3 else [0, 0],
                    mode='lines+markers',
                    name=f'v{i+1} (imaginary part)',
                    line=dict(color='red', width=4),
                    marker=dict(size=4)
                ))
                
                # Add orthogonal vector
                fig_imag.add_trace(go.Scatter3d(
                    x=[0, u_imag[0]], y=[0, u_imag[1]], z=[0, u_imag[2]] if len(u) == 3 else [0, 0],
                    mode='lines+markers',
                    name=f'u{i+1} (imaginary part)',
                    line=dict(color='blue', width=4),
                    marker=dict(size=4)
                ))
                
                # Add orthonormal vector
                fig_imag.add_trace(go.Scatter3d(
                    x=[0, e_imag[0]], y=[0, e_imag[1]], z=[0, e_imag[2]] if len(e) == 3 else [0, 0],
                    mode='lines+markers',
                    name=f'e{i+1} (imaginary part)',
                    line=dict(color='green', width=4),
                    marker=dict(size=4)
                ))
                
                # Add previous orthogonal vectors
                for j in range(i):
                    prev_u = np.imag(orthogonal[j])
                    fig_imag.add_trace(go.Scatter3d(
                        x=[0, prev_u[0]], y=[0, prev_u[1]], z=[0, prev_u[2]] if len(prev_u) == 3 else [0, 0],
                        mode='lines',
                        name=f'u{j+1} (imaginary part)',
                        line=dict(color='gray', width=2, dash='dash')
                    ))
                
                # Update layout
                fig_imag.update_layout(
                    title=f'Step {i+1} Visualization (Imaginary Parts Only)',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z' if len(v) == 3 else '',
                        aspectmode='data'
                    ),
                    showlegend=True
                )
                st.plotly_chart(fig_imag)
    
    # Display final orthonormal basis
    st.subheader("Final Orthonormal Basis")
    for i, e in enumerate(orthonormal):
        st.latex(f"\\mathbf{{e}}_{i+1} = {np.array2string(e, precision=2)}")
    
    # Verify orthonormality
    st.subheader("Orthonormality Verification")
    for i in range(len(orthonormal)):
        for j in range(len(orthonormal)):
            if space_type == 'real':
                inner_prod = np.dot(orthonormal[i], orthonormal[j])
            else:
                inner_prod = np.vdot(orthonormal[i], orthonormal[j])
            
            if i == j:
                if abs(inner_prod - 1) < 1e-10:
                    st.success(f"e{i+1} is normalized")
                else:
                    st.error(f"e{i+1} is not normalized (inner product = {inner_prod:.2e})")
            else:
                if abs(inner_prod) < 1e-10:
                    st.success(f"e{i+1} and e{j+1} are orthogonal")
                else:
                    st.error(f"e{i+1} and e{j+1} are not orthogonal (inner product = {inner_prod:.2e})") 