import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import orth

def classify_basis(matrix):
    """Analyze and classify the basis of a matrix"""
    # Ensure matrix is 2D or 3D
    if matrix.shape[0] not in [2, 3] or matrix.shape[1] not in [2, 3]:
        st.error("Matrix must be 2x2 or 3x3")
        return
    
    # Compute matrix properties
    rank = np.linalg.matrix_rank(matrix)
    det = np.linalg.det(matrix)
    is_singular = abs(det) < 1e-10
    
    # Compute orthonormal basis
    try:
        orth_basis = orth(matrix)
    except:
        orth_basis = None
    
    # Display matrix information
    st.markdown("### Matrix Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Matrix")
        st.latex(r"A = " + np.array2string(matrix, separator=', '))
        
        st.markdown("#### Rank")
        st.latex(r"\text{rank}(A) = " + str(rank))
        
        st.markdown("#### Determinant")
        st.latex(r"\det(A) = " + f"{det:.4f}")
    
    with col2:
        st.markdown("#### Basis Properties")
        
        # Check linear independence
        if rank == matrix.shape[1]:
            st.success("Columns are linearly independent")
        else:
            st.warning(f"Columns are linearly dependent (rank = {rank})")
        
        # Check orthogonality
        if orth_basis is not None and np.allclose(matrix, orth_basis):
            st.success("Columns form an orthogonal basis")
        else:
            st.info("Columns do not form an orthogonal basis")
        
        # Check if basis
        if rank == matrix.shape[1] == matrix.shape[0]:
            st.success("Columns form a basis for the space")
        else:
            st.warning("Columns do not form a basis for the space")
    
    # Visualize the basis vectors
    st.markdown("### Basis Visualization")
    
    if matrix.shape[0] == 2:
        fig = go.Figure()
        
        # Plot basis vectors
        for i, col in enumerate(matrix.T):
            fig.add_trace(
                go.Scatter(
                    x=[0, col[0]], 
                    y=[0, col[1]],
                    mode='lines+markers',
                    name=f'Column {i+1}',
                    line=dict(width=3),
                    marker=dict(size=10)
                )
            )
        
        # Plot orthonormal basis if available
        if orth_basis is not None:
            for i, col in enumerate(orth_basis.T):
                fig.add_trace(
                    go.Scatter(
                        x=[0, col[0]], 
                        y=[0, col[1]],
                        mode='lines',
                        name=f'Orthonormal {i+1}',
                        line=dict(dash='dash', width=2)
                    )
                )
        
        fig.update_layout(
            showlegend=True,
            height=500,
            width=500,
            title="Basis Vectors"
        )
        
        # Set equal aspect ratio
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        
    else:  # 3D case
        fig = go.Figure()
        
        # Plot basis vectors
        for i, col in enumerate(matrix.T):
            fig.add_trace(
                go.Scatter3d(
                    x=[0, col[0]], 
                    y=[0, col[1]],
                    z=[0, col[2]],
                    mode='lines+markers',
                    name=f'Column {i+1}',
                    line=dict(width=3),
                    marker=dict(size=5)
                )
            )
        
        # Plot orthonormal basis if available
        if orth_basis is not None:
            for i, col in enumerate(orth_basis.T):
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, col[0]], 
                        y=[0, col[1]],
                        z=[0, col[2]],
                        mode='lines',
                        name=f'Orthonormal {i+1}',
                        line=dict(dash='dash', width=2)
                    )
                )
        
        fig.update_layout(
            showlegend=True,
            height=600,
            width=600,
            title="Basis Vectors",
            scene=dict(
                aspectmode='cube'
            )
        )
    
    st.plotly_chart(fig)
    
    # Display additional information
    st.markdown("### Additional Information")
    
    # Condition number
    try:
        cond = np.linalg.cond(matrix)
        st.markdown("#### Condition Number")
        st.latex(r"\kappa(A) = " + f"{cond:.4f}")
        if cond > 100:
            st.warning("Matrix is ill-conditioned (high condition number)")
    except:
        st.error("Could not compute condition number")
    
    # Singular values
    try:
        svd = np.linalg.svd(matrix, compute_uv=False)
        st.markdown("#### Singular Values")
        st.latex(r"\sigma = " + np.array2string(svd, separator=', '))
    except:
        st.error("Could not compute singular values")
    
    # Gram-Schmidt process visualization
    if not is_singular:
        st.markdown("### Gram-Schmidt Process")
        
        # Perform Gram-Schmidt
        q, r = np.linalg.qr(matrix)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Orthogonal Matrix Q")
            st.latex(r"Q = " + np.array2string(q, separator=', '))
        
        with col2:
            st.markdown("#### Upper Triangular Matrix R")
            st.latex(r"R = " + np.array2string(r, separator=', ')) 