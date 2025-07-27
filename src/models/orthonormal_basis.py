import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import qr, svd, norm

def orthonormal_basis_generator(dimension, space_type='real', method='qr'):
    """
    Generate and verify orthonormal bases for different spaces.
    
    Parameters:
    -----------
    dimension : int
        Dimension of the space
    space_type : str
        Type of space ('real' or 'complex')
    method : str
        Method to generate basis ('qr', 'svd', or 'random')
    """
    # Generate random matrix
    if space_type == 'real':
        A = np.random.randn(dimension, dimension)
    else:  # complex
        A = np.random.randn(dimension, dimension) + 1j * np.random.randn(dimension, dimension)
    
    # Generate orthonormal basis
    if method == 'qr':
        Q, _ = qr(A)
        basis = Q
    elif method == 'svd':
        U, _, _ = svd(A)
        basis = U
    else:  # random
        basis = np.random.randn(dimension, dimension)
        if space_type == 'complex':
            basis = basis + 1j * np.random.randn(dimension, dimension)
        # Normalize columns
        for i in range(dimension):
            basis[:, i] = basis[:, i] / norm(basis[:, i])
    
    # Display basis
    st.subheader("Generated Orthonormal Basis")
    st.latex("B = " + str(basis))
    
    # Verify orthonormality
    st.subheader("Orthonormality Verification")
    
    # Check pairwise orthogonality and normalization
    for i in range(dimension):
        for j in range(dimension):
            if space_type == 'real':
                inner_prod = np.dot(basis[:, i], basis[:, j])
            else:
                inner_prod = np.vdot(basis[:, i], basis[:, j])
            
            if i == j:
                if abs(inner_prod - 1) < 1e-10:
                    st.success(f"Vector {i+1} is normalized")
                else:
                    st.error(f"Vector {i+1} is not normalized")
            else:
                if abs(inner_prod) < 1e-10:
                    st.success(f"Vectors {i+1} and {j+1} are orthogonal")
                else:
                    st.error(f"Vectors {i+1} and {j+1} are not orthogonal")
    
    # Visualize basis vectors (for 2D or 3D)
    if dimension <= 3:
        fig = go.Figure()
        
        # Add basis vectors
        colors = ['red', 'blue', 'green']
        for i in range(dimension):
            if dimension == 2:
                fig.add_trace(go.Scatter(
                    x=[0, np.real(basis[0, i])],
                    y=[0, np.real(basis[1, i])],
                    mode='lines+markers',
                    name=f'v{i+1}',
                    line=dict(color=colors[i], width=4),
                    marker=dict(size=4)
                ))
            else:  # 3D
                fig.add_trace(go.Scatter3d(
                    x=[0, np.real(basis[0, i])],
                    y=[0, np.real(basis[1, i])],
                    z=[0, np.real(basis[2, i])],
                    mode='lines+markers',
                    name=f'v{i+1}',
                    line=dict(color=colors[i], width=4),
                    marker=dict(size=4)
                ))
        
        # Add unit sphere/circle
        if dimension == 2:
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode='lines',
                name='Unit Circle',
                line=dict(color='gray', dash='dash')
            ))
        else:  # 3D
            phi = np.linspace(0, np.pi, 20)
            theta = np.linspace(0, 2*np.pi, 20)
            phi, theta = np.meshgrid(phi, theta)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                opacity=0.2,
                colorscale='gray',
                showscale=False,
                name='Unit Sphere'
            ))
        
        # Update layout
        fig.update_layout(
            title='Basis Vectors',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z' if dimension == 3 else None,
                aspectmode='data'
            ),
            showlegend=True
        )
        st.plotly_chart(fig)
    
    # Check if basis spans the space
    st.subheader("Span Verification")
    if np.linalg.matrix_rank(basis) == dimension:
        st.success("Basis spans the entire space")
    else:
        st.error("Basis does not span the entire space")
    
    # Check if basis is complete
    st.subheader("Completeness Verification")
    if abs(np.linalg.det(basis)) > 1e-10:
        st.success("Basis is complete")
    else:
        st.error("Basis is not complete")
    
    # Display change of basis matrix
    st.subheader("Change of Basis Matrix")
    if space_type == 'real':
        st.latex("B^T B = " + str(basis.T @ basis))
    else:
        st.latex("B^* B = " + str(basis.conj().T @ basis))
    
    # Check if basis is orthonormal
    if space_type == 'real':
        identity_check = basis.T @ basis
    else:
        identity_check = basis.conj().T @ basis
    
    if np.allclose(identity_check, np.eye(dimension)):
        st.success("Basis is orthonormal")
    else:
        st.error("Basis is not orthonormal") 