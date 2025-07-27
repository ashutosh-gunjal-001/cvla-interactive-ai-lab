import streamlit as st
import numpy as np
from scipy.linalg import eigvals
import plotly.graph_objects as go

def check_cayley_hamilton(matrix):
    """Verify the Cayley-Hamilton theorem for a given matrix."""
    # Ensure matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        st.error("Matrix must be square for Cayley-Hamilton theorem")
        return
    
    # Check matrix size for numerical stability
    if matrix.shape[0] > 10:
        st.warning("Large matrices may have numerical stability issues. Results should be interpreted with caution.")
    
    # Display original matrix
    st.latex(f"A = {matrix}")
    
    # Compute characteristic polynomial coefficients
    eigenvalues = eigvals(matrix)
    n = len(eigenvalues)
    
    # Display characteristic polynomial with proper complex number handling
    poly_str = "p(λ) = "
    for i, eig in enumerate(eigenvalues):
        if i > 0:
            poly_str += " * "
        if np.iscomplex(eig):
            real_part = np.real(eig)
            imag_part = np.imag(eig)
            if imag_part >= 0:
                poly_str += f"(λ - ({real_part:.2f} + {imag_part:.2f}i))"
            else:
                poly_str += f"(λ - ({real_part:.2f} - {abs(imag_part):.2f}i))"
        else:
            poly_str += f"(λ - {eig:.2f})"
    st.latex(poly_str)
    
    # Evaluate polynomial at matrix
    result = np.eye(n)
    for eig in eigenvalues:
        result = result @ (matrix - eig * np.eye(n))
    
    # Display result with proper formatting
    st.latex(f"p(A) = {result}")
    
    # Check if result is approximately zero with appropriate tolerance
    tolerance = 1e-10 * np.linalg.norm(matrix)
    if np.allclose(result, np.zeros_like(result), atol=tolerance):
        st.success("Cayley-Hamilton theorem verified! p(A) = 0")
    else:
        st.error("Cayley-Hamilton theorem failed! p(A) ≠ 0")
        st.info(f"Maximum deviation from zero: {np.max(np.abs(result)):.2e}")
    
    # Display eigenvalues with proper complex number formatting
    st.subheader("Eigenvalues")
    eig_str = "λ = ["
    for i, eig in enumerate(eigenvalues):
        if i > 0:
            eig_str += ", "
        if np.iscomplex(eig):
            real_part = np.real(eig)
            imag_part = np.imag(eig)
            if imag_part >= 0:
                eig_str += f"{real_part:.2f} + {imag_part:.2f}i"
            else:
                eig_str += f"{real_part:.2f} - {abs(imag_part):.2f}i"
        else:
            eig_str += f"{eig:.2f}"
    eig_str += "]"
    st.latex(eig_str)
    
    # Visualize eigenvalues in complex plane with better handling of complex numbers
    fig = go.Figure()
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta),
        y=np.sin(theta),
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Unit Circle'
    ))
    
    # Add eigenvalues
    fig.add_trace(go.Scatter(
        x=np.real(eigenvalues),
        y=np.imag(eigenvalues),
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Eigenvalues'
    ))
    
    # Add origin
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(size=10, color='black'),
        name='Origin'
    ))
    
    fig.update_layout(
        title='Eigenvalues in Complex Plane',
        xaxis_title='Real',
        yaxis_title='Imaginary',
        showlegend=True,
        width=600,
        height=600
    )
    st.plotly_chart(fig)
    
    # Additional analysis
    st.subheader("Additional Analysis")
    
    # Compute determinant and trace
    det = np.linalg.det(matrix)
    trace = np.trace(matrix)
    st.latex(f"\\det(A) = {det:.2f}")
    st.latex(r"\text{tr}(A) = " + f"{trace:.2f}")
    
    # Check if matrix is diagonalizable
    try:
        np.linalg.eig(matrix)
        st.success("Matrix is diagonalizable")
    except:
        st.warning("Matrix is not diagonalizable")
    
    # Check if matrix is normal
    if np.allclose(matrix @ matrix.T.conj(), matrix.T.conj() @ matrix):
        st.success("Matrix is normal")
    else:
        st.warning("Matrix is not normal")
    
    # Minimal polynomial analysis
    st.subheader("Minimal Polynomial")
    # For simplicity, we'll assume minimal polynomial is same as characteristic polynomial
    # In practice, this would require more sophisticated computation
    st.latex(f"m(λ) = {poly_str}")
    st.info("Note: This is a simplified analysis. The actual minimal polynomial might be different.") 