import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import null_space, orth

def explore_subspace(matrix, vector=None):
    """Explore the subspaces of a matrix"""
    # Ensure matrix is 2D or 3D
    if matrix.shape[0] not in [2, 3] or matrix.shape[1] not in [2, 3]:
        st.error("Matrix must be 2x2 or 3x3")
        return
    
    # Compute matrix properties
    rank = np.linalg.matrix_rank(matrix)
    det = np.linalg.det(matrix)
    is_singular = abs(det) < 1e-10
    
    # Compute subspaces
    try:
        col_space = orth(matrix)
        null_space_basis = null_space(matrix)
    except:
        col_space = None
        null_space_basis = None
    
    # Display matrix information
    st.markdown("### Matrix Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Matrix")
        st.latex(r"A = " + np.array2string(matrix, separator=', '))
        
        st.markdown("#### Rank")
        st.latex(r"\text{rank}(A) = " + str(rank))
        
        st.markdown("#### Dimension")
        st.latex(r"\dim(\text{Col}(A)) = " + str(rank))
        if null_space_basis is not None:
            st.latex(r"\dim(\text{Null}(A)) = " + str(null_space_basis.shape[1]))
    
    with col2:
        st.markdown("#### Subspace Properties")
        
        # Check if matrix is full rank
        if rank == matrix.shape[1]:
            st.success("Matrix is full rank")
        else:
            st.warning(f"Matrix is rank deficient (rank = {rank})")
        
        # Check if vector is in column space
        if vector is not None:
            try:
                # Solve Ax = b
                x = np.linalg.solve(matrix, vector)
                st.success("Vector is in the column space")
            except:
                st.warning("Vector is not in the column space")
    
    # Visualize subspaces
    st.markdown("### Subspace Visualization")
    
    if matrix.shape[0] == 2:
        fig = go.Figure()
        
        # Plot column space
        if col_space is not None:
            for i, col in enumerate(col_space.T):
                fig.add_trace(
                    go.Scatter(
                        x=[0, col[0]], 
                        y=[0, col[1]],
                        mode='lines',
                        name=f'Column Space Basis {i+1}',
                        line=dict(color='blue', width=2)
                    )
                )
        
        # Plot null space
        if null_space_basis is not None and null_space_basis.size > 0:
            for i, col in enumerate(null_space_basis.T):
                fig.add_trace(
                    go.Scatter(
                        x=[0, col[0]], 
                        y=[0, col[1]],
                        mode='lines',
                        name=f'Null Space Basis {i+1}',
                        line=dict(color='red', width=2, dash='dash')
                    )
                )
        
        # Plot vector if provided
        if vector is not None:
            fig.add_trace(
                go.Scatter(
                    x=[0, vector[0]], 
                    y=[0, vector[1]],
                    mode='lines+markers',
                    name='Input Vector',
                    line=dict(color='green', width=3),
                    marker=dict(size=10)
                )
            )
        
        fig.update_layout(
            showlegend=True,
            height=500,
            width=500,
            title="Subspaces"
        )
        
        # Set equal aspect ratio
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        
    else:  # 3D case
        fig = go.Figure()
        
        # Plot column space
        if col_space is not None:
            for i, col in enumerate(col_space.T):
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, col[0]], 
                        y=[0, col[1]],
                        z=[0, col[2]],
                        mode='lines',
                        name=f'Column Space Basis {i+1}',
                        line=dict(color='blue', width=2)
                    )
                )
        
        # Plot null space
        if null_space_basis is not None and null_space_basis.size > 0:
            for i, col in enumerate(null_space_basis.T):
                fig.add_trace(
                    go.Scatter3d(
                        x=[0, col[0]], 
                        y=[0, col[1]],
                        z=[0, col[2]],
                        mode='lines',
                        name=f'Null Space Basis {i+1}',
                        line=dict(color='red', width=2, dash='dash')
                    )
                )
        
        # Plot vector if provided
        if vector is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=[0, vector[0]], 
                    y=[0, vector[1]],
                    z=[0, vector[2]],
                    mode='lines+markers',
                    name='Input Vector',
                    line=dict(color='green', width=3),
                    marker=dict(size=5)
                )
            )
        
        fig.update_layout(
            showlegend=True,
            height=600,
            width=600,
            title="Subspaces",
            scene=dict(
                aspectmode='cube'
            )
        )
    
    st.plotly_chart(fig)
    
    # Display subspace information
    st.markdown("### Subspace Information")
    
    # Column space
    st.markdown("#### Column Space")
    if col_space is not None:
        st.latex(r"\text{Col}(A) = \text{span}\left(" + 
                ", ".join([f"\\begin{{bmatrix}}{col[0]} \\\\ {col[1]}" + 
                          (" \\\\ " + str(col[2]) if len(col) > 2 else "") + 
                          "\\end{bmatrix}" for col in col_space.T]) + 
                "\\right)")
    else:
        st.error("Could not compute column space basis")
    
    # Null space
    st.markdown("#### Null Space")
    if null_space_basis is not None and null_space_basis.size > 0:
        st.latex(r"\text{Null}(A) = \text{span}\left(" + 
                ", ".join([f"\\begin{{bmatrix}}{col[0]} \\\\ {col[1]}" + 
                          (" \\\\ " + str(col[2]) if len(col) > 2 else "") + 
                          "\\end{bmatrix}" for col in null_space_basis.T]) + 
                "\\right)")
    else:
        st.info("Null space is trivial (only contains the zero vector)")
    
    # Projection onto column space
    if vector is not None and col_space is not None:
        st.markdown("#### Projection onto Column Space")
        try:
            # Compute projection
            proj = col_space @ (col_space.T @ vector)
            st.latex(r"\text{proj}_{\text{Col}(A)}(\mathbf{v}) = " + 
                    f"\\begin{{bmatrix}}{proj[0]} \\\\ {proj[1]}" + 
                    (" \\\\ " + str(proj[2]) if len(proj) > 2 else "") + 
                    "\\end{bmatrix}")
        except:
            st.error("Could not compute projection") 