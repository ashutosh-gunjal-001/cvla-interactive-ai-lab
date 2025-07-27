import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import inv

def animate_inversion(matrix):
    """Animate the process of matrix inversion"""
    # Ensure matrix is 2D or 3D
    if matrix.shape[0] not in [2, 3] or matrix.shape[1] not in [2, 3]:
        st.error("Matrix must be 2x2 or 3x3")
        return
    
    # Check if matrix is invertible
    det = np.linalg.det(matrix)
    if abs(det) < 1e-10:
        st.error("Matrix is not invertible (determinant is zero)")
        return
    
    # Compute inverse
    try:
        inverse = inv(matrix)
    except:
        st.error("Could not compute matrix inverse")
        return
    
    # Display original matrix and its inverse
    st.markdown("### Matrix Inversion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Matrix")
        st.latex(r"A = " + np.array2string(matrix, separator=', '))
        st.markdown("#### Determinant")
        st.latex(r"\det(A) = " + str(det))
    
    with col2:
        st.markdown("#### Inverse Matrix")
        st.latex(r"A^{-1} = " + np.array2string(inverse, separator=', '))
        st.markdown("#### Verification")
        st.latex(r"A \cdot A^{-1} = I")
        st.latex(np.array2string(matrix @ inverse, separator=', '))
    
    # Create animation frames
    frames = []
    steps = 20
    
    # For 2D case
    if matrix.shape[0] == 2:
        # Create unit square
        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
        
        # Create figure
        fig = go.Figure()
        
        # Add original square
        fig.add_trace(
            go.Scatter(
                x=unit_square[:, 0],
                y=unit_square[:, 1],
                mode='lines',
                name='Original Square',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add transformed square
        transformed_square = unit_square @ matrix
        fig.add_trace(
            go.Scatter(
                x=transformed_square[:, 0],
                y=transformed_square[:, 1],
                mode='lines',
                name='Transformed Square',
                line=dict(color='red', width=2)
            )
        )
        
        # Create animation frames
        for i in range(steps + 1):
            t = i / steps
            interpolated_matrix = (1 - t) * matrix + t * inverse
            interpolated_square = unit_square @ interpolated_matrix
            
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter(
                            x=unit_square[:, 0],
                            y=unit_square[:, 1],
                            mode='lines',
                            line=dict(color='blue', width=2)
                        ),
                        go.Scatter(
                            x=interpolated_square[:, 0],
                            y=interpolated_square[:, 1],
                            mode='lines',
                            line=dict(color='red', width=2)
                        )
                    ],
                    name=f'frame_{i}'
                )
            )
        
        fig.frames = frames
        
        # Add play button
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                        )
                    ]
                )
            ]
        )
        
    else:  # 3D case
        # Create unit cube
        unit_cube = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1],
            [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0]
        ])
        
        # Create figure
        fig = go.Figure()
        
        # Add original cube
        fig.add_trace(
            go.Scatter3d(
                x=unit_cube[:, 0],
                y=unit_cube[:, 1],
                z=unit_cube[:, 2],
                mode='lines',
                name='Original Cube',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add transformed cube
        transformed_cube = unit_cube @ matrix
        fig.add_trace(
            go.Scatter3d(
                x=transformed_cube[:, 0],
                y=transformed_cube[:, 1],
                z=transformed_cube[:, 2],
                mode='lines',
                name='Transformed Cube',
                line=dict(color='red', width=2)
            )
        )
        
        # Create animation frames
        for i in range(steps + 1):
            t = i / steps
            interpolated_matrix = (1 - t) * matrix + t * inverse
            interpolated_cube = unit_cube @ interpolated_matrix
            
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter3d(
                            x=unit_cube[:, 0],
                            y=unit_cube[:, 1],
                            z=unit_cube[:, 2],
                            mode='lines',
                            line=dict(color='blue', width=2)
                        ),
                        go.Scatter3d(
                            x=interpolated_cube[:, 0],
                            y=interpolated_cube[:, 1],
                            z=interpolated_cube[:, 2],
                            mode='lines',
                            line=dict(color='red', width=2)
                        )
                    ],
                    name=f'frame_{i}'
                )
            )
        
        fig.frames = frames
        
        # Add play button
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                        )
                    ]
                )
            ],
            scene=dict(
                aspectmode='cube'
            )
        )
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        height=600,
        width=600,
        title="Matrix Inversion Animation"
    )
    
    st.plotly_chart(fig)
    
    # Display additional information
    st.markdown("### Matrix Properties")
    
    # Eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        st.markdown("#### Eigenvalues")
        st.latex(r"\lambda = " + np.array2string(eigenvalues, separator=', '))
        
        st.markdown("#### Eigenvectors")
        for i, vec in enumerate(eigenvectors.T):
            st.latex(r"\mathbf{v}_{" + str(i+1) + "} = " + 
                    f"\\begin{{bmatrix}}{vec[0]} \\\\ {vec[1]}" + 
                    (" \\\\ " + str(vec[2]) if len(vec) > 2 else "") + 
                    "\\end{bmatrix}")
    except:
        st.warning("Could not compute eigenvalues and eigenvectors")
    
    # Condition number
    try:
        cond = np.linalg.cond(matrix)
        st.markdown("#### Condition Number")
        st.latex(r"\kappa(A) = " + str(cond))
        if cond > 1000:
            st.warning("Matrix is ill-conditioned")
    except:
        st.warning("Could not compute condition number") 