import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def explore_pca(data, n_components=None):
    """
    Explore and visualize PCA transformation of data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data matrix (samples × features)
    n_components : int, optional
        Number of components to keep
    """
    if data.shape[1] < 2:
        st.error("Data must have at least 2 features")
        return
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Compute PCA
    if n_components is None:
        n_components = min(data.shape[1], 3)
    
    pca = PCA(n_components=n_components)
    data_transformed = pca.fit_transform(data_scaled)
    
    # Display explained variance ratio
    st.markdown("### Explained Variance Ratio")
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    fig_variance = go.Figure()
    
    # Add individual explained variance
    fig_variance.add_trace(
        go.Bar(
            x=[f"PC{i+1}" for i in range(len(explained_variance))],
            y=explained_variance * 100,
            name="Individual"
        )
    )
    
    # Add cumulative explained variance
    fig_variance.add_trace(
        go.Scatter(
            x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
            y=cumulative_variance * 100,
            name="Cumulative",
            yaxis="y2"
        )
    )
    
    fig_variance.update_layout(
        title="Explained Variance by Principal Components",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance (%)",
        yaxis2=dict(
            title="Cumulative Explained Variance (%)",
            overlaying="y",
            side="right"
        ),
        showlegend=True
    )
    
    st.plotly_chart(fig_variance)
    
    # Display component loadings
    st.markdown("### Component Loadings")
    loadings = pca.components_
    
    fig_loadings = go.Figure()
    
    for i in range(n_components):
        fig_loadings.add_trace(
            go.Bar(
                x=[f"Feature {j+1}" for j in range(data.shape[1])],
                y=loadings[i],
                name=f"PC{i+1}"
            )
        )
    
    fig_loadings.update_layout(
        title="PCA Component Loadings",
        xaxis_title="Features",
        yaxis_title="Loading Value",
        barmode='group',
        showlegend=True
    )
    
    st.plotly_chart(fig_loadings)
    
    # Visualize transformed data
    st.markdown("### Data Visualization")
    
    if data_transformed.shape[1] >= 2:
        fig_transform = go.Figure()
        
        if data_transformed.shape[1] == 2:
            # 2D scatter plot
            fig_transform.add_trace(
                go.Scatter(
                    x=data_transformed[:, 0],
                    y=data_transformed[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=np.arange(len(data_transformed)),
                        colorscale='Viridis',
                        showscale=True
                    )
                )
            )
            
            fig_transform.update_layout(
                title="Data in Principal Component Space",
                xaxis_title="First Principal Component",
                yaxis_title="Second Principal Component"
            )
            
        else:  # 3D scatter plot
            fig_transform.add_trace(
                go.Scatter3d(
                    x=data_transformed[:, 0],
                    y=data_transformed[:, 1],
                    z=data_transformed[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=np.arange(len(data_transformed)),
                        colorscale='Viridis',
                        showscale=True
                    )
                )
            )
            
            fig_transform.update_layout(
                title="Data in Principal Component Space",
                scene=dict(
                    xaxis_title="First Principal Component",
                    yaxis_title="Second Principal Component",
                    zaxis_title="Third Principal Component"
                )
            )
        
        st.plotly_chart(fig_transform)
    
    # Display reconstruction information
    if st.checkbox("Show Data Reconstruction"):
        # Reconstruct the data
        data_reconstructed = pca.inverse_transform(data_transformed)
        data_reconstructed = scaler.inverse_transform(data_reconstructed)
        
        # Compute reconstruction error
        mse = np.mean((data - data_reconstructed) ** 2)
        st.markdown(f"### Reconstruction Error (MSE): {mse:.4f}")
        
        # Show sample reconstructions
        n_samples = min(5, len(data))
        st.markdown("### Sample Reconstructions")
        
        for i in range(n_samples):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Original Sample {i+1}**")
                st.write(data[i])
            with col2:
                st.markdown(f"**Reconstructed Sample {i+1}**")
                st.write(data_reconstructed[i]) 