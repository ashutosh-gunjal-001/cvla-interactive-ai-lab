import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

def embedding_comparator(data, embedding_type='pca', metric='euclidean'):
    """
    Compare different embeddings of data in various spaces.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data matrix (samples x features)
    embedding_type : str
        Type of embedding ('pca', 'tsne', 'mds')
    metric : str
        Distance metric for MDS ('euclidean', 'cosine', etc.)
    """
    # Ensure data is 2D
    if len(data.shape) != 2:
        st.error("Data must be a 2D array")
        return
    
    n_samples, n_features = data.shape
    
    # Compute original distances
    original_distances = squareform(pdist(data, metric))
    
    # Perform embedding
    if embedding_type == 'pca':
        embedder = PCA(n_components=2)
        embedded = embedder.fit_transform(data)
        explained_variance = embedder.explained_variance_ratio_
    elif embedding_type == 'tsne':
        embedder = TSNE(n_components=2, perplexity=min(30, n_samples-1))
        embedded = embedder.fit_transform(data)
        explained_variance = None
    else:  # mds
        embedder = MDS(n_components=2, dissimilarity='precomputed')
        embedded = embedder.fit_transform(original_distances)
        explained_variance = None
    
    # Compute embedded distances
    embedded_distances = squareform(pdist(embedded, metric))
    
    # Display embedding information
    st.subheader("Embedding Information")
    st.write(f"Original dimension: {n_features}")
    st.write(f"Embedded dimension: 2")
    
    if explained_variance is not None:
        st.write("Explained variance ratio:")
        for i, ratio in enumerate(explained_variance):
            st.latex(f"PC_{i+1}: {ratio:.4f}")
    
    # Visualize original data (if 2D or 3D)
    if n_features <= 3:
        fig_original = go.Figure()
        
        if n_features == 2:
            fig_original.add_trace(go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                mode='markers',
                name='Data Points'
            ))
        else:  # 3D
            fig_original.add_trace(go.Scatter3d(
                x=data[:, 0],
                y=data[:, 1],
                z=data[:, 2],
                mode='markers',
                name='Data Points'
            ))
        
        fig_original.update_layout(
            title='Original Data',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Feature 3' if n_features == 3 else None
        )
        st.plotly_chart(fig_original)
    
    # Visualize embedded data
    fig_embedded = go.Figure()
    fig_embedded.add_trace(go.Scatter(
        x=embedded[:, 0],
        y=embedded[:, 1],
        mode='markers',
        name='Embedded Points'
    ))
    fig_embedded.update_layout(
        title=f'{embedding_type.upper()} Embedding',
        xaxis_title='Component 1',
        yaxis_title='Component 2'
    )
    st.plotly_chart(fig_embedded)
    
    # Compare distances
    st.subheader("Distance Preservation Analysis")
    
    # Compute distance preservation metrics
    original_dist_flat = original_distances[np.triu_indices(n_samples, k=1)]
    embedded_dist_flat = embedded_distances[np.triu_indices(n_samples, k=1)]
    
    # Compute correlation between original and embedded distances
    correlation = np.corrcoef(original_dist_flat, embedded_dist_flat)[0, 1]
    st.latex(f"\\text{{Distance Correlation}} = {correlation:.4f}")
    
    # Compute stress (for MDS)
    if embedding_type == 'mds':
        stress = np.sum((original_dist_flat - embedded_dist_flat)**2) / np.sum(original_dist_flat**2)
        st.latex(f"\\text{{Stress}} = {stress:.4f}")
    
    # Display distance matrices
    st.subheader("Distance Matrices")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Original Distances")
        st.write(original_distances)
    
    with col2:
        st.write("Embedded Distances")
        st.write(embedded_distances)
    
    # Display distance preservation plot
    fig_preservation = go.Figure()
    fig_preservation.add_trace(go.Scatter(
        x=original_dist_flat,
        y=embedded_dist_flat,
        mode='markers',
        name='Distance Pairs'
    ))
    
    # Add line of perfect preservation
    max_dist = max(np.max(original_dist_flat), np.max(embedded_dist_flat))
    fig_preservation.add_trace(go.Scatter(
        x=[0, max_dist],
        y=[0, max_dist],
        mode='lines',
        name='Perfect Preservation',
        line=dict(color='red', dash='dash')
    ))
    
    fig_preservation.update_layout(
        title='Distance Preservation',
        xaxis_title='Original Distance',
        yaxis_title='Embedded Distance'
    )
    st.plotly_chart(fig_preservation) 