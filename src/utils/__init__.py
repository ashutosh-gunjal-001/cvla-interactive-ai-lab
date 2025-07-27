# Utils module for CVLA Interactive AI Lab 

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def get_figure_size(default_size=None):
    """
    Get the appropriate figure size based on user preferences.
    
    Parameters:
    -----------
    default_size : tuple, optional
        Default figure size (width, height) if no specific size is set
        
    Returns:
    --------
    tuple
        Figure size as (width, height)
    """
    # Get the current visualization type
    viz_type = st.session_state.get("current_visualization", "default")
    
    # Define size mappings based on preference
    size_mappings = {
        "small": {
            "default": (6, 4),
            "wide": (8, 3),
            "complex_function": (5, 5),
            "harmonic_flow_wide": (8, 3),
            "conformal_map_wide": (8, 3),
            "integral_contour": (5, 5),
        },
        "medium": {
            "default": (8, 6),
            "wide": (10, 4),
            "complex_function": (7, 7),
            "harmonic_flow_wide": (10, 4),
            "conformal_map_wide": (10, 4),
            "integral_contour": (7, 7),
        },
        "large": {
            "default": (10, 8),
            "wide": (12, 5),
            "complex_function": (9, 9),
            "harmonic_flow_wide": (12, 5),
            "conformal_map_wide": (12, 5),
            "integral_contour": (9, 9),
        }
    }
    
    # Get user's preference from session state
    size_preference = st.session_state.get("visualization_size", "medium")
    
    # Get appropriate size based on visualization type and preference
    if viz_type in size_mappings[size_preference]:
        return size_mappings[size_preference][viz_type]
    else:
        # Return default for the preference
        if default_size:
            return default_size
        return size_mappings[size_preference]["default"]

def apply_image_container(fig, caption=None, max_height=None):
    """
    Apply a container to a matplotlib figure and display it with a caption.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to display
    caption : str, optional
        Caption text to display below the figure
    max_height : int, optional
        Maximum height for the image in pixels
    """
    # Create container for the figure
    with st.container():
        # Display the figure with the proper styling
        st.pyplot(fig)
        
        # Add caption if provided
        if caption:
            st.markdown(f"<div class='image-caption'>{caption}</div>", unsafe_allow_html=True)
    
    # Close the figure to free memory
    plt.close(fig)

def create_3d_toggle():
    """
    Create a toggle button for switching between 2D and 3D visualizations.
    
    Returns:
    --------
    bool
        True if 3D visualization is selected, False otherwise
    """
    # Create a container for the toggle
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Create the toggle switch
            show_3d = st.checkbox("Show 3D visualization", value=False, key=f"3d_toggle_{st.session_state.get('current_visualization', 'default')}")
            
        with col1:
            if show_3d:
                st.info("3D visualization will allow you to explore the function's magnitude as a surface. "
                        "Drag to rotate, scroll to zoom, and double-click to reset view.")
    
    return show_3d

def load_plotly_requirements():
    """
    Check if plotly is installed and provide installation instructions if needed.
    """
    try:
        import plotly
        return True
    except ImportError:
        st.error("Plotly is required for 3D visualizations but is not installed.")
        st.code("pip install plotly", language="bash")
        return False 