import streamlit as st
import numpy as np
from scipy.linalg import null_space, orth
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import model components with fallbacks if not available
try:
    from models.matrix_transformation_visualizer import visualize_matrix_transform
except ImportError:
    def visualize_matrix_transform(matrix, vector, show_basis):
        st.error("Matrix Transformation Visualizer module not found. Please check your installation.")
        st.code("Error importing: models.matrix_transformation_visualizer")

try:
    from models.basis_classifier import classify_basis
except ImportError:
    def classify_basis(matrix):
        st.error("Basis Classifier module not found. Please check your installation.")
        st.code("Error importing: models.basis_classifier")

try:
    from models.subspace_explorer import explore_subspace
except ImportError:
    def explore_subspace(matrix, vector):
        st.error("Subspace Explorer module not found. Please check your installation.")
        st.code("Error importing: models.subspace_explorer")

try:
    from models.inversion_animation import animate_inversion
except ImportError:
    def animate_inversion(matrix):
        st.error("Inversion Animation Engine module not found. Please check your installation.")
        st.code("Error importing: models.inversion_animation")

# Available models dictionary with descriptions
MATRIX_MODELS = {
    "Matrix Transformation Visualizer": {
        "description": "Visualize how matrices transform vectors and spaces in 2D and 3D",
        "examples": {
            "Rotation": "[[0, -1], [1, 0]]",
            "Scaling": "[[2, 0], [0, 0.5]]",
            "Shear": "[[1, 1], [0, 1]]",
            "Reflection": "[[1, 0], [0, -1]]"
        },
        "default": "[[1, 0], [0, 1]]"
    },
    "Basis Classifier": {
        "description": "Analyze and classify matrix bases, including orthogonality and independence",
        "examples": {
            "Orthogonal": "[[1, 0], [0, 1]]",
            "Dependent": "[[1, 2], [2, 4]]",
            "Independent": "[[1, 1], [1, -1]]"
        },
        "default": "[[1, 0], [0, 1]]"
    },
    "Subspace Explorer": {
        "description": "Explore column space, null space, and other subspaces of matrices",
        "examples": {
            "Full Rank": "[[1, 0], [0, 1]]",
            "Rank Deficient": "[[1, 2], [2, 4]]",
            "3D Subspace": "[[1, 0, 0], [0, 1, 0], [0, 0, 0]]"
        },
        "default": "[[1, 0], [0, 1]]"
    },
    "Inversion Animation Engine": {
        "description": "Visualize matrix inversion and its geometric interpretation",
        "examples": {
            "2x2 Invertible": "[[1, 2], [3, 4]]",
            "3x3 Invertible": "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
            "Singular": "[[1, 1], [1, 1]]"
        },
        "default": "[[1, 0], [0, 1]]"
    }
}

def parse_matrix(matrix_str):
    """Parse matrix string into numpy array"""
    try:
        # Remove any whitespace and split into rows
        matrix_str = matrix_str.strip()
        rows = matrix_str.split('],')
        matrix = []
        for row in rows:
            # Remove brackets and split into elements
            row = row.replace('[', '').replace(']', '').strip()
            elements = [float(x.strip()) for x in row.split(',')]
            matrix.append(elements)
        return np.array(matrix)
    except Exception as e:
        st.error(f"Error parsing matrix: {str(e)}")
        return None

def display():
    """Main function to display Matrixland content"""
    st.markdown("<h2 class='basket-header'>Matrixland: Linear Algebra Visualizer</h2>", unsafe_allow_html=True)
    st.markdown("""
    Explore matrix transformations, basis classification, subspace analysis, and matrix inversion.
    Select a model below and enter your matrix to see it in action.
    """)
    
    # Initialize session state for selected model if not exists
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "Matrix Transformation Visualizer"
    
    # Model selection
    st.markdown("### Select a Matrix Model")
    
    # Use tabs for better model selection UI
    tabs = st.tabs(list(MATRIX_MODELS.keys()))
    
    # Update selected model based on tab selection
    for i, (model_name, model_info) in enumerate(MATRIX_MODELS.items()):
        with tabs[i]:
            st.markdown(f"**{model_name}**")
            st.markdown(model_info["description"])
            st.markdown("Examples:")
            for name, example in model_info["examples"].items():
                st.code(f"{name}: {example}")
            
            if st.button(f"Select {model_name}", key=f"select_btn_{model_name}"):
                st.session_state["selected_model"] = model_name
                st.rerun()
    
    # Show selected model with highlight
    selected_model = st.session_state.get("selected_model", "Matrix Transformation Visualizer")
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: #EEF2FF; border-left: 5px solid #4F46E5;">
        <h3 style="margin: 0; color: #1E3A8A;">Selected: {selected_model}</h3>
        <p style="margin: 5px 0 0 0; color: #4B5563;">{MATRIX_MODELS[selected_model]['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Matrix input section
    st.markdown("---")
    st.markdown("### Matrix Input")
    
    model_info = MATRIX_MODELS[selected_model]
    
    # Show examples as chips
    st.markdown("##### Examples")
    example_cols = st.columns(len(model_info["examples"]))
    for i, (name, example) in enumerate(model_info["examples"].items()):
        if example_cols[i].button(name, key=f"example_{i}"):
            st.session_state["matrix_input"] = example
    
    # Matrix input
    with st.form(key="matrix_form"):
        matrix_input = st.text_area(
            "Enter Matrix (use [[a, b], [c, d]] format):",
            value=st.session_state.get("matrix_input", model_info["default"]),
            help="Enter a matrix in Python list format"
        )
        
        # Additional parameters based on model
        if selected_model == "Matrix Transformation Visualizer":
            vector_input = st.text_input(
                "Enter Vector (optional, use [x, y] format):",
                value="[1, 1]",
                help="Enter a vector to transform"
            )
            show_basis = st.checkbox("Show Basis Vectors", value=True)
        
        # Submit button
        submitted = st.form_submit_button("Visualize", type="primary")
        
        if submitted:
            try:
                st.session_state["matrix_input"] = matrix_input
                matrix = parse_matrix(matrix_input)
                
                if matrix is None:
                    st.error("Invalid matrix format")
                    return
                
                # Call the appropriate visualization function based on selected model
                if selected_model == "Matrix Transformation Visualizer":
                    vector = parse_matrix(vector_input)
                    if vector is None:
                        st.error("Invalid vector format")
                        return
                    with st.spinner("Generating transformation visualization..."):
                        visualize_matrix_transform(matrix, vector, show_basis)
                
                elif selected_model == "Basis Classifier":
                    with st.spinner("Analyzing matrix basis..."):
                        classify_basis(matrix)
                
                elif selected_model == "Subspace Explorer":
                    vector = parse_matrix(vector_input) if "vector_input" in locals() else None
                    with st.spinner("Exploring subspaces..."):
                        explore_subspace(matrix, vector)
                
                elif selected_model == "Inversion Animation Engine":
                    with st.spinner("Generating inversion animation..."):
                        animate_inversion(matrix)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    
    # Educational content
    with st.expander("Learn about Linear Algebra"):
        st.markdown("""
        ## Linear Algebra Concepts
        
        ### Matrix Transformations
        
        Matrices represent linear transformations that can rotate, scale, shear, or reflect vectors.
        The determinant of a matrix tells us about the scaling factor of the transformation.
        
        ### Basis and Dimension
        
        A basis is a set of linearly independent vectors that span a vector space.
        The dimension of a space is the number of vectors in any basis for that space.
        
        ### Subspaces
        
        Important subspaces include:
        - Column space: All possible linear combinations of the columns
        - Null space: All vectors that map to zero
        - Row space: All possible linear combinations of the rows
        
        ### Matrix Inversion
        
        A matrix is invertible if and only if its determinant is non-zero.
        The inverse matrix undoes the transformation of the original matrix.
        """) 