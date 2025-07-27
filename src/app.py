import streamlit as st
import numpy as np
from PIL import Image
import os

# Import basket modules
from baskets.basket1.complex_mapping import display as complex_mapping_display
from models.subspace_explorer import explore_subspace
from models.inversion_animation import animate_inversion
from models.eigen_motion import simulate_eigen_motion
from models.pca_explorer import explore_pca
from models.equation_solver import solve_equation, solve_system
from models.cayley_hamilton import check_cayley_hamilton
from models.inner_product_intuition import inner_product_intuition
from models.gram_schmidt_animator import gram_schmidt_animator
from models.embedding_comparator import embedding_comparator
from models.orthonormal_basis import orthonormal_basis_generator
from utils.sidebar import create_sidebar

# Set page configuration
st.set_page_config(
    page_title="CVLA Interactive AI Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def load_css():
    css = """
    <style>
        /* Base theme */
        :root {
            --neon-blue: #00f3ff;
            --neon-purple: #ff00ff;
            --neon-green: #00ff00;
            --neon-yellow: #ffff00;
            --dark-bg: #0a0a0a;
            --darker-bg: #030303;
            --text-primary: #ffffff;
            --text-secondary: #ECEFCA;
            --text-tertiary: #7AE2CF;
            --text-inactive: #94B4C1;
        }
        
        /* Main background */
        .stApp {
            background-color: var(--dark-bg);
            color: var(--text-primary);
        }
        
        /* Grid background */
        .grid-background {
            background-color: var(--darker-bg);
            color: var(--text-tertiary);
            background-image: 
            linear-gradient(to right, rgba(0, 243, 255, 0.1) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(0, 243, 255, 0.1) 1px, transparent 1px);
            background-size: 20px 20px;
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
            border: 2px solid rgba(0, 243, 255, 0.2);
            box-shadow: 0 0 10px rgba(0, 243, 255, 0.1);
        }
        
        /* Headers */
        .main-header {
            font-size: 2.5rem;
            color: var(--text-tertiary);
            font-weight: 800;
            text-shadow: 0 0 10px var(--text-tertiary);
            letter-spacing: 2px;
        }
        
        .basket-header {
            font-size: 1.8rem;
            color: var(--neon-blue);
            font-weight: 600;
            margin-top: 1rem;
            text-shadow: 0 0 8px var(--neon-blue);
            letter-spacing: 1px;
        }
        
        .subheader {
            font-size: 1.2rem;
            color: var(--text-tertiary);
            font-weight: 400;
            letter-spacing: 1px;
            text-shadow: 0 0 5px var(--text-tertiary);
        }
        
        /* Ensure h1 in grid-background uses the correct color */
        .grid-background h1 {
            color: var(--text-tertiary);
            text-shadow: 0 0 10px var(--text-tertiary);
        }
        
        /* Cards */
        .model-card {
            border: 2px solid rgba(0, 243, 255, 0.3);
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
            background-color: var(--darker-bg);
            transition: all 0.3s;
            box-shadow: 0 0 15px rgba(0, 243, 255, 0.1);
            color: var(--text-tertiary);
        }
        
        .model-card:hover {
            border-color: var(--neon-blue);
            box-shadow: 0 0 20px var(--neon-blue);
            transform: translateY(-2px);
            color: var(--text-primary);
        }
        
        .selected-model {
            border: 2px solid var(--neon-purple);
            background-color: rgba(255, 0, 255, 0.1);
            box-shadow: 0 0 20px var(--neon-purple);
            color: var(--text-primary);
        }
        
        /* Tabs */
        div[data-testid="stTabs"] > div > div[role="tablist"] button[aria-selected="true"] {
            background-color: var(--darker-bg);
            border-bottom-color: var(--neon-blue);
            color: var(--neon-yellow);
            font-weight: bold;
            text-shadow: 0 0 5px var(--neon-yellow);
        }
        
        div[data-testid="stTabs"] > div > div[role="tablist"] > button {
            min-width: 120px;
            transition: all 0.3s;
            color: var(--text-secondary);
            background-color: var(--dark-bg);
            border: 2px solid rgba(0, 243, 255, 0.2);
        }
        
        div[data-testid="stTabs"] > div > div[role="tablist"] > button:hover {
            background-color: var(--darker-bg);
            color: var(--neon-yellow);
            border-color: var(--neon-blue);
            box-shadow: 0 0 10px var(--neon-blue);
        }
        
        /* Buttons */
        button[kind="primary"] {
            background-color: var(--darker-bg);
            border: 2px solid var(--neon-blue);
            color: var(--neon-yellow);
            padding: 0.5rem 1rem;
            transition: all 0.3s;
            text-shadow: 0 0 5px var(--neon-yellow);
        }
        
        button[kind="primary"]:hover {
            background-color: rgba(0, 243, 255, 0.1);
            color: var(--neon-yellow);
            box-shadow: 0 0 15px var(--neon-blue);
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {
            background-color: var(--darker-bg);
            border: 2px solid rgba(0, 243, 255, 0.3);
            color: var(--text-primary);
        }
        
        /* Sliders */
        .stSlider > div > div > div {
            background-color: var(--darker-bg);
        }
        
        .stSlider > div > div > div > div {
            background-color: var(--neon-blue);
            box-shadow: 0 0 5px var(--neon-blue);
        }
        
        /* Plot containers */
        .element-container {
            background-color: var(--darker-bg);
            border: 2px solid rgba(0, 243, 255, 0.2);
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #213448;
            border-right: 2px solid rgba(0, 243, 255, 0.2);
        }
        
        /* Markdown text */
        .stMarkdown {
            color: var(--text-secondary);
        }
        
        /* Code blocks */
        pre {
            background-color: var(--darker-bg);
            border: 2px solid rgba(0, 243, 255, 0.2);
            color: var(--neon-green);
        }
        
        /* General text improvements */
        p, div, span {
            color: var(--text-secondary);
            text-shadow: 0 0 2px rgba(255, 255, 255, 0.2);
        }
        
        /* Form labels */
        label {
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        /* Select boxes */
        .stSelectbox > div > div > div {
            color: var(--text-primary);
        }
        
        /* Checkbox labels */
        .stCheckbox > label {
            color: var(--text-secondary);
        }
        
        /* Basket buttons */
        .stButton > button {
            color: var(--text-inactive);
            background-color: #213448;
        }
        
        .stButton > button[kind="primary"] {
            color: var(--text-secondary);
            background-color: #213448;
        }
        
        .stButton > button:hover {
            background-color: #2a4258;
        }
        
        /* Expander sections */
        .streamlit-expanderHeader {
            background-color: #547792;
            color: #547792;
        }
        
        .streamlit-expanderContent {
            background-color: #547792;
            color: var(--darker-bg);
        }
        
        /* Sidebar expander */
        .css-1d391kg .streamlit-expanderHeader {
            background-color: #547792;
            color: var(--darker-bg);
        }
        
        .css-1d391kg .streamlit-expanderContent {
            background-color: #547792;
            color: var(--darker-bg);
        }
        
        /* About Image Controls expander */
        div[data-testid="stExpander"]:has(div:contains("About Image Controls")) .streamlit-expanderHeader {
            background-color: #547792;
            color: #547792;
        }
        
        div[data-testid="stExpander"]:has(div:contains("About Image Controls")) .streamlit-expanderContent {
            background-color: #547792;
            color: var(--darker-bg);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def main():
    # Load custom CSS
    load_css()
    
    # Create sidebar with navigation
    selected_basket = create_sidebar()
    
    
    # Main header
    st.markdown("<h1 class='grid-background'>CVLA Interactive AI Lab</h1>", unsafe_allow_html=True)
    # st.markdown('<div class="grid-background">Your content here</div>', unsafe_allow_html=True)   
    st.markdown("<p class='subheader'>An AI-powered interactive platform for exploring Complex Variables & Linear Algebra visually</p>", unsafe_allow_html=True)
    # Set default max height for visualizations
    if "max_viz_height" not in st.session_state:
        st.session_state["max_viz_height"] = 400
    
    # Show different content based on selected basket
    if selected_basket == "Complex Mapping":
        complex_mapping_display()
    
    elif selected_basket == "Matrixland":
        st.markdown("<h2 class='basket-header'>Matrixland & Vector Playground</h2>", unsafe_allow_html=True)
        
        # Create tabs for different matrix operations
        tab1, tab2 = st.tabs(["Subspace Explorer", "Matrix Inversion"])
        
        with tab1:
            st.markdown("<h3 class='basket-header'>Subspace Explorer</h3>", unsafe_allow_html=True)
            # Create a 2x2 or 3x3 matrix input
            matrix_size = st.selectbox("Select Matrix Size", ["2x2", "3x3"])
            
            if matrix_size == "2x2":
                col1, col2 = st.columns(2)
                with col1:
                    a = st.number_input("a", value=1.0)
                    c = st.number_input("c", value=0.0)
                with col2:
                    b = st.number_input("b", value=0.0)
                    d = st.number_input("d", value=1.0)
                matrix = np.array([[a, b], [c, d]])
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    a = st.number_input("a", value=1.0)
                    d = st.number_input("d", value=0.0)
                    g = st.number_input("g", value=0.0)
                with col2:
                    b = st.number_input("b", value=0.0)
                    e = st.number_input("e", value=1.0)
                    h = st.number_input("h", value=0.0)
                with col3:
                    c = st.number_input("c", value=0.0)
                    f = st.number_input("f", value=0.0)
                    i = st.number_input("i", value=1.0)
                matrix = np.array([[a, b, c], [d, e, f], [g, h, i]])
            
            # Optional vector input
            if st.checkbox("Include Vector", value=False):
                if matrix_size == "2x2":
                    v1 = st.number_input("Vector x", value=1.0)
                    v2 = st.number_input("Vector y", value=1.0)
                    vector = np.array([v1, v2])
                else:
                    v1 = st.number_input("Vector x", value=1.0)
                    v2 = st.number_input("Vector y", value=1.0)
                    v3 = st.number_input("Vector z", value=1.0)
                    vector = np.array([v1, v2, v3])
            else:
                vector = None
            
            if st.button("Explore Subspace"):
                explore_subspace(matrix, vector)
        
        with tab2:
            st.markdown("<h3 class='basket-header'>Matrix Inversion</h3>", unsafe_allow_html=True)
            # Create a 2x2 or 3x3 matrix input
            matrix_size = st.selectbox("Select Matrix Size", ["2x2", "3x3"], key="inversion_size")
            
            if matrix_size == "2x2":
                col1, col2 = st.columns(2)
                with col1:
                    a = st.number_input("a", value=1.0, key="inv_a")
                    c = st.number_input("c", value=0.0, key="inv_c")
                with col2:
                    b = st.number_input("b", value=0.0, key="inv_b")
                    d = st.number_input("d", value=1.0, key="inv_d")
                matrix = np.array([[a, b], [c, d]])
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    a = st.number_input("a", value=1.0, key="inv_a")
                    d = st.number_input("d", value=0.0, key="inv_d")
                    g = st.number_input("g", value=0.0, key="inv_g")
                with col2:
                    b = st.number_input("b", value=0.0, key="inv_b")
                    e = st.number_input("e", value=1.0, key="inv_e")
                    h = st.number_input("h", value=0.0, key="inv_h")
                with col3:
                    c = st.number_input("c", value=0.0, key="inv_c")
                    f = st.number_input("f", value=0.0, key="inv_f")
                    i = st.number_input("i", value=1.0, key="inv_i")
                matrix = np.array([[a, b, c], [d, e, f], [g, h, i]])
            
            if st.button("Animate Inversion"):
                animate_inversion(matrix)
    
    elif selected_basket == "Eigen Exploratorium":
        st.markdown("<h2 class='basket-header'>Eigen Exploratorium</h2>", unsafe_allow_html=True)
        
        # Create tabs for different eigen-related operations
        tab1, tab2, tab3, tab4 = st.tabs([
            "EigenMotion Simulator",
            "PCA Explorer",
            "Equation Solver AI",
            "Cayley-Hamilton Checker"
        ])
        
        with tab1:
            st.markdown("<h3 class='basket-header'>EigenMotion Simulator</h3>", unsafe_allow_html=True)
            # Create a 2x2 or 3x3 matrix input
            matrix_size = st.selectbox("Select Matrix Size", ["2x2", "3x3"], key="eigen_size")
            
            if matrix_size == "2x2":
                col1, col2 = st.columns(2)
                with col1:
                    a = st.number_input("a", value=1.0, key="eigen_a")
                    c = st.number_input("c", value=0.0, key="eigen_c")
                with col2:
                    b = st.number_input("b", value=0.0, key="eigen_b")
                    d = st.number_input("d", value=1.0, key="eigen_d")
                matrix = np.array([[a, b], [c, d]])
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    a = st.number_input("a", value=1.0, key="eigen_a")
                    d = st.number_input("d", value=0.0, key="eigen_d")
                    g = st.number_input("g", value=0.0, key="eigen_g")
                with col2:
                    b = st.number_input("b", value=0.0, key="eigen_b")
                    e = st.number_input("e", value=1.0, key="eigen_e")
                    h = st.number_input("h", value=0.0, key="eigen_h")
                with col3:
                    c = st.number_input("c", value=0.0, key="eigen_c")
                    f = st.number_input("f", value=0.0, key="eigen_f")
                    i = st.number_input("i", value=1.0, key="eigen_i")
                matrix = np.array([[a, b, c], [d, e, f], [g, h, i]])
            
            if st.button("Simulate Eigen Motion"):
                simulate_eigen_motion(matrix)
        
        with tab2:
            st.markdown("<h3 class='basket-header'>PCA Explorer</h3>", unsafe_allow_html=True)
            
            # Sample data generation
            st.markdown("#### Generate Sample Data")
            n_samples = st.slider("Number of samples", 10, 1000, 100)
            n_features = st.slider("Number of features", 2, 10, 3)
            
            if st.button("Generate Random Data"):
                # Generate random data with some structure
                true_dim = min(n_features - 1, 3)
                components = np.random.randn(true_dim, n_features)
                latent = np.random.randn(n_samples, true_dim)
                data = latent @ components
                data += 0.1 * np.random.randn(n_samples, n_features)  # Add noise
                
                explore_pca(data)
            
            # Upload own data
            st.markdown("#### Or Upload Your Own Data")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                data = np.loadtxt(uploaded_file, delimiter=",")
                explore_pca(data)
        
        with tab3:
            st.markdown("<h3 class='basket-header'>Equation Solver AI</h3>", unsafe_allow_html=True)
            
            # Choose between single equation and system
            eq_type = st.radio("Choose equation type:", ["Single Equation", "System of Equations"])
            
            if eq_type == "Single Equation":
                st.markdown("""
                Enter an equation in the form: `expression = expression`
                Examples:
                - x**2 + 2*x - 1 = 0
                - sin(x) = cos(x)
                - x**3 - 2*x**2 + x - 1 = x + 2
                """)
                
                equation = st.text_input("Enter equation:")
                if st.button("Solve Equation"):
                    solve_equation(equation)
            
            else:  # System of Equations
                st.markdown("""
                Enter equations one per line in the form: `expression = expression`
                Examples:
                x + y = 1
                x - y = 3
                """)
                
                equations = st.text_area("Enter system of equations:")
                if st.button("Solve System"):
                    equations = [eq.strip() for eq in equations.split("\n") if eq.strip()]
                    solve_system(equations)
        
        with tab4:
            st.markdown("<h3 class='basket-header'>Cayley-Hamilton Checker</h3>", unsafe_allow_html=True)
            # Create a 2x2 or 3x3 matrix input
            matrix_size = st.selectbox("Select Matrix Size", ["2x2", "3x3"], key="ch_size")
            
            if matrix_size == "2x2":
                col1, col2 = st.columns(2)
                with col1:
                    a = st.number_input("a", value=1.0, key="ch_a")
                    c = st.number_input("c", value=0.0, key="ch_c")
                with col2:
                    b = st.number_input("b", value=0.0, key="ch_b")
                    d = st.number_input("d", value=1.0, key="ch_d")
                matrix = np.array([[a, b], [c, d]])
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    a = st.number_input("a", value=1.0, key="ch_a")
                    d = st.number_input("d", value=0.0, key="ch_d")
                    g = st.number_input("g", value=0.0, key="ch_g")
                with col2:
                    b = st.number_input("b", value=0.0, key="ch_b")
                    e = st.number_input("e", value=1.0, key="ch_e")
                    h = st.number_input("h", value=0.0, key="ch_h")
                with col3:
                    c = st.number_input("c", value=0.0, key="ch_c")
                    f = st.number_input("f", value=0.0, key="ch_f")
                    i = st.number_input("i", value=1.0, key="ch_i")
                matrix = np.array([[a, b, c], [d, e, f], [g, h, i]])
            
            if st.button("Check Cayley-Hamilton"):
                check_cayley_hamilton(matrix)
    
    elif selected_basket == "Inner Product Lab":
        st.title("Inner Product Lab")
        
        # Create tabs for different operations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Inner Product Intuition Machine",
            "Gram-Schmidt Animator",
            "Embedding Comparator",
            "Orthonormal Basis Generator"
        ])
        
        with tab1:
            st.header("Inner Product Intuition Machine")
            st.write("Visualize and understand inner products in different spaces.")
            
            # Space type
            space_type = st.radio("Space Type", ["Real", "Complex"])
            
            # Vector input
            vector_size = st.selectbox("Vector Size", ["2D", "3D"])
            if vector_size == "2D":
                col1, col2 = st.columns(2)
                with col1:
                    v1_x = st.number_input("v1_x", value=1.0)
                    v1_y = st.number_input("v1_y", value=0.0)
                    v1 = np.array([v1_x, v1_y])
                with col2:
                    v2_x = st.number_input("v2_x", value=0.0)
                    v2_y = st.number_input("v2_y", value=1.0)
                    v2 = np.array([v2_x, v2_y])
            else:
                col1, col2 = st.columns(2)
                with col1:
                    v1_x = st.number_input("v1_x", value=1.0)
                    v1_y = st.number_input("v1_y", value=0.0)
                    v1_z = st.number_input("v1_z", value=0.0)
                    v1 = np.array([v1_x, v1_y, v1_z])
                with col2:
                    v2_x = st.number_input("v2_x", value=0.0)
                    v2_y = st.number_input("v2_y", value=1.0)
                    v2_z = st.number_input("v2_z", value=0.0)
                    v2 = np.array([v2_x, v2_y, v2_z])
            
            # Explore inner product
            inner_product_intuition(v1, v2, space_type.lower())
        
        with tab2:
            st.header("Gram-Schmidt Animator")
            st.write("Visualize the Gram-Schmidt process step by step.")
            
            # Space type
            space_type = st.radio("Space Type", ["Real", "Complex"], key="gram_schmidt")
            
            # Number of vectors
            n_vectors = st.slider("Number of Vectors", 2, 3, 2)
            
            # Vector input
            vectors = []
            for i in range(n_vectors):
                st.subheader(f"Vector {i+1}")
                if n_vectors == 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.number_input(f"x_{i+1}", value=1.0 if i == 0 else 0.0)
                    with col2:
                        y = st.number_input(f"y_{i+1}", value=0.0 if i == 0 else 1.0)
                    vectors.append(np.array([x, y]))
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x = st.number_input(f"x_{i+1}", value=1.0 if i == 0 else 0.0)
                    with col2:
                        y = st.number_input(f"y_{i+1}", value=0.0 if i == 1 else 0.0)
                    with col3:
                        z = st.number_input(f"z_{i+1}", value=0.0 if i == 2 else 0.0)
                    vectors.append(np.array([x, y, z]))
            
            # Animate Gram-Schmidt process
            gram_schmidt_animator(vectors, space_type.lower())
        
        with tab3:
            st.header("Embedding Comparator")
            st.write("Compare different embeddings of data in various spaces.")
            
            # Data input options
            data_option = st.radio("Data Source", ["Generate Random Data", "Upload CSV"], key="embedding")
            
            if data_option == "Generate Random Data":
                n_samples = st.slider("Number of Samples", 10, 1000, 100, key="embedding_samples")
                n_features = st.slider("Number of Features", 2, 10, 5, key="embedding_features")
                data = np.random.randn(n_samples, n_features)
            else:
                uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key="embedding_file")
                if uploaded_file is not None:
                    data = np.loadtxt(uploaded_file, delimiter=',')
                else:
                    st.info("Please upload a CSV file")
                    return
            
            # Embedding type
            embedding_type = st.selectbox("Embedding Type", ["PCA", "t-SNE", "MDS"])
            
            # Distance metric (for MDS)
            if embedding_type == "MDS":
                metric = st.selectbox("Distance Metric", ["euclidean", "cosine", "cityblock"])
            else:
                metric = "euclidean"
            
            # Compare embeddings
            embedding_comparator(data, embedding_type.lower(), metric)
        
        with tab4:
            st.header("Orthonormal Basis Generator")
            st.write("Generate and verify orthonormal bases for different spaces.")
            
            # Space type
            space_type = st.radio("Space Type", ["Real", "Complex"], key="basis")
            
            # Dimension
            dimension = st.slider("Dimension", 2, 3, 2)
            
            # Generation method
            method = st.selectbox("Generation Method", ["QR", "SVD", "Random"])
            
            # Generate basis
            orthonormal_basis_generator(dimension, space_type.lower(), method.lower())

if __name__ == "__main__":
    main()