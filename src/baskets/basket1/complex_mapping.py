import streamlit as st
import numpy as np

# Try to import model components with fallbacks if not available
try:
    from models.complex_function_grapher import visualize_complex_function
except ImportError:
    def visualize_complex_function(func_str, domain_min, domain_max, resolution, show_colorwheel):
        st.error("Complex Function Grapher module not found. Please check your installation.")
        st.code("Error importing: models.complex_function_grapher")

try:
    from models.conformal_map_animator import visualize_conformal_map
except ImportError:
    def visualize_conformal_map(func_str, domain_min, domain_max, resolution):
        st.error("Conformal Map Animator module not found. Please check your installation.")
        st.code("Error importing: models.conformal_map_animator")

try:
    from models.harmonic_flow_predictor import visualize_harmonic_flow
except ImportError:
    def visualize_harmonic_flow(func_str, domain_min, domain_max, resolution):
        st.error("Harmonic Flow Predictor module not found. Please check your installation.")
        st.code("Error importing: models.harmonic_flow_predictor")

try:
    from models.integral_contour_interpreter import visualize_integral_contour
except ImportError:
    def visualize_integral_contour(func_str, domain_min, domain_max, resolution):
        st.error("Integral Contour Interpreter module not found. Please check your installation.")
        st.code("Error importing: models.integral_contour_interpreter")

# Available models dictionary with descriptions
COMPLEX_MODELS = {
    "Analytic Functions": {
        "description": "Visualize analytic functions with domain coloring to explore their behavior",
        "examples": ["z**2", "z**3 - 1", "sin(z)", "exp(z)", "1/(z**2 + 1)", "z * log(z)"],
        "default": "z**2"
    },
    "Cauchy-Riemann Equations": {
        "description": "Visualize conformal mappings to understand the Cauchy-Riemann equations",
        "examples": ["1/z", "z**2", "exp(z)", "z + 1/z", "sin(z)", "(z+1)/(z-1)"],
        "default": "1/z"
    },
    "Harmonic Functions": {
        "description": "Visualize harmonic functions and their flow fields",
        "examples": ["log(z)", "z**2", "1/z", "z*exp(-z**2)"],
        "default": "log(z)"
    },
    "Complex Integration": {
        "description": "Compute and visualize contour integrals in the complex plane",
        "examples": ["z**2-1", "1/z", "exp(z)", "sin(z)/z", "z**2/(z**2+1)", "1/(z-1)"],
        "default": "1/(z-1)"
    }
}

def display():
    """Main function to display Complex Mapping basket content"""
    st.markdown("<h2 class='basket-header'>Complex Mapping & Analytic Function Simulator</h2>", unsafe_allow_html=True)
    st.markdown("""
    Explore complex functions through interactive visualizations using the models below.
    Learn about key concepts in complex analysis in the educational section.
    """)

    # Educational content
    st.markdown("## Key Concepts in Complex Analysis")
    with st.expander("Learn about Complex Functions", expanded=True):
        st.markdown("""
        ### Analytic Functions
        Analytic functions satisfy the Cauchy-Riemann equations and are infinitely differentiable.
        Their conformal mappings preserve angles between curves.

        ### Cauchy-Riemann Equations
        These equations ensure a function is complex differentiable, a requirement for analyticity.
        They relate the partial derivatives of the real and imaginary parts of a function.

        ### Harmonic Functions
        Harmonic functions satisfy Laplace's equation. The real and imaginary parts of any
        analytic function are harmonic functions.

        ### Complex Integration
        Complex integration along contours is a powerful technique with applications in
        physics, engineering, and mathematics.
        """)

    # Initialize session state for selected model if not exists
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "Analytic Functions"

    # Model selection
    st.markdown("### Select a Visualization Model")

    # Use tabs for model selection UI
    tabs = st.tabs(list(COMPLEX_MODELS.keys()))

    # Update selected model based on tab selection
    for i, (model_name, model_info) in enumerate(COMPLEX_MODELS.items()):
        with tabs[i]:
            st.markdown(f"**{model_name}**")
            st.markdown(model_info["description"])
            st.markdown(f"Example: `{model_info['default']}`")

            if st.button(f"Select {model_name}", key=f"select_btn_{model_name}"):
                st.session_state["selected_model"] = model_name
                st.rerun()

    # Show selected model with highlight
    selected_model = st.session_state.get("selected_model", "Analytic Functions")
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: #EEF2FF; border-left: 5px solid #4F46E5;">
        <h3 style="margin: 0; color: #1E3A8A;">Selected: {selected_model}</h3>
        <p style="margin: 5px 0 0 0; color: #4B5563;">{COMPLEX_MODELS[selected_model]['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Function input section
    st.markdown("---")
    st.markdown("### Function Input")

    model_info = COMPLEX_MODELS[selected_model]

    # Show examples as chips
    st.markdown("##### Examples")
    example_cols = st.columns(4)
    for i, example in enumerate(model_info["examples"]):
        if example_cols[i % 4].button(example, key=f"example_{i}"):
            st.session_state["function_input"] = example

    # Function input
    with st.form(key="function_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            function_input = st.text_input(
                "f(z) = ",
                value=st.session_state.get("function_input", model_info["default"]),
                help="Enter a complex function using z as the variable"
            )

        # Parameter settings
        col1, col2, col3 = st.columns(3)
        with col1:
            domain_min = st.slider("Domain Minimum", -10.0, -0.1, -5.0)
        with col2:
            domain_max = st.slider("Domain Maximum", 0.1, 10.0, 5.0)
        with col3:
            resolution = st.slider("Resolution", 100, 1000, 500, 100)

        # Additional options
        show_colorwheel = st.checkbox("Show Color Legend", value=True)

        # Submit button
        submitted = st.form_submit_button("Visualize", type="primary")

        if submitted:
            try:
                st.session_state["function_input"] = function_input

                # Call the appropriate visualization function based on selected model
                if selected_model == "Analytic Functions":
                    with st.spinner("Generating domain coloring visualization..."):
                        visualize_complex_function(
                            function_input,
                            domain_min,
                            domain_max,
                            resolution,
                            show_colorwheel
                        )
                elif selected_model == "Cauchy-Riemann Equations":
                    with st.spinner("Generating conformal map animation..."):
                        visualize_conformal_map(
                            function_input,
                            domain_min,
                            domain_max,
                            resolution
                        )
                elif selected_model == "Harmonic Functions":
                    with st.spinner("Generating harmonic flow visualization..."):
                        visualize_harmonic_flow(
                            function_input,
                            domain_min,
                            domain_max,
                            resolution
                        )
                elif selected_model == "Complex Integration":
                    with st.spinner("Generating contour integral visualization..."):
                        visualize_integral_contour(
                            function_input,
                            domain_min,
                            domain_max,
                            resolution
                        )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)