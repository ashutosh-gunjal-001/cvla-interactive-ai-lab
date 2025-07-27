import streamlit as st
import base64

def create_sidebar():
    with st.sidebar:
        st.markdown("""
        <h1 style='color: #7AE2CF; text-shadow: 0 0 5px #7AE2CF;'>🧠 CVLA Interactive AI Lab</h1>
        """, unsafe_allow_html=True)
        
        # Course Information in a styled box
        st.markdown("""
        <div style='
            background-color: #213448;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #e9ecef;
        '>
            <p style='font-size: 18px; font-weight: bold; color: #ECEFCA; margin-bottom: 8px;'>BMAT201L - Complex Variables and Linear Algebra</p>
            <p style='font-size: 16px; font-weight: bold; color: #ECEFCA; margin-bottom: 5px;'>Developed by:</p>
            <p style='font-size: 15px; color: #ECEFCA; margin: 0;'>• Divyanshu Patel (23BAI1214)</p>
            <p style='font-size: 15px; color: #ECEFCA; margin: 0;'>• Akshat Pal (23BRS1353)</p>
            <p style='font-size: 15px; color: #ECEFCA; margin: 0;'>• Ashutosh Gunjal (23BRS1354)</p>
            <p style='font-size: 16px; font-weight: bold; color: #ECEFCA; margin-top: 8px;'>Guided by: Dr. Vijay Kumar P.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a separator with gradient styling
        st.markdown("""
        <div style="background: linear-gradient(90deg, #4F46E5, #818CF8); height: 3px; margin: 10px 0 20px 0;"></div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for selected basket if not exists
        if "selected_basket" not in st.session_state:
            st.session_state["selected_basket"] = "Complex Mapping"
        
        # Basket selection
        st.markdown("### <span style='color: #ECEFCA;'>Choose a Basket</span>", unsafe_allow_html=True)
        
        # Create custom basket selector buttons
        baskets = [
            {"name": "Complex Mapping", "icon": "📊", "description": "Complex Mapping & Analytic Function Simulator"},
            {"name": "Matrixland", "icon": "🧩", "description": "Matrixland & Vector Playground"},
            {"name": "Eigen Exploratorium", "icon": "🔍", "description": "Eigen Exploratorium"},
            {"name": "Inner Product Lab", "icon": "⚙️", "description": "Inner Product & Orthonormalization Lab"}
        ]
        
        # Create styled selection buttons
        for basket in baskets:
            # Check if this basket is selected
            is_selected = st.session_state["selected_basket"] == basket["name"]
            
            # Create styled button with columns
            col1, col2 = st.columns([1, 5])
            
            with col1:
                st.markdown(f"<div style='font-size:24px; color: #ECEFCA;'>{basket['icon']}</div>", unsafe_allow_html=True)
            
            with col2:
                button_label = f"{basket['name']}"
                if st.button(
                    button_label, 
                    key=f"btn_{basket['name']}", 
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state["selected_basket"] = basket["name"]
                    st.rerun()
                
                # Description under the button
                if basket["name"] == st.session_state["selected_basket"]:
                    st.markdown(f"<p style='margin: 0 0 10px 0; font-size: 0.85rem; color: #ECEFCA;'>{basket['description']}</p>", unsafe_allow_html=True)
        
        # Add separator before basket details
        st.markdown("""
        <div style="background: linear-gradient(90deg, #4F46E5, #818CF8); height: 1px; margin: 20px 0;"></div>
        """, unsafe_allow_html=True)
        
        # Add image size settings
        st.markdown("### <span style='color: #ECEFCA;'>Visualization Settings</span>", unsafe_allow_html=True)
        
        # Initialize default image size if not in session state
        if "default_image_size" not in st.session_state:
            st.session_state["default_image_size"] = "medium"
        
        # Image size selector
        st.session_state["default_image_size"] = st.select_slider(
            "Default Image Size",
            options=["small", "medium", "large"],
            value=st.session_state["default_image_size"]
        )
        
        # Add explanation for image controls
        with st.expander("About Image Controls"):
            st.markdown("""
            <div style='color: #ECEFCA;'>
            **Image Sizing Options:**
            - Each visualization now has a zoom slider
            - You can adjust the default size using the slider above
            - Individual controls appear under each visualization
            </div>
            """, unsafe_allow_html=True)
        
        # Add another separator
        st.markdown("""
        <div style="background: linear-gradient(90deg, #4F46E5, #818CF8); height: 1px; margin: 20px 0;"></div>
        """, unsafe_allow_html=True)
        
        # Display basket information
        if st.session_state["selected_basket"] == "Complex Mapping":
            st.markdown("#### <span style='color: #ECEFCA;'>📦 Complex Mapping & Analytic Function Simulator</span>", unsafe_allow_html=True)
            st.markdown("""
            <div style='color: #ECEFCA;'>
            Explore and visualize:
            - Analytic Functions
            - Cauchy-Riemann Equations
            - Harmonic Functions
            - Complex Integration
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state["selected_basket"] == "Matrixland":
            st.markdown("#### <span style='color: #ECEFCA;'>📦 Matrixland & Vector Playground</span>", unsafe_allow_html=True)
            st.markdown("""
            <div style='color: #ECEFCA;'>
            Explore and visualize:
            - Vector Spaces
            - Subspaces
            - Transformations
            - Linear Independence
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state["selected_basket"] == "Eigen Exploratorium":
            st.markdown("#### <span style='color: #ECEFCA;'>📦 Eigen Exploratorium</span>", unsafe_allow_html=True)
            st.markdown("""
            <div style='color: #ECEFCA;'>
            Explore and visualize:
            - Eigenvalues
            - Eigenvectors
            - Diagonalization
            - Applications
            </div>
            """, unsafe_allow_html=True)
        elif st.session_state["selected_basket"] == "Inner Product Lab":
            st.markdown("#### <span style='color: #ECEFCA;'>📦 Inner Product & Orthonormalization Lab</span>", unsafe_allow_html=True)
            st.markdown("""
            <div style='color: #ECEFCA;'>
            Explore and visualize:
            - Dot Products
            - Gram-Schmidt Process
            - Orthonormal Bases
            - Projections
            </div>
            """, unsafe_allow_html=True)
        
        # About section
        with st.expander("About CVLA Lab"):
            st.markdown("""
            <div style='color: #ECEFCA;'>
            This interactive lab helps explore complex 
            variables and linear algebra through visual tools
            and AI-powered simulations.
            
            Built with Python, Streamlit, and mathematical libraries.
            </div>
            """, unsafe_allow_html=True)
    
    return st.session_state["selected_basket"] 