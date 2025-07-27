import streamlit as st
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import plotly.graph_objects as go

def solve_equation(equation_str, variables=None):
    """
    Solve mathematical equations using symbolic computation.
    
    Parameters:
    -----------
    equation_str : str
        String representation of the equation (e.g., "x**2 + 2*x - 1 = 0")
    variables : list of str, optional
        List of variable names to solve for
    """
    try:
        # Parse equation string
        if "=" in equation_str:
            lhs, rhs = equation_str.split("=")
            equation = parse_expr(lhs) - parse_expr(rhs)
        else:
            equation = parse_expr(equation_str)
        
        # Get variables if not provided
        if variables is None:
            variables = list(equation.free_symbols)
        else:
            variables = [sp.Symbol(var) for var in variables]
        
        # Display the equation in LaTeX
        st.markdown("### Original Equation")
        st.latex(sp.latex(equation) + " = 0")
        
        # Solve the equation
        solutions = sp.solve(equation, variables)
        
        st.markdown("### Solutions")
        if isinstance(solutions, list):
            for i, sol in enumerate(solutions):
                if isinstance(sol, tuple):
                    # Handle tuple solutions (e.g., from solve_linear_system)
                    sol_str = ", ".join([sp.latex(s) for s in sol])
                    st.latex(f"x_{i+1} = ({sol_str})")
                else:
                    st.latex(f"x_{i+1} = {sp.latex(sol)}")
        elif isinstance(solutions, dict):
            for var, sol in solutions.items():
                st.latex(f"{sp.latex(var)} = {sp.latex(sol)}")
        else:
            st.latex(f"x = {sp.latex(solutions)}")
        
        # Verify solutions
        st.markdown("### Solution Verification")
        if isinstance(solutions, list):
            for i, sol in enumerate(solutions):
                if isinstance(sol, tuple):
                    # Handle tuple solutions
                    substituted = equation
                    for var, s in zip(variables, sol):
                        substituted = substituted.subs(var, s)
                else:
                    substituted = equation.subs(variables[0], sol)
                st.latex(f"\\text{{Solution {i+1} verification: }}" + sp.latex(substituted) + " = 0")
        elif isinstance(solutions, dict):
            substituted = equation
            for var, sol in solutions.items():
                substituted = substituted.subs(var, sol)
            st.latex("\\text{Solution verification: }" + sp.latex(substituted) + " = 0")
        
        # Additional analysis
        st.markdown("### Additional Analysis")
        
        # Check if equation is polynomial for each variable
        for var in variables:
            if equation.is_polynomial(var):
                degree = sp.degree(equation, gen=var)
                st.markdown(f"- This is a polynomial in {sp.latex(var)} of degree {degree}")
                
                if degree <= 4:
                    try:
                        discriminant = sp.discriminant(equation, var)
                        st.markdown(f"- Discriminant in {sp.latex(var)}: {sp.latex(discriminant)}")
                    except:
                        pass
        
        # Check if equation has real solutions
        real_sols = []
        if isinstance(solutions, list):
            for sol in solutions:
                if isinstance(sol, tuple):
                    real_sols.extend([s for s in sol if s.is_real])
                else:
                    if sol.is_real:
                        real_sols.append(sol)
        st.markdown(f"- Number of real solutions: {len(real_sols)}")
        
        # Check if equation has complex solutions
        complex_sols = []
        if isinstance(solutions, list):
            for sol in solutions:
                if isinstance(sol, tuple):
                    complex_sols.extend([s for s in sol if not s.is_real])
                else:
                    if not sol.is_real:
                        complex_sols.append(sol)
        if complex_sols:
            st.markdown("- Complex solutions exist")
            for i, sol in enumerate(complex_sols):
                st.latex(f"z_{i+1} = {sp.latex(sol)}")
        
        # Visualize if possible (for single-variable equations)
        if len(variables) == 1:
            st.markdown("### Visualization")
            
            # Create points for plotting
            x_var = variables[0]
            x_range = np.linspace(-10, 10, 1000)
            y_values = []
            
            for x in x_range:
                try:
                    y = float(equation.subs(x_var, x))
                    y_values.append(y)
                except:
                    y_values.append(np.nan)
            
            # Create plot
            fig = go.Figure()
            
            # Add equation curve
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_values,
                    mode='lines',
                    name='f(x)'
                )
            )
            
            # Add x-axis
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[0] * len(x_range),
                    mode='lines',
                    line=dict(color='black', width=1),
                    name='y = 0'
                )
            )
            
            # Add solution points
            if isinstance(solutions, list):
                for i, sol in enumerate(solutions):
                    if not isinstance(sol, tuple):  # Only plot non-tuple solutions
                        try:
                            x_val = float(sol)
                            fig.add_trace(
                                go.Scatter(
                                    x=[x_val],
                                    y=[0],
                                    mode='markers',
                                    marker=dict(size=10, color='red'),
                                    name=f'Solution {i+1}'
                                )
                            )
                        except:
                            pass
            
            fig.update_layout(
                title='Equation Visualization',
                xaxis_title='x',
                yaxis_title='f(x)',
                showlegend=True
            )
            
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"Error solving equation: {str(e)}")
        st.error("Please check your equation format and try again.")

def solve_system(equations, variables=None):
    """
    Solve a system of equations.
    
    Parameters:
    -----------
    equations : list of str
        List of equation strings
    variables : list of str, optional
        List of variable names to solve for
    """
    try:
        # Parse equations
        system = []
        for eq_str in equations:
            if "=" in eq_str:
                lhs, rhs = eq_str.split("=")
                system.append(parse_expr(lhs) - parse_expr(rhs))
            else:
                system.append(parse_expr(eq_str))
        
        # Get variables if not provided
        if variables is None:
            variables = list(set().union(*[eq.free_symbols for eq in system]))
        else:
            variables = [sp.Symbol(var) for var in variables]
        
        # Display the system in LaTeX
        st.markdown("### System of Equations")
        st.latex("\\begin{cases} " + " \\\\ ".join([sp.latex(eq) + " = 0" for eq in system]) + " \\end{cases}")
        
        # Solve the system
        solutions = sp.solve(system, variables)
        
        st.markdown("### Solutions")
        if isinstance(solutions, list):
            for i, sol in enumerate(solutions):
                st.latex(f"\\text{{Solution {i+1}:}} \\quad " + ", ".join([f"{sp.latex(var)} = {sp.latex(val)}" for var, val in zip(variables, sol)]))
        elif isinstance(solutions, dict):
            st.latex(", ".join([f"{sp.latex(var)} = {sp.latex(sol)}" for var, sol in solutions.items()]))
        
        # Verify solutions
        st.markdown("### Solution Verification")
        if isinstance(solutions, list):
            for i, sol in enumerate(solutions):
                st.markdown(f"**Solution {i+1}:**")
                for j, eq in enumerate(system):
                    substituted = eq
                    for var, val in zip(variables, sol):
                        substituted = substituted.subs(var, val)
                    st.latex(f"\\text{{Equation {j+1}: }}" + sp.latex(substituted) + " = 0")
        elif isinstance(solutions, dict):
            for i, eq in enumerate(system):
                substituted = eq
                for var, sol in solutions.items():
                    substituted = substituted.subs(var, sol)
                st.latex(f"\\text{{Equation {i+1}: }}" + sp.latex(substituted) + " = 0")
        
        # Visualize if possible (for 2D systems)
        if len(variables) == 2:
            st.markdown("### Visualization")
            
            # Create meshgrid for plotting
            x_var, y_var = variables[:2]
            x = np.linspace(-10, 10, 100)
            y = np.linspace(-10, 10, 100)
            X, Y = np.meshgrid(x, y)
            
            # Create plot
            fig = go.Figure()
            
            # Plot each equation
            for eq in system:
                Z = np.zeros_like(X)
                for i in range(len(x)):
                    for j in range(len(y)):
                        try:
                            Z[j, i] = float(eq.subs([(x_var, x[i]), (y_var, y[j])]))
                        except:
                            Z[j, i] = np.nan
                
                fig.add_trace(
                    go.Contour(
                        x=x,
                        y=y,
                        z=Z,
                        contours=dict(
                            start=-1,
                            end=1,
                            size=0.1,
                            showlabels=True
                        ),
                        name=f"{sp.latex(eq)} = 0"
                    )
                )
            
            # Add solution points
            if isinstance(solutions, list):
                x_sols = []
                y_sols = []
                for sol in solutions:
                    try:
                        x_val = float(sol[0].evalf())
                        y_val = float(sol[1].evalf())
                        if np.isreal(x_val) and np.isreal(y_val):
                            x_sols.append(x_val)
                            y_sols.append(y_val)
                    except:
                        continue
                
                if x_sols:
                    fig.add_trace(
                        go.Scatter(
                            x=x_sols,
                            y=y_sols,
                            mode='markers',
                            marker=dict(size=10, color='red'),
                            name='Solutions'
                        )
                    )
            
            fig.update_layout(
                title="System Visualization",
                xaxis_title=f"{sp.latex(x_var)}",
                yaxis_title=f"{sp.latex(y_var)}",
                showlegend=True
            )
            
            st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f"Error solving system: {str(e)}")
        st.markdown("""
        Please ensure:
        - All equations are properly formatted
        - Variables are properly defined
        - The system is solvable
        """) 