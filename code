import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import io

# Set up the plotting style
plt.style.use('seaborn-v0_8-darkgrid')

def calculate_kinetics(df):
    """
    Performs kinetic calculations to generate the second table and determine parameters.
    """
    # Ensure columns are numeric
    df['Time, Sec'] = pd.to_numeric(df['Time, Sec'], errors='coerce')
    df['W remaining (gm)'] = pd.to_numeric(df['W remaining (gm)'], errors='coerce')

    # Drop rows with NaN in critical columns
    df_cleaned = df.dropna(subset=['Time, Sec', 'W remaining (gm)']).copy()

    if len(df_cleaned) < 2:
        st.warning("Insufficient data (less than 2 valid rows) for rate calculation.")
        return None, (np.nan, np.nan, np.nan)

    # --- Table 2 Calculations ---
    df_table2 = pd.DataFrame()
    df_table2['Time'] = df_cleaned['Time, Sec']
    df_table2['W remaining (gm)'] = df_cleaned['W remaining (gm)']
    df_table2['Ln (Wremaining)'] = np.log(df_table2['W remaining (gm)'])

    # 1. Rate Calculation (Approximation using finite difference)
    # Rate = -dW / dt
    time_diff = np.diff(df_cleaned['Time, Sec'])
    w_diff = np.diff(df_cleaned['W remaining (gm)'])

    # Rates: -ΔW / Δt (Disappearance rate is positive)
    rates_actual = -w_diff / time_diff

    # Associate rate with the average concentration and time of the interval
    w_remaining_for_rate_plot = (df_cleaned['W remaining (gm)'].iloc[:-1].values + df_cleaned['W remaining (gm)'].iloc[1:].values) / 2
    time_for_rate_plot = (df_cleaned['Time, Sec'].iloc[:-1].values + df_cleaned['Time, Sec'].iloc[1:].values) / 2

    # Prepare columns for Table 2 display (add NaNs for alignment)
    rate_col = [np.nan] + list(rates_actual)
    df_table2['Rate'] = rate_col

    # 2. Ln(rate) Calculation
    # Note: We take log only for positive rates.
    df_table2['Ln (rate)'] = np.log(df_table2['Rate'].mask(df_table2['Rate'] <= 0, np.nan))
    
    # 3. Filter for Ln(Rate) vs Ln(Wremaining) Plot
    valid_indices = (rates_actual > 0) & (w_remaining_for_rate_plot > 0)
    rates_for_log = rates_actual[valid_indices]
    w_remaining_for_log = w_remaining_for_rate_plot[valid_indices]
    
    # --- Requirements (Kinetics Parameters) Calculation ---
    slope_n, intercept_ln_k, k_value = np.nan, np.nan, np.nan
    
    if len(rates_for_log) >= 2:
        try:
            # Perform linear regression for Ln(Rate) vs Ln(Wremaining)
            ln_w = np.log(w_remaining_for_log)
            ln_rate = np.log(rates_for_log)
            
            # Linear Regression: Ln(Rate) = n * Ln(Wremaining) + Ln(K)
            slope_n, intercept_ln_k, _, _, _ = linregress(ln_w, ln_rate)
            k_value = np.exp(intercept_ln_k)
            
        except Exception as e:
            st.error(f"Error during linear regression: {e}")
            
    # Reorder columns for display
    df_table2 = df_table2[['Time', 'W remaining (gm)', 'Ln (Wremaining)', 'Rate', 'Ln (rate)']]
    
    return df_table2, (slope_n, intercept_ln_k, k_value)

def plot_graphs(df1, df2, slope_n, intercept_ln_k):
    """
    Generates the three required kinetic graphs.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Reaction Kinetics Analysis', fontsize=16)

    # --- Graph 1: W remaining Versus Time ---
    axes[0].plot(df1['Time, Sec'], df1['W remaining (gm)'], marker='o', color='blue')
    axes[0].set_title('Graph 1: W remaining Versus Time')
    axes[0].set_xlabel('Time (sec)')
    axes[0].set_ylabel('W remaining (gm)')

    # --- Graph 2: Ln(Wremaining) Versus Time ---
    axes[1].plot(df2['Time'], df2['Ln (Wremaining)'], marker='o', color='green')
    axes[1].set_title('Graph 2: Ln(Wremaining) Versus Time')
    axes[1].set_xlabel('Time (sec)')
    axes[1].set_ylabel('Ln(Wremaining)')

    # --- Graph 3: Ln(Rate) Versus Ln(Wremaining) ---
    axes[2].set_title('Graph 3: Ln(Rate) Versus Ln(Wremaining)')
    axes[2].set_xlabel('Ln(Wremaining)')
    axes[2].set_ylabel('Ln(Rate)')
    
    # Data preparation for Ln(Rate) vs Ln(Wremaining) plot
    # Use the cleaned data points from the calculation function
    df_rate_kinetics = df2.dropna(subset=['Ln (rate)', 'Ln (Wremaining)']).copy()
    
    if len(df_rate_kinetics) >= 2 and not np.isnan(slope_n):
        ln_w_points = np.log(df2['W remaining (gm)'].iloc[:-1].values + df2['W remaining (gm)'].iloc[1:].values) / 2
        ln_rate_points = df2['Ln (rate)'].iloc[1:].values
        
        # Filter for valid log values
        valid_indices = (~np.isnan(ln_w_points)) & (~np.isnan(ln_rate_points))
        ln_w_points = ln_w_points[valid_indices]
        ln_rate_points = ln_rate_points[valid_indices]
        
        axes[2].scatter(ln_w_points, ln_rate_points, color='red')
        
        # Plot linear regression line
        x_lin_reg = np.array([ln_w_points.min() - 0.1, ln_w_points.max() + 0.1])
        y_lin_reg = slope_n * x_lin_reg + intercept_ln_k
        axes[2].plot(x_lin_reg, y_lin_reg, color='blue', linestyle='--', 
                     label=f'Linear Fit: y = {slope_n:.2f}x + {intercept_ln_k:.2f}')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'Insufficient valid data for Ln(Rate) vs Ln(Wremaining)', 
                     horizontalalignment='center', verticalalignment='center', 
                     transform=axes[2].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Reaction Kinetics Analyzer")

st.title("🔬 Reaction Kinetics Analysis App")
st.markdown("Upload your experimental data to calculate kinetic parameters and visualize results.")

# Example Data from the user's image (for easy start)
example_data = {
    'Time, Sec': [0, 120, 240, 360, 480, 600],
    'Volume of Bio-Oil (ml)': [0, 1.17, 2.13, 2.85, 3.0, 3.0],
    'W remaining (gm)': [15.0, 13.98, 12.74, 11.979, 11.82, 11.82]
}
example_df = pd.DataFrame(example_data)

st.header("1. Input Data (Table 1)")

# Data input method selection
data_method = st.radio("Choose Input Method:", ('Paste/Manual Entry', 'Upload CSV/Excel'), index=0)

df_input = pd.DataFrame()
if data_method == 'Paste/Manual Entry':
    st.info("Edit the table below or use the 'Upload CSV/Excel' option.")
    df_input = st.data_editor(
        example_df, 
        num_rows="dynamic",
        use_container_width=True
    )
    
elif data_method == 'Upload CSV/Excel':
    uploaded_file = st.file_uploader("Upload your CSV or Excel file (must contain 'Time, Sec', 'Volume of Bio-Oil (ml)', and 'W remaining (gm)')", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
            
            # Ensure required columns exist and are displayed
            required_cols = ['Time, Sec', 'Volume of Bio-Oil (ml)', 'W remaining (gm)']
            for col in required_cols:
                if col not in df_input.columns:
                    st.error(f"Required column '{col}' is missing in the uploaded file.")
                    df_input = pd.DataFrame() # Clear df_input if validation fails
                    break
            
            if not df_input.empty:
                st.write("Uploaded Data Preview:")
                df_input = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Check if we have valid data to process
if not df_input.empty and len(df_input) > 1 and st.button("Calculate Kinetics and Generate Results"):
    
    with st.spinner('Calculating...'):
        # Rename the resulting data to match the original structure (Table 1)
        df_table1 = df_input.rename(columns={
            'Time, Sec': 'Time, Sec', 
            'Volume of Bio-Oil (ml)': 'Volume of Bio-Oil (ml)', 
            'W remaining (gm)': 'W remaining (gm)'
        })
        
        # Run calculations
        df_table2, (slope_n, intercept_ln_k, rate_constant_k) = calculate_kinetics(df_table1)

        if df_table2 is not None:
            
            # --- Results Display ---
            st.header("2. Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Table 1: Input Data (With Catalyst Data)")
                st.dataframe(df_table1.iloc[:, :3], use_container_width=True) # Show only the first 3 input columns

            with col2:
                st.subheader("Table 2: Concentration, Rate, Ln values")
                st.dataframe(df_table2.round(4), use_container_width=True)
                
            st.markdown("---")
            
            # --- Requirements Display ---
            st.subheader("3. Requirements: Kinetic Parameters")
            
            # Display results in columns
            req_col1, req_col2, req_col3 = st.columns(3)
            
            req_col1.metric("Slope ($n$) (Reaction Order)", f"{slope_n:.4f}" if not np.isnan(slope_n) else "N/A")
            req_col2.metric("Intercept ($\text{Ln}(K)$)", f"{intercept_ln_k:.4f}" if not np.isnan(intercept_ln_k) else "N/A")
            req_col3.metric("K (Rate Constant)", f"{rate_constant_k:.4e}" if not np.isnan(rate_constant_k) else "N/A")

            # --- Graphs Display ---
            st.header("4. Kinetic Graphs")
            fig = plot_graphs(df_table1, df_table2, slope_n, intercept_ln_k)
            st.pyplot(fig)
            
        else:
            st.error("Please provide at least two valid data points (Time and W remaining) to perform the analysis.")
