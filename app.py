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
    Performs kinetic calculations using the differential method to generate 
    the second table and determine parameters.
    """
    # Ensure columns are numeric
    df['Time, Sec'] = pd.to_numeric(df['Time, Sec'], errors='coerce')
    df['W remaining (gm)'] = pd.to_numeric(df['W remaining (gm)'], errors='coerce')

    # Drop rows with NaN in critical columns
    df_cleaned = df.dropna(subset=['Time, Sec', 'W remaining (gm)']).copy().reset_index(drop=True)

    if len(df_cleaned) < 2:
        st.warning("Insufficient data (less than 2 valid rows) for rate calculation.")
        return None, None, (np.nan, np.nan, np.nan)

    # --- Differential Method Calculations ---
    
    # Differences
    time_diff = np.diff(df_cleaned['Time, Sec'])
    w_diff = np.diff(df_cleaned['W remaining (gm)'])

    # 1. Rate: -ΔW / Δt (Disappearance rate is positive)
    rates_actual = -w_diff / time_diff

    # 2. Average W_remaining (Concentration) for the interval
    w_remaining_avg = (df_cleaned['W remaining (gm)'].iloc[:-1].values + df_cleaned['W remaining (gm)'].iloc[1:].values) / 2
    
    # 3. Time (midpoint of the interval, for potential plotting)
    time_avg = (df_cleaned['Time, Sec'].iloc[:-1].values + df_cleaned['Time, Sec'].iloc[1:].values) / 2

    # Create the Kinetics DataFrame (Table 2's core data)
    df_kinetics = pd.DataFrame({
        'Time (sec)': time_avg,
        'W remaining (gm) (Avg)': w_remaining_avg,
        'Rate': rates_actual,
    })
    
    # Filter for valid log entries (Rate > 0 and W_remaining > 0)
    df_kinetics_valid = df_kinetics[(df_kinetics['Rate'] > 0) & (df_kinetics['W remaining (gm) (Avg)'] > 0)].copy()

    # 4. Ln calculations
    df_kinetics_valid['Ln (Wremaining)'] = np.log(df_kinetics_valid['W remaining (gm) (Avg)'])
    df_kinetics_valid['Ln (rate)'] = np.log(df_kinetics_valid['Rate'])

    # --- Final Table 2 Display ---
    # Prepare data for display in the format of the provided image (Time, W remaining, Ln(Wremaining) from input)
    df_table2_display = df_cleaned[['W remaining (gm)', 'Time, Sec']].copy()
    df_table2_display['Ln (Wremaining)'] = np.log(df_table2_display['W remaining (gm)'])
    
    # Add Rate and Ln(rate) columns with appropriate NaN padding to visualize the interval nature
    nan_row = {'Rate': np.nan, 'Ln (rate)': np.nan}
    df_rate_cols = pd.DataFrame([nan_row] + df_kinetics[['Rate', 'Ln (rate)']].to_dict('records'))
    df_table2_display['Rate'] = df_rate_cols['Rate'].values
    df_table2_display['Ln (rate)'] = df_rate_cols['Ln (rate)'].values
    
    # --- Requirements (Kinetics Parameters) Calculation ---
    slope_n, intercept_ln_k, k_value = np.nan, np.nan, np.nan
    
    if len(df_kinetics_valid) >= 2:
        try:
            ln_w = df_kinetics_valid['Ln (Wremaining)']
            ln_rate = df_kinetics_valid['Ln (rate)']
            
            # Linear Regression: Ln(Rate) = n * Ln(Wremaining) + Ln(K)
            slope_n, intercept_ln_k, _, _, _ = linregress(ln_w, ln_rate)
            k_value = np.exp(intercept_ln_k)
            
        except Exception as e:
            st.error(f"Error during linear regression: {e}")
            
    # Reorder and rename columns for display
    df_table2_display.rename(columns={'Time, Sec': 'Time'}, inplace=True)

    return df_table2_display, df_kinetics_valid, (slope_n, intercept_ln_k, k_value)

def plot_graphs(df1, df_kinetics, slope_n, intercept_ln_k):
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

    # --- Graph 2: Ln(Wremaining) Versus Time (First-order check) ---
    axes[1].plot(df1['Time, Sec'], np.log(df1['W remaining (gm)']), marker='o', color='green')
    axes[1].set_title('Graph 2: Ln(Wremaining) Versus Time (First-order Check)')
    axes[1].set_xlabel('Time (sec)')
    axes[1].set_ylabel('Ln(Wremaining)')

    # --- Graph 3: Ln(Rate) Versus Ln(Wremaining) (Order & K Determination) ---
    axes[2].set_title('Graph 3: Ln(Rate) Versus Ln(Wremaining)')
    axes[2].set_xlabel('Ln(Wremaining) (from Avg Conc)')
    axes[2].set_ylabel('Ln(Rate)')
    
    # Data preparation for Ln(Rate) vs Ln(Wremaining) plot
    df_plot_data = df_kinetics.dropna(subset=['Ln (rate)', 'Ln (Wremaining)']).copy()
    
    if len(df_plot_data) >= 2 and not np.isnan(slope_n):
        ln_w = df_plot_data['Ln (Wremaining)']
        ln_rate = df_plot_data['Ln (rate)']
        
        axes[2].scatter(ln_w, ln_rate, color='red')
        
        # Plot linear regression line
        x_lin_reg = np.array([ln_w.min() - 0.1, ln_w.max() + 0.1])
        y_lin_reg = slope_n * x_lin_reg + intercept_ln_k
        axes[2].plot(x_lin_reg, y_lin_reg, color='blue', linestyle='--', 
                     label=f'Linear Fit: $\\ln(r) = {slope_n:.2f}\\ln(W) + {intercept_ln_k:.2f}$')
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, 'Insufficient valid data for Ln(Rate) vs Ln(Wremaining) plot and regression.', 
                     horizontalalignment='center', verticalalignment='center', 
                     transform=axes[2].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Reaction Kinetics Analyzer")

st.title("🔬 Organic Technology Lab: Reaction Kinetics Analysis")
st.markdown("---")
st.markdown("""
This app analyzes the pyrolysis of rice straw using the differential method: $\\ln(r) = \\ln(K) + n \\ln(W_{\\text{remaining}})$.
The rate is approximated by the slope over time intervals: $r \\approx -\\frac{\\Delta W}{\\Delta t}$.
""")

# Example Data from the user's image (for easy start)
example_data = {
    'Time, Sec': [0, 120, 240, 360, 480, 600],
    'Volume of Bio-Oil (ml)': [0, 1.17, 2.13, 2.85, 3.0, 3.0],
    'W remaining (gm)': [15.0, 13.98, 12.74, 11.979, 11.82, 11.82]
}
example_df = pd.DataFrame(example_data)

st.header("1. Input Data (Table 1: With Catalyst Data)")

# Data input method selection
data_method = st.radio("Choose Input Method:", ('Paste/Manual Entry', 'Upload CSV/Excel'), index=0)

df_input = pd.DataFrame()
if data_method == 'Paste/Manual Entry':
    st.info("Edit the data in the table below. Ensure columns are named exactly 'Time, Sec', 'Volume of Bio-Oil (ml)', and 'W remaining (gm)'.")
    df_input = st.data_editor(
        example_df, 
        num_rows="dynamic",
        use_container_width=True
    )
    
elif data_method == 'Upload CSV/Excel':
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
            
            # Basic validation check
            required_cols = ['Time, Sec', 'Volume of Bio-Oil (ml)', 'W remaining (gm)']
            if not all(col in df_input.columns for col in required_cols):
                 st.error(f"Uploaded file must contain all required columns: {required_cols}")
                 df_input = pd.DataFrame() 
            elif not df_input.empty:
                st.write("Uploaded Data Preview (editable):")
                df_input = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Check if we have valid data to process
if not df_input.empty and len(df_input) > 1 and st.button("Calculate Kinetics and Generate Results"):
    
    with st.spinner('Calculating...'):
        df_table1 = df_input.copy()
        
        # Run calculations
        df_table2_display, df_kinetics_valid, (slope_n, intercept_ln_k, rate_constant_k) = calculate_kinetics(df_table1)

        if df_table2_display is not None:
            
            # --- Results Display ---
            st.header("2. Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Table 1: Input Data (With Catalyst Data)")
                st.dataframe(df_table1.iloc[:, :3], use_container_width=True)

            with col2:
                st.subheader("Table 2: Concentration, Rate, Ln values")
                st.markdown("*(Note: Rate and Ln(rate) are calculated for the interval between rows. $\\text{Ln}(W_{\\text{remaining}})$ is from the input concentration.)*")
                st.dataframe(df_table2_display[['W remaining (gm)', 'Time', 'Rate', 'Ln (Wremaining)', 'Ln (rate)']].round(4), use_container_width=True)
                
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
            fig = plot_graphs(df_table1, df_kinetics_valid, slope_n, intercept_ln_k)
            st.pyplot(fig)
            
        else:
            st.error("Please provide at least two valid data points (Time and W remaining) to perform the analysis.")
