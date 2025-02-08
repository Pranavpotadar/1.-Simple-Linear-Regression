import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score

def load_model_and_data():
    """Load the trained model and dataset"""
    model = joblib.load('appliance_energy_model.pkl')
    df = pd.read_csv('appliance_energy.csv')
    return model, df

def create_prediction_plot(df, model):
    """Create scatter plot with regression line"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot actual data points
    ax.scatter(df['Temperature (°C)'], df['Energy Consumption (kWh)'], 
              color='blue', alpha=0.5, label='Actual Data')
    
    # Create regression line
    temp_range = np.linspace(df['Temperature (°C)'].min(), 
                            df['Temperature (°C)'].max(), 100)
    energy_pred = model.predict(temp_range.reshape(-1, 1))
    
    # Plot regression line
    ax.plot(temp_range, energy_pred, color='red', label='Regression Line')
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Energy Consumption (kWh)')
    ax.set_title('Energy Consumption vs Temperature')
    ax.legend()
    
    return fig

def main():
    st.set_page_config(page_title="Energy Consumption Predictor", layout="wide")
    
    st.title("⚡ Appliance Energy Consumption Predictor")
    st.write("Predict energy consumption based on temperature using Linear Regression")
    
    try:
        # Load model and data
        model, df = load_model_and_data()
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Model Visualization")
            fig = create_prediction_plot(df, model)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Make a Prediction")
            
            # Temperature input
            temperature = st.number_input(
                "Enter Temperature (°C):",
                min_value=float(df['Temperature (°C)'].min()),
                max_value=float(df['Temperature (°C)'].max()),
                value=20.0,
                step=0.1
            )
            
            if st.button("Predict Energy Consumption"):
                prediction = model.predict([[temperature]])[0]
                st.success(f"Predicted Energy Consumption: {prediction:.2f} kWh")
                
                # Add energy-saving tips based on prediction
                if prediction > 2.5:
                    st.warning("⚠️ High energy consumption predicted! Consider these tips:")
                    st.markdown("""
                    - Adjust thermostat settings
                    - Check for appliance efficiency
                    - Use natural ventilation when possible
                    """)
        
        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Calculate and display model metrics
        st.subheader("Model Performance Metrics")
        
        # Make predictions on entire dataset
        y_pred = model.predict(df[['Temperature (°C)']])
        
        # Calculate metrics
        mse = mean_squared_error(df['Energy Consumption (kWh)'], y_pred)
        r2 = r2_score(df['Energy Consumption (kWh)'], y_pred)
        
        # Display metrics in columns
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with metric_col2:
            st.metric("R² Score", f"{r2:.4f}")
            
        # Add correlation analysis
        st.subheader("Statistical Analysis")
        correlation = df['Temperature (°C)'].corr(df['Energy Consumption (kWh)'])
        st.write(f"Correlation between Temperature and Energy Consumption: {correlation:.4f}")
        
        # Download prediction feature
        if st.button("Generate Sample Predictions CSV"):
            # Create sample predictions for temperature range
            temp_range = np.linspace(df['Temperature (°C)'].min(), 
                                   df['Temperature (°C)'].max(), 50)
            predictions = model.predict(temp_range.reshape(-1, 1))
            
            # Create DataFrame with predictions
            pred_df = pd.DataFrame({
                'Temperature (°C)': temp_range,
                'Predicted Energy Consumption (kWh)': predictions
            })
            
            # Convert to CSV
            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="energy_predictions.csv",
                mime="text/csv"
            )
            
    except FileNotFoundError:
        st.error("Please ensure 'appliance_energy_model.pkl' and 'appliance_energy.csv' are in the same directory as the app.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()