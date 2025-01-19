import pandas as pd
import numpy as np

def transform_corneal_data(input_file, output_file):
    """
    Transform corneal topography data from grid format to structured format.
    Final version with verified coordinate calculations.
    """
    try:
        # 1. Read input data
        grid_data = pd.read_csv(input_file, header=None)
        
        # 2. Create indices
        radial_indices = np.repeat(np.arange(1, 257), 32)    # 1 to 256
        meridian_indices = np.tile(np.arange(1, 33), 256)    # 1 to 32
        
        # 3. Create DataFrame
        df = pd.DataFrame({
            'Meridian_Index': meridian_indices,
            'Radial_Index': radial_indices
        })
        
        # 4. Calculate angles
        df['Meridian_Angle_Deg'] = (df['Meridian_Index'] - 1) * (360.0/32.0)
        df['Meridian_Angle_Rad'] = np.radians(df['Meridian_Angle_Deg'])
        
        # 5. Add keratometry values
        df['Keratometry_Value'] = [grid_data.iloc[r-1, m-1] 
                                 for r, m in zip(df['Radial_Index'], 
                                               df['Meridian_Index'])]
        
        # 6. Calculate normalized radius
        df['Normalized_Radius'] = (df['Radial_Index'] - 1) / 255.0
        
        # 7. Calculate trigonometric components and coordinates
        df['Cos_Theta'] = np.cos(df['Meridian_Angle_Rad'])
        df['Sin_Theta'] = np.sin(df['Meridian_Angle_Rad'])
        df['X_Coordinate'] = df['Normalized_Radius'] * df['Cos_Theta']
        df['Y_Coordinate'] = df['Normalized_Radius'] * df['Sin_Theta']
        
        # 8. Save to CSV
        df.to_csv(output_file, index=False, float_format='%.8f')
        
        # 9. Display verification samples
        print("\nVerification Samples:")
        
        # Show points at different radii for first meridian (0 degrees)
        print("\nPoints along first meridian (0 degrees):")
        meridian1_samples = df[
            (df['Meridian_Index'] == 1) & 
            (df['Radial_Index'].isin([1, 64, 128, 192, 256]))
        ].copy()
        print(meridian1_samples[['Radial_Index', 'Normalized_Radius', 
                               'X_Coordinate', 'Y_Coordinate']].to_string())
        
        # Show points at same radius for different meridians
        print("\nPoints at middle radius (128) for first few meridians:")
        radius128_samples = df[
            (df['Radial_Index'] == 128) & 
            (df['Meridian_Index'].isin([1, 2, 3, 4]))
        ].copy()
        print(radius128_samples[['Meridian_Index', 'Meridian_Angle_Deg',
                               'X_Coordinate', 'Y_Coordinate']].to_string())
        
        return df

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

# Usage
input_file = '/home/aricept094/mydata/testrm.csv'
output_file = '/home/aricept094/mydata/transformed_corneal_data_final.csv'

# Run transformation
transformed_data = transform_corneal_data(input_file, output_file)