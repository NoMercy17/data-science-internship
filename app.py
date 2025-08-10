from flask import Flask, jsonify
import pandas as pd
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'simple-secret-key'

@app.route('/')
def index():
    """Home page displaying the engineered data info as JSON"""
    try:
        # Path to your engineered data
        data_path = 'app/models/preprocessors/engineered_data.csv'
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Basic info about the dataset
            info = {
                'status': 'success',
                'message': 'Feature-engineered data loaded successfully',
                'filename': 'engineered_data.csv',
                'shape': {
                    'rows': int(df.shape[0]),
                    'columns': int(df.shape[1])
                },
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'summary_stats': df.describe().to_dict(),
                'sample_data': df.head(5).to_dict('records')
            }
            
            return jsonify(info)
        else:
            return jsonify({
                'status': 'error',
                'message': f'Data file not found at: {data_path}',
                'current_directory': os.getcwd(),
                'files_in_app': os.listdir('app') if os.path.exists('app') else 'app directory not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error loading data: {str(e)}',
            'current_directory': os.getcwd()
        }), 500

@app.route('/api/data')
def get_data():
    """API endpoint to get the data as JSON"""
    try:
        data_path = 'app/models/preprocessors/engineered_data.csv'
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return jsonify({
                'status': 'success',
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data': df.head(50).to_dict('records')
            })
        else:
            return jsonify({'status': 'error', 'message': 'Data file not found'}), 404
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'Flask app is running',
        'current_directory': os.getcwd(),
        'python_path': os.sys.path[0] if hasattr(os, 'sys') else 'N/A'
    })

@app.route('/files')
def list_files():
    """Debug endpoint to see file structure"""
    try:
        current_dir = os.getcwd()
        files_info = {
            'current_directory': current_dir,
            'files_in_root': os.listdir('.'),
        }
        
        # Check if app directory exists
        if os.path.exists('app'):
            files_info['app_contents'] = os.listdir('app')
            
            # Check models directory
            if os.path.exists('app/models'):
                files_info['models_contents'] = os.listdir('app/models')
                
                # Check preprocessors directory
                if os.path.exists('app/models/preprocessors'):
                    files_info['preprocessors_contents'] = os.listdir('app/models/preprocessors')
        
        return jsonify(files_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask app...")
    print(f"Current working directory: {os.getcwd()}")
    print("Looking for data at: app/models/preprocessors/engineered_data.csv")
    print(f"File exists: {os.path.exists('app/models/preprocessors/engineered_data.csv')}")
    app.run(host='0.0.0.0', port=5000, debug=True)