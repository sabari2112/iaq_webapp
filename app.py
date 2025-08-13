from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from final_robust_iaqi_model import FinalRobustIAQIModel
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return "Flask app is working!"

@app.route('/debug')
def debug():
    return f"Upload folder: {app.config['UPLOAD_FOLDER']}, Exists: {os.path.exists(app.config['UPLOAD_FOLDER'])}"

@app.route('/status')
def status():
    return """
    <h1>🚀 Flask App Status</h1>
    <p><strong>✅ Flask app is running!</strong></p>
    <p><strong>Upload folder:</strong> {}</p>
    <p><strong>Folder exists:</strong> {}</p>
    <p><strong>Model files:</strong></p>
    <ul>
        <li>minimal_lstm_model.h5: {}</li>
        <li>minimal_scaler.pkl: {}</li>
    </ul>
    <p><a href="/">← Back to Home</a></p>
    """.format(
        app.config['UPLOAD_FOLDER'],
        os.path.exists(app.config['UPLOAD_FOLDER']),
        os.path.exists('minimal_lstm_model.h5'),
        os.path.exists('minimal_scaler.pkl')
    )

@app.route('/upload', methods=['GET'])
def upload_get():
    return "Upload route is accessible (GET method)"

@app.route('/upload', methods=['POST'])
def upload_file():
    print("\n" + "="*50)
    print("🚀 UPLOAD ROUTE ACCESSED!")
    print("="*50)
    print(f"📁 Request files: {request.files}")
    print(f"📋 Form data: {request.form}")
    print(f"🌐 Request method: {request.method}")
    print(f"📄 Content type: {request.content_type}")
    print(f"📊 Content length: {request.content_length}")
    
    # More robust file checking
    if not request.files or 'csvFile' not in request.files:
        print("❌ No csvFile in request.files")
        print(f"Available files: {list(request.files.keys())}")
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['csvFile']
    print(f"File object: {file}, Filename: {file.filename}")
    
    if file.filename == '':
        print("Empty filename")
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            print(f"Processing file: {file.filename}")
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Initialize the final robust IAQI model (handles all dataset formats)
            print("Initializing FinalRobustIAQIModel...")
            model = FinalRobustIAQIModel()
            print("✅ Final robust IAQI model initialized successfully!")
            
            # Process the CSV file with robust loading
            print("Reading CSV file with robust loader...")
            df = model.safe_load_csv(filepath)
            if df is None:
                raise Exception("Failed to load CSV file with any encoding/format")
            print(f"CSV loaded with {len(df)} rows and columns: {list(df.columns)}")
            
            # Use the last 30 rows for prediction (more stable)
            recent_data = df.tail(30)
            print(f"Using last {len(recent_data)} rows for prediction")
            print(f"Recent data columns: {list(recent_data.columns)}")
            
            # Make predictions for next 7 days using robust model
            print("Generating robust predictions...")
            predictions = model.predict_week(recent_data)
            print(f"Robust predictions generated: {len(predictions)} days")
            
            # Generate future dates
            future_dates = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                           for i in range(7)]
            
            # Prepare results for template (convert numpy types to Python types for JSON serialization)
            results = []
            for i, (date, prediction) in enumerate(zip(future_dates, predictions)):
                results.append({
                    'day': i + 1,
                    'date': date,
                    'iaqi': float(prediction['value']),  # Convert to Python float
                    'category': str(prediction['category']),
                    'emoji': str(prediction['emoji']),
                    'range': str(prediction['range']),
                    'description': str(prediction['description'])
                })
            
            # Calculate summary statistics (convert to Python float for JSON serialization)
            iaqi_values = [float(p['value']) for p in predictions]  # Convert numpy values
            summary = {
                'average': round(sum(iaqi_values) / len(iaqi_values), 1),
                'min': round(min(iaqi_values), 1),
                'max': round(max(iaqi_values), 1),
                'filename': filename
            }
            
            # Store results in session
            session['results'] = results
            session['summary'] = summary
            
            # Clean up uploaded file
            os.remove(filepath)
            
            print("✅ Prediction successful! Redirecting to results...")
            return redirect(url_for('results'))
            
        except Exception as e:
            print(f"DETAILED ERROR: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Invalid file type. Please upload a CSV file.')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'results' not in session:
        flash('No prediction results found. Please upload a CSV file first.')
        return redirect(url_for('index'))
    
    results = session['results']
    summary = session['summary']
    
    return render_template('result.html', results=results, summary=summary)

@app.route('/learnmore')
def learnmore():
    return render_template('learnmore.html')

if __name__ == '__main__':
    print("🚀 Starting Flask App with Minimal IAQ Model")
    print("=" * 50)
    
    # Check if model files exist
    model_exists = os.path.exists('minimal_lstm_model.h5')
    scaler_exists = os.path.exists('minimal_scaler.pkl')
    
    print(f"📊 Model file exists: {model_exists}")
    print(f"📊 Scaler file exists: {scaler_exists}")
    
    if not (model_exists and scaler_exists):
        print("⚠️ Warning: Model files not found!")
        print("📋 Please run the following commands first:")
        print("   1. python train_minimal_lstm.py")
        print("   2. python test_minimal_model.py")
        print("   3. Then run this Flask app")
    else:
        print("✅ Model files found! Ready to serve predictions.")
    
    print("🌐 Starting Flask development server...")
    print("📍 Visit: http://127.0.0.1:5000")
    print("=" * 50)
    
    app.run(debug=True)
