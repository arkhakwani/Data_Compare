# Intelligent File Comparison Tool

A modern and user-friendly application for comparing CSV and Excel files to find differences in data.

## Features

- Compare CSV and Excel files (.csv, .xlsx, .xls)
- Find unique values in each column between files
- Interactive and modern UI
- Detailed and summary views of differences
- Visual representation of differences using charts
- Ignores null and blank values
- Real-time file preview

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload two files to compare:
   - Click "Browse files" to upload your first file
   - Click "Browse files" to upload your second file
   - Click the "Compare Files" button to see the differences

4. View the results:
   - Detailed View: Shows all unique values for each column
   - Summary View: Shows a bar chart of the number of differences per column

## Requirements

- Python 3.7+
- Dependencies listed in requirements.txt

## Note

- The application ignores null and blank values during comparison
- Only common columns between the files are compared
- Files must be in CSV or Excel format 