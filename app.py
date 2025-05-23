import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import mysql.connector
import psycopg2
from sqlalchemy import create_engine
import numpy as np
from typing import Optional, Dict, Any
import json
from difflib import unified_diff
import re
from collections import Counter
from PIL import Image, ImageChops, ImageStat
import io
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="File Comparison Tool",
    page_icon="📊",
    layout="wide"
)

# Simplified CSS
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 25px;
        font-weight: bold;
    }
    .error-message {
        color: #ff0000;
        font-weight: bold;
    }
    .success-message {
        color: #008000;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def load_file(file):
    """Load file based on its extension"""
    if file is None:
        return None
    
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file, engine='python')
        elif file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def load_from_sql(connection_type: str, connection_params: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Load data from SQL database"""
    try:
        if connection_type == "sqlite":
            conn = sqlite3.connect(connection_params["database"])
        elif connection_type == "mysql":
            conn = mysql.connector.connect(
                host=connection_params["host"],
                user=connection_params["user"],
                password=connection_params["password"],
                database=connection_params["database"]
            )
        elif connection_type == "postgresql":
            conn = psycopg2.connect(
                host=connection_params["host"],
                user=connection_params["user"],
                password=connection_params["password"],
                database=connection_params["database"]
            )
        else:
            st.error("Unsupported database type")
            return None

        query = connection_params["query"]
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

def compare_dataframes(df1, df2):
    """Compare two dataframes and return differences"""
    if df1 is None or df2 is None:
        return None
    
    # Check if dataframes have any common columns
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        st.error("The files have no common columns to compare!")
        return None
    
    # Check if dataframes have similar structure
    if len(df1.columns) != len(df2.columns):
        st.warning("The files have different number of columns!")
    
    differences = {}
    for col in common_cols:
        # Get unique values in each dataframe
        values1 = set(df1[col].dropna().astype(str).unique())
        values2 = set(df2[col].dropna().astype(str).unique())
        
        # Find values that are in one set but not in the other
        only_in_1 = values1 - values2
        only_in_2 = values2 - values1
        
        if only_in_1 or only_in_2:
            differences[col] = {
                'only_in_file1': list(only_in_1),
                'only_in_file2': list(only_in_2)
            }
    
    return differences

def calculate_similarity_score(df1, df2):
    """Calculate similarity score between two dataframes"""
    if df1 is None or df2 is None:
        return 0
    
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        return 0
    
    total_similarity = 0
    for col in common_cols:
        values1 = set(df1[col].dropna().astype(str).unique())
        values2 = set(df2[col].dropna().astype(str).unique())
        
        if values1 and values2:
            intersection = len(values1.intersection(values2))
            union = len(values1.union(values2))
            similarity = intersection / union
            total_similarity += similarity
    
    return (total_similarity / len(common_cols)) * 100

def load_unstructured_file(file):
    """Load unstructured file based on its extension"""
    if file is None:
        return None
    
    try:
        if file.name.endswith('.txt'):
            content = file.getvalue().decode('utf-8')
            return {'type': 'text', 'content': content}
        elif file.name.endswith('.json'):
            content = json.loads(file.getvalue().decode('utf-8'))
            return {'type': 'json', 'content': content}
        else:
            st.error("Unsupported unstructured file format. Please upload TXT or JSON files.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def compare_text_files(text1, text2):
    """Compare two text files and return differences"""
    if not text1 or not text2:
        return None
    
    # Split text into lines for comparison
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    
    # Generate unified diff
    diff = list(unified_diff(lines1, lines2, lineterm=''))
    
    # Count word frequencies
    words1 = Counter(re.findall(r'\w+', text1.lower()))
    words2 = Counter(re.findall(r'\w+', text2.lower()))
    
    # Find unique words
    unique_words1 = set(words1.keys()) - set(words2.keys())
    unique_words2 = set(words2.keys()) - set(words1.keys())
    
    return {
        'diff': diff,
        'unique_words1': list(unique_words1),
        'unique_words2': list(unique_words2),
        'word_freq1': dict(words1),
        'word_freq2': dict(words2)
    }

def compare_json_files(json1, json2):
    """Compare two JSON files and return differences"""
    if not json1 or not json2:
        return None
    
    def flatten_json(json_obj, prefix=''):
        items = {}
        for key, value in json_obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(flatten_json(value, new_key))
            else:
                items[new_key] = value
        return items
    
    flat1 = flatten_json(json1)
    flat2 = flatten_json(json2)
    
    # Find differences
    only_in_1 = {k: v for k, v in flat1.items() if k not in flat2 or flat2[k] != v}
    only_in_2 = {k: v for k, v in flat2.items() if k not in flat1 or flat1[k] != v}
    
    return {
        'only_in_1': only_in_1,
        'only_in_2': only_in_2
    }

def load_image_file(file):
    """Load image file and return image data and metadata"""
    if file is None:
        return None
    
    try:
        if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Read image
            image = Image.open(file)
            
            # Get image metadata
            metadata = {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height
            }
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return {
                'type': 'image',
                'content': image,
                'metadata': metadata,
                'filename': file.name
            }
        else:
            st.error("Unsupported image format. Please upload PNG, JPG, JPEG, BMP, or GIF files.")
            return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def compare_images(img1_data, img2_data):
    """Compare two images and return differences using PIL instead of OpenCV"""
    if not img1_data or not img2_data:
        return None
    
    img1 = img1_data['content']
    img2 = img2_data['content']
    
    # Resize images to same size if they're different
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)
    
    # Calculate difference image
    diff = ImageChops.difference(img1, img2)
    
    # Convert difference to grayscale
    diff_gray = diff.convert('L')
    
    # Calculate statistics
    stat = ImageStat.Stat(diff_gray)
    mean_diff = stat.mean[0]
    max_diff = stat.extrema[0][1]
    
    # Create visualization of differences
    diff_vis = img1.copy()
    diff_pixels = diff_gray.point(lambda x: 255 if x > 30 else 0)
    diff_vis.paste((0, 255, 0), mask=diff_pixels)
    
    # Calculate similarity score (0-1)
    similarity_score = 1 - (mean_diff / 255)
    
    # Calculate color histograms
    hist1 = img1.histogram()
    hist2 = img2.histogram()
    
    # Calculate histogram similarity
    hist_similarity = sum(min(h1, h2) for h1, h2 in zip(hist1, hist2)) / sum(hist1)
    
    return {
        'similarity_score': similarity_score,
        'histogram_similarity': hist_similarity,
        'difference_image': diff_vis,
        'metadata_diff': {
            'format': img1_data['metadata']['format'] != img2_data['metadata']['format'],
            'size': img1_data['metadata']['size'] != img2_data['metadata']['size'],
            'mode': img1_data['metadata']['mode'] != img2_data['metadata']['mode']
        },
        'num_differences': int(mean_diff * img1.size[0] * img1.size[1] / 255)
    }

def main():
    st.title("📊 Intelligent File Comparison Tool")
    st.markdown("Compare structured, unstructured, and image data to find differences")
    
    # Data source selection
    data_source = st.radio(
        "Select data source type:",
        ["File Upload", "SQL Database", "Unstructured Data", "Image Comparison"]
    )
    
    if data_source == "File Upload":
        # File upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### First File")
            file1 = st.file_uploader("Upload first file", type=['csv', 'xlsx', 'xls'])
            if file1:
                df1 = load_file(file1)
                if df1 is not None:
                    st.write(f"Preview of {file1.name}:")
                    st.dataframe(df1.head())
        
        with col2:
            st.markdown("### Second File")
            file2 = st.file_uploader("Upload second file", type=['csv', 'xlsx', 'xls'])
            if file2:
                df2 = load_file(file2)
                if df2 is not None:
                    st.write(f"Preview of {file2.name}:")
                    st.dataframe(df2.head())
        
        # Compare button
        if st.button("Compare Files", type="primary"):
            if df1 is not None and df2 is not None:
                # Calculate similarity score
                similarity_score = calculate_similarity_score(df1, df2)
                st.markdown(f"### Similarity Score: {similarity_score:.2f}%")
                
                if similarity_score < 20:
                    st.error("The datasets are significantly different!")
                
                differences = compare_dataframes(df1, df2)
                
                if differences:
                    st.markdown("### 🔍 Comparison Results")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Detailed View", "Summary View", "Data Quality"])
                    
                    with tab1:
                        for col, diff in differences.items():
                            with st.expander(f"Column: {col}", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"**Only in first dataset:**")
                                    if diff['only_in_file1']:
                                        st.write(diff['only_in_file1'])
                                    else:
                                        st.write("No unique values")
                                
                                with col2:
                                    st.markdown(f"**Only in second dataset:**")
                                    if diff['only_in_file2']:
                                        st.write(diff['only_in_file2'])
                                    else:
                                        st.write("No unique values")
                    
                    with tab2:
                        # Create summary visualization
                        summary_data = []
                        for col, diff in differences.items():
                            summary_data.append({
                                'Column': col,
                                'Unique in Dataset 1': len(diff['only_in_file1']),
                                'Unique in Dataset 2': len(diff['only_in_file2'])
                            })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            fig = px.bar(summary_df, 
                                       x='Column',
                                       y=['Unique in Dataset 1', 'Unique in Dataset 2'],
                                       title='Summary of Differences',
                                       barmode='group',
                                       template='plotly_white')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Data quality metrics
                        st.markdown("### Data Quality Metrics")
                        
                        # Compare row counts
                        st.write(f"Row count in first dataset: {len(df1)}")
                        st.write(f"Row count in second dataset: {len(df2)}")
                        
                        # Compare null values
                        null_diff = pd.DataFrame({
                            'Column': common_cols,
                            'Nulls in Dataset 1': [df1[col].isnull().sum() for col in common_cols],
                            'Nulls in Dataset 2': [df2[col].isnull().sum() for col in common_cols]
                        })
                        
                        fig = px.bar(null_diff,
                                   x='Column',
                                   y=['Nulls in Dataset 1', 'Nulls in Dataset 2'],
                                   title='Null Values Comparison',
                                   barmode='group',
                                   template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No differences found between the datasets!")
            else:
                st.warning("Please upload both files to compare.")
    
    elif data_source == "SQL Database":
        # SQL connection section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### First Database")
            db_type1 = st.selectbox("Select database type", ["sqlite", "mysql", "postgresql"], key="db_type_1")
            if db_type1 == "sqlite":
                db_path1 = st.text_input("Database path", key="db_path_1")
                query1 = st.text_area("SQL Query", key="query_1")
                if db_path1 and query1:
                    df1 = load_from_sql("sqlite", {"database": db_path1, "query": query1})
            else:
                host1 = st.text_input("Host", key="host_1")
                user1 = st.text_input("Username", key="user_1")
                password1 = st.text_input("Password", type="password", key="password_1")
                database1 = st.text_input("Database name", key="database_1")
                query1 = st.text_area("SQL Query", key="query_1_alt")
                if all([host1, user1, password1, database1, query1]):
                    df1 = load_from_sql(db_type1, {
                        "host": host1,
                        "user": user1,
                        "password": password1,
                        "database": database1,
                        "query": query1
                    })
        
        with col2:
            st.markdown("### Second Database")
            db_type2 = st.selectbox("Select database type", ["sqlite", "mysql", "postgresql"], key="db_type_2")
            if db_type2 == "sqlite":
                db_path2 = st.text_input("Database path", key="db_path_2")
                query2 = st.text_area("SQL Query", key="query_2")
                if db_path2 and query2:
                    df2 = load_from_sql("sqlite", {"database": db_path2, "query": query2})
            else:
                host2 = st.text_input("Host", key="host_2")
                user2 = st.text_input("Username", key="user_2")
                password2 = st.text_input("Password", type="password", key="password_2")
                database2 = st.text_input("Database name", key="database_2")
                query2 = st.text_area("SQL Query", key="query_2_alt")
                if all([host2, user2, password2, database2, query2]):
                    df2 = load_from_sql(db_type2, {
                        "host": host2,
                        "user": user2,
                        "password": password2,
                        "database": database2,
                        "query": query2
                    })
        
        # Compare button for SQL
        if st.button("Compare SQL Data", type="primary"):
            if df1 is not None and df2 is not None:
                # Calculate similarity score
                similarity_score = calculate_similarity_score(df1, df2)
                st.markdown(f"### Similarity Score: {similarity_score:.2f}%")
                
                if similarity_score < 20:
                    st.error("The datasets are significantly different!")
                
                differences = compare_dataframes(df1, df2)
                
                if differences:
                    st.markdown("### 🔍 Comparison Results")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Detailed View", "Summary View", "Data Quality"])
                    
                    with tab1:
                        for col, diff in differences.items():
                            with st.expander(f"Column: {col}", expanded=True):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"**Only in first dataset:**")
                                    if diff['only_in_file1']:
                                        st.write(diff['only_in_file1'])
                                    else:
                                        st.write("No unique values")
                                
                                with col2:
                                    st.markdown(f"**Only in second dataset:**")
                                    if diff['only_in_file2']:
                                        st.write(diff['only_in_file2'])
                                    else:
                                        st.write("No unique values")
                    
                    with tab2:
                        # Create summary visualization
                        summary_data = []
                        for col, diff in differences.items():
                            summary_data.append({
                                'Column': col,
                                'Unique in Dataset 1': len(diff['only_in_file1']),
                                'Unique in Dataset 2': len(diff['only_in_file2'])
                            })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            fig = px.bar(summary_df, 
                                       x='Column',
                                       y=['Unique in Dataset 1', 'Unique in Dataset 2'],
                                       title='Summary of Differences',
                                       barmode='group',
                                       template='plotly_white')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Data quality metrics
                        st.markdown("### Data Quality Metrics")
                        
                        # Compare row counts
                        st.write(f"Row count in first dataset: {len(df1)}")
                        st.write(f"Row count in second dataset: {len(df2)}")
                        
                        # Compare null values
                        null_diff = pd.DataFrame({
                            'Column': common_cols,
                            'Nulls in Dataset 1': [df1[col].isnull().sum() for col in common_cols],
                            'Nulls in Dataset 2': [df2[col].isnull().sum() for col in common_cols]
                        })
                        
                        fig = px.bar(null_diff,
                                   x='Column',
                                   y=['Nulls in Dataset 1', 'Nulls in Dataset 2'],
                                   title='Null Values Comparison',
                                   barmode='group',
                                   template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("No differences found between the datasets!")
            else:
                st.warning("Please provide both datasets to compare.")
    
    elif data_source == "Unstructured Data":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### First File")
            file1 = st.file_uploader("Upload first file", type=['txt', 'json'], key="unstruct_file1")
            if file1:
                data1 = load_unstructured_file(file1)
                if data1 is not None:
                    st.write(f"Preview of {file1.name}:")
                    if data1['type'] == 'text':
                        st.text_area("Content", data1['content'][:500] + "...", height=200)
                    else:
                        st.json(data1['content'])
        
        with col2:
            st.markdown("### Second File")
            file2 = st.file_uploader("Upload second file", type=['txt', 'json'], key="unstruct_file2")
            if file2:
                data2 = load_unstructured_file(file2)
                if data2 is not None:
                    st.write(f"Preview of {file2.name}:")
                    if data2['type'] == 'text':
                        st.text_area("Content", data2['content'][:500] + "...", height=200)
                    else:
                        st.json(data2['content'])
        
        # Compare button for unstructured data
        if st.button("Compare Unstructured Data", type="primary"):
            if data1 is not None and data2 is not None:
                if data1['type'] == data2['type']:
                    if data1['type'] == 'text':
                        results = compare_text_files(data1['content'], data2['content'])
                        if results:
                            st.markdown("### 🔍 Text Comparison Results")
                            
                            tab1, tab2, tab3 = st.tabs(["Line Differences", "Word Analysis", "Word Frequency"])
                            
                            with tab1:
                                st.markdown("#### Line-by-Line Differences")
                                for line in results['diff']:
                                    if line.startswith('+'):
                                        st.markdown(f"<span style='color: green'>{line}</span>", unsafe_allow_html=True)
                                    elif line.startswith('-'):
                                        st.markdown(f"<span style='color: red'>{line}</span>", unsafe_allow_html=True)
                                    else:
                                        st.text(line)
                            
                            with tab2:
                                st.markdown("#### Unique Words Analysis")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Words only in first file:**")
                                    st.write(results['unique_words1'])
                                with col2:
                                    st.markdown("**Words only in second file:**")
                                    st.write(results['unique_words2'])
                            
                            with tab3:
                                st.markdown("#### Word Frequency Comparison")
                                # Create word frequency visualization
                                words = list(set(results['word_freq1'].keys()) | set(results['word_freq2'].keys()))
                                freq_data = []
                                for word in words:
                                    freq_data.append({
                                        'Word': word,
                                        'Frequency in File 1': results['word_freq1'].get(word, 0),
                                        'Frequency in File 2': results['word_freq2'].get(word, 0)
                                    })
                                
                                if freq_data:
                                    freq_df = pd.DataFrame(freq_data)
                                    fig = px.bar(freq_df.head(20), 
                                               x='Word',
                                               y=['Frequency in File 1', 'Frequency in File 2'],
                                               title='Top 20 Words Frequency Comparison',
                                               barmode='group')
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        results = compare_json_files(data1['content'], data2['content'])
                        if results:
                            st.markdown("### 🔍 JSON Comparison Results")
                            
                            tab1, tab2 = st.tabs(["Differences", "Summary"])
                            
                            with tab1:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Only in first file:**")
                                    st.json(results['only_in_1'])
                                with col2:
                                    st.markdown("**Only in second file:**")
                                    st.json(results['only_in_2'])
                            
                            with tab2:
                                st.markdown("#### Summary")
                                st.write(f"Number of differences in first file: {len(results['only_in_1'])}")
                                st.write(f"Number of differences in second file: {len(results['only_in_2'])}")
                else:
                    st.error("Cannot compare different file types. Please upload files of the same type.")
            else:
                st.warning("Please upload both files to compare.")
    
    elif data_source == "Image Comparison":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### First Image")
            file1 = st.file_uploader("Upload first image", type=['png', 'jpg', 'jpeg', 'bmp', 'gif'], key="img_file1")
            if file1:
                img1_data = load_image_file(file1)
                if img1_data is not None:
                    st.write(f"Preview of {file1.name}:")
                    st.image(img1_data['content'], caption=file1.name, use_column_width=True)
                    st.write("Image Metadata:")
                    st.json(img1_data['metadata'])
        
        with col2:
            st.markdown("### Second Image")
            file2 = st.file_uploader("Upload second image", type=['png', 'jpg', 'jpeg', 'bmp', 'gif'], key="img_file2")
            if file2:
                img2_data = load_image_file(file2)
                if img2_data is not None:
                    st.write(f"Preview of {file2.name}:")
                    st.image(img2_data['content'], caption=file2.name, use_column_width=True)
                    st.write("Image Metadata:")
                    st.json(img2_data['metadata'])
        
        # Compare button for images
        if st.button("Compare Images", type="primary"):
            if img1_data is not None and img2_data is not None:
                results = compare_images(img1_data, img2_data)
                if results:
                    st.markdown("### 🔍 Image Comparison Results")
                    
                    # Display similarity scores
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Structural Similarity", f"{results['similarity_score']:.2%}")
                    with col2:
                        st.metric("Histogram Similarity", f"{results['histogram_similarity']:.2%}")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Visual Differences", "Metadata", "Analysis"])
                    
                    with tab1:
                        st.markdown("#### Visual Differences")
                        st.image(results['difference_image'], caption="Differences highlighted in green", use_column_width=True)
                        st.write(f"Number of differences detected: {results['num_differences']}")
                    
                    with tab2:
                        st.markdown("#### Metadata Comparison")
                        st.write("Metadata Differences:")
                        for key, value in results['metadata_diff'].items():
                            if value:
                                st.warning(f"Different {key}")
                            else:
                                st.success(f"Same {key}")
                    
                    with tab3:
                        st.markdown("#### Image Analysis")
                        # Create a summary of the comparison
                        if results['similarity_score'] > 0.95:
                            st.success("Images are very similar")
                        elif results['similarity_score'] > 0.8:
                            st.info("Images are somewhat similar")
                        else:
                            st.error("Images are significantly different")
                        
                        # Additional analysis
                        st.write("Detailed Analysis:")
                        st.write(f"- Structural similarity score: {results['similarity_score']:.2%}")
                        st.write(f"- Color histogram similarity: {results['histogram_similarity']:.2%}")
                        st.write(f"- Number of different regions: {results['num_differences']}")
            else:
                st.warning("Please upload both images to compare.")

if __name__ == "__main__":
    main() 
