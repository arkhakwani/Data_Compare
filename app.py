import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import mysql.connector
import psycopg2
from sqlalchemy import create_engine
import numpy as np
from typing import Optional, Dict, Any

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

def main():
    st.title("📊 Intelligent File Comparison Tool")
    st.markdown("Compare CSV, Excel files, or SQL databases to find differences in data")
    
    # Data source selection
    data_source = st.radio(
        "Select data source type:",
        ["File Upload", "SQL Database"]
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
    else:
        # SQL connection section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### First Database")
            db_type1 = st.selectbox("Select database type", ["sqlite", "mysql", "postgresql"])
            if db_type1 == "sqlite":
                db_path1 = st.text_input("Database path")
                query1 = st.text_area("SQL Query")
                if db_path1 and query1:
                    df1 = load_from_sql("sqlite", {"database": db_path1, "query": query1})
            else:
                host1 = st.text_input("Host")
                user1 = st.text_input("Username")
                password1 = st.text_input("Password", type="password")
                database1 = st.text_input("Database name")
                query1 = st.text_area("SQL Query")
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
            db_type2 = st.selectbox("Select database type", ["sqlite", "mysql", "postgresql"])
            if db_type2 == "sqlite":
                db_path2 = st.text_input("Database path")
                query2 = st.text_area("SQL Query")
                if db_path2 and query2:
                    df2 = load_from_sql("sqlite", {"database": db_path2, "query": query2})
            else:
                host2 = st.text_input("Host")
                user2 = st.text_input("Username")
                password2 = st.text_input("Password", type="password")
                database2 = st.text_input("Database name")
                query2 = st.text_area("SQL Query")
                if all([host2, user2, password2, database2, query2]):
                    df2 = load_from_sql(db_type2, {
                        "host": host2,
                        "user": user2,
                        "password": password2,
                        "database": database2,
                        "query": query2
                    })
    
    # Compare button
    if st.button("Compare Data", type="primary"):
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

if __name__ == "__main__":
    main() 
