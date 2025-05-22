import streamlit as st
import pandas as pd
import plotly.express as px

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
    </style>
    """, unsafe_allow_html=True)

def load_file(file):
    """Load file based on its extension"""
    if file is None:
        return None
    
    try:
        if file.name.endswith('.csv'):
            # Using python engine for better compatibility with pandas 2.1.3
            return pd.read_csv(file, engine='python')
        elif file.name.endswith(('.xlsx', '.xls')):
            # Using openpyxl engine for Excel files
            return pd.read_excel(file, engine='openpyxl')
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def compare_dataframes(df1, df2):
    """Compare two dataframes and return differences"""
    if df1 is None or df2 is None:
        return None
    
    # Get common columns
    common_cols = list(set(df1.columns) & set(df2.columns))
    
    differences = {}
    for col in common_cols:
        # Get unique values in each dataframe
        # Using astype(str) for better compatibility with numpy 1.26.4
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

def main():
    st.title("📊 Intelligent File Comparison Tool")
    st.markdown("Compare CSV and Excel files to find differences in data")
    
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
        if file1 and file2:
            differences = compare_dataframes(df1, df2)
            
            if differences:
                st.markdown("### 🔍 Comparison Results")
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(["Detailed View", "Summary View"])
                
                with tab1:
                    for col, diff in differences.items():
                        with st.expander(f"Column: {col}", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**Only in {file1.name}:**")
                                if diff['only_in_file1']:
                                    st.write(diff['only_in_file1'])
                                else:
                                    st.write("No unique values")
                            
                            with col2:
                                st.markdown(f"**Only in {file2.name}:**")
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
                            'Unique in File 1': len(diff['only_in_file1']),
                            'Unique in File 2': len(diff['only_in_file2'])
                        })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        fig = px.bar(summary_df, 
                                   x='Column',
                                   y=['Unique in File 1', 'Unique in File 2'],
                                   title='Summary of Differences',
                                   barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No differences found between the files!")
        else:
            st.warning("Please upload both files to compare.")

if __name__ == "__main__":
    main() 
