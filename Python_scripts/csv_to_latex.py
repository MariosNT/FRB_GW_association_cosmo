import pandas as pd
import argparse
import sys

# For a longtable or deluxetable LaTeX table with reference in footnote
def dataframe_to_latex_long_ref(data, output_file=None, caption="", label="", head2="", refcol='Ref.'):
    try:
        # type
        if isinstance(data, pd.DataFrame):
            df = data.copy() 
        elif isinstance(data, str):
            df = pd.read_csv(data)
        else:
            raise ValueError("input must be pandas DataFrame or CSV path")
        
        # Reference column to number
        ref_mapping = {}
        ref_comments = []
        
        if refcol in df.columns:
            # Get only reference
            all_refs = set()
            for cell in df[refcol].dropna():
                if isinstance(cell, str):
                    # split by ','
                    refs_in_cell = [ref.strip() for ref in cell.split(',') if ref.strip()]
                    all_refs.update(refs_in_cell)
            
            # Give number for reference
            for i, ref in enumerate(sorted(all_refs), 1):
                ref_mapping[ref] = str(i)
                ref_comments.append(f"({i}) \\citet{{{ref}}}")
            
            # Replace refcol to number
            def convert_refs_to_numbers(cell):
                if pd.isna(cell) or cell == '':
                    return ''
                refs_in_cell = [ref.strip() for ref in str(cell).split(',') if ref.strip()]
                numbers = [ref_mapping.get(ref, ref) for ref in refs_in_cell]
                return ', '.join(numbers)
            
            df[refcol] = df[refcol].apply(convert_refs_to_numbers)
        
        df.columns = [format_column_name(col) for col in df.columns]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('%', '\\%')
                df[col] = df[col].str.replace('#', '\\#')
                df[col] = df[col].str.replace('&', '\\&')

        num_cols = len(df.columns)
        col_format = 'c' * num_cols
        
        latex_code = f"""\\startlongtable"""

        latex_code += f"""\n\\begin{{deluxetable}}{{{col_format}}}"""

        if label:
            latex_code += f"\n\\label{{{label}}}"
            
        latex_code += "\n\\centering"
        
        if caption:
            latex_code += f"\n\\tablecaption{{{caption}}}\n"
        
        # head
        colhead_format = [f'\\colhead{{{col}}}' for col in df.columns]
        header_row = '\\tablehead{' + ' & '.join(colhead_format) + ' \\\\'
        latex_code += header_row
        if head2:
            latex_code += f"\n{head2}"
        latex_code += '\n}\n'
        
        # data
        latex_code += """\\startdata\n"""
        for _, row in df.iterrows():
            data_row = ' & '.join([str(val) for val in row.values]) + ' \\\\'
            latex_code += data_row + '\n'
        
        latex_code += """\\enddata"""
        
        # Add reference to tablecomments
        if ref_comments:
            latex_code += f"\n\\tablecomments{{{'Reference: '+'; '.join(ref_comments)}}}"
        
        latex_code += "\n\\end{deluxetable}"
        
        # output
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"LaTeX already save to: {output_file}")
        else:
            print("LaTeX table code:")
            print("-" * 50)
            print(latex_code)
        
        return latex_code
        
    except FileNotFoundError:
        print(f"error: cannot find file '{data}'")
        return None
    except ValueError as e:
        print(f"input error: {str(e)}")
        return None
    except Exception as e:
        print(f"error when convert: {str(e)}")
        return None

# For a longtable or deluxetable LaTeX table
def dataframe_to_latex_long(data, output_file=None, caption="", label="",head2=""):
    try:
        # type
        if isinstance(data, pd.DataFrame):
            df = data.copy() 
        elif isinstance(data, str):
            df = pd.read_csv(data)
        else:
            raise ValueError("input must be pandas DataFrame or CSV path")
        
        """ df.columns = [col.strip().replace('_', '\\_').replace('%', '\\%').replace('#', '\\#') 
                     for col in df.columns] """
        df.columns = [format_column_name(col) for col in df.columns]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # df[col] = df[col].astype(str).str.replace('_', '\\_')
                df[col] = df[col].str.replace('%', '\\%')
                df[col] = df[col].str.replace('#', '\\#')
                df[col] = df[col].str.replace('&', '\\&')

        num_cols = len(df.columns)
        col_format = 'c' * num_cols
        
        latex_code = f"""\\startlongtable"""

        latex_code += f"""\n\\begin{{deluxetable}}{{{col_format}}}"""

        if label:
            latex_code += f"\n\\label{{{label}}}"
            
        latex_code += "\n\\centering"
        
        if caption:
            latex_code += f"\n\\tablecaption{{{caption}}}\n"
        
        # head
        colhead_format = [f'\\colhead{{{col}}}' for col in df.columns]
        header_row = '\\tablehead{' + ' & '.join(colhead_format) + ' \\\\'
        latex_code += header_row
        if head2:
            latex_code += f"\n{head2}"
        latex_code += '\n}\n'
        
        # data
        latex_code += """\\startdata\n"""
        for _, row in df.iterrows():
            data_row = ' & '.join([str(val) for val in row.values]) + ' \\\\'
            latex_code += data_row + '\n'
        
        latex_code += """\\enddata
\\end{deluxetable}"""
        
        # output
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"LaTeX already save to: {output_file}")
        else:
            print("LaTeX table code:")
            print("-" * 50)
            print(latex_code)
        
        return latex_code
        
    except FileNotFoundError:
        print(f"error: cannot find file '{data}'")
        return None
    except ValueError as e:
        print(f"input error: {str(e)}")
        return None
    except Exception as e:
        print(f"error when convert: {str(e)}")
        return None


def csv_to_latex(csv_file, output_file=None, caption="", label="", position="htbp"):
    return dataframe_to_latex(csv_file, output_file, caption, label, position)

def main():
    parser = argparse.ArgumentParser(description='from CSV to LaTeX table')
    parser.add_argument('csv_file', help='input CSV file path or pandas DataFrame')
    parser.add_argument('-o', '--output', help='output LaTeX file path')
    parser.add_argument('-c', '--caption', default='', help='table caption')
    parser.add_argument('-l', '--label', default='', help='table label')
    parser.add_argument('-p', '--position', default='htbp', help='table position parameter (default: htbp)')
    
    args = parser.parse_args()
    
    dataframe_to_latex(args.csv_file, args.output, args.caption, args.label, args.position)


def df_to_latex_simple(df, caption="", label=""):
    return dataframe_to_latex(df, caption=caption, label=label)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("CSV/DataFrame to LaTeX")
        print("\n methos:")
        print("python csv_to_latex.py <CSV file path> [options]")
        print("\n options:")
        print("  -o, --output    output LaTeX file path")
        print("  -c, --caption   table caption")
        print("  -l, --label     table label")
        print("  -p, --position  table position parameter (default: htbp)")
        print("\n example:")
        print("1. tranfer csv to latex:")
        print("   python csv_to_latex.py data.csv -o table.tex -c 'table' -l 'tab:data'")
        print("\n2. in code use pandas DataFrame:")
        print("   import pandas as pd")
        print("   df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})")
        print("   dataframe_to_latex(df, 'output.tex', 'my table', 'tab:mytable')")
        print("   # or use df_to_latex_simple:")
        print("   df_to_latex_simple(df, 'simple table', 'tab:simple')")
    else:
        main()

# For a simple LaTeX table without longtable or deluxetable
def dataframe_to_latex(data, output_file=None, caption="", label="", position="htbp",scale=""):
    try:
        # type
        if isinstance(data, pd.DataFrame):
            df = data.copy() 
        elif isinstance(data, str):
            df = pd.read_csv(data)
        else:
            raise ValueError("input must be pandas DataFrame or CSV path")
        
        df.columns = [format_column_name(col) for col in df.columns]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # df[col] = df[col].astype(str).str.replace('_', '\\_')
                df[col] = df[col].str.replace('%', '\\%')
                df[col] = df[col].str.replace('#', '\\#')
                df[col] = df[col].str.replace('&', '\\&')
        
        num_cols = len(df.columns)
        col_format = '|' + 'c|' * num_cols
        
        latex_code = f"""\\begin{{table}}[{position}]
\\centering"""
        
        if caption:
            latex_code += f"\n\\caption{{{caption}}}"
        
        if label:
            latex_code += f"\n\\label{{{label}}}"
            
        if scale:
            latex_code += f"\n\\scalebox{{{scale}}}{{ "
        
        latex_code += f"""
\\begin{{tabular}}{{{col_format}}}
\\hline
"""
        
        # head
        header_row = ' & '.join(df.columns) + ' \\\\'
        latex_code += header_row + '\n\\hline\n'
        
        # data
        for _, row in df.iterrows():
            data_row = ' & '.join([str(val) for val in row.values]) + ' \\\\'
            latex_code += data_row + '\n'
        
        latex_code += """\\hline
\\end{tabular}"""

        if scale:
            latex_code += """\n} """
            
        latex_code += """
\\end{table}"""
        
        # output
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"LaTeX already save to: {output_file}")
        else:
            print("LaTeX table code:")
            print("-" * 50)
            print(latex_code)
        
        return latex_code
        
    except FileNotFoundError:
        print(f"error: cannot find file '{data}'")
        return None
    except ValueError as e:
        print(f"input error: {str(e)}")
        return None
    except Exception as e:
        print(f"error when convert: {str(e)}")
        return None


def csv_to_latex(csv_file, output_file=None, caption="", label="", position="htbp"):
    return dataframe_to_latex(csv_file, output_file, caption, label, position)

def main():
    parser = argparse.ArgumentParser(description='from CSV to LaTeX table')
    parser.add_argument('csv_file', help='input CSV file path or pandas DataFrame')
    parser.add_argument('-o', '--output', help='output LaTeX file path')
    parser.add_argument('-c', '--caption', default='', help='table caption')
    parser.add_argument('-l', '--label', default='', help='table label')
    parser.add_argument('-p', '--position', default='htbp', help='table position parameter (default: htbp)')
    
    args = parser.parse_args()
    
    dataframe_to_latex(args.csv_file, args.output, args.caption, args.label, args.position)


def df_to_latex_simple(df, caption="", label=""):
    return dataframe_to_latex(df, caption=caption, label=label)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("CSV/DataFrame to LaTeX")
        print("\n methos:")
        print("python csv_to_latex.py <CSV file path> [options]")
        print("\n options:")
        print("  -o, --output    output LaTeX file path")
        print("  -c, --caption   table caption")
        print("  -l, --label     table label")
        print("  -p, --position  table position parameter (default: htbp)")
        print("\n example:")
        print("1. tranfer csv to latex:")
        print("   python csv_to_latex.py data.csv -o table.tex -c 'table' -l 'tab:data'")
        print("\n2. in code use pandas DataFrame:")
        print("   import pandas as pd")
        print("   df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})")
        print("   dataframe_to_latex(df, 'output.tex', 'my table', 'tab:mytable')")
        print("   # or use df_to_latex_simple:")
        print("   df_to_latex_simple(df, 'simple table', 'tab:simple')")
    else:
        main()
        
def format_column_name(col):
    col = col.strip()
    
    """ if 'DM_MW, ISM' in col:
        col = col.replace('DM_MW, ISM', r'$\DM_{\rm MW,ISM}$')
    else:
        col = col.replace('_', r'\_') """
    
    # col = col.replace('_', r'\_')
    col = col.replace('%', '\\%')
    col = col.replace('#', '\\#')
    
    return col