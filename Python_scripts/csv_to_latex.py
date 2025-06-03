import pandas as pd
import argparse
import sys

def dataframe_to_latex(data, output_file=None, caption="", label="", position="htbp"):
    try:
        # type
        if isinstance(data, pd.DataFrame):
            df = data.copy() 
        elif isinstance(data, str):
            df = pd.read_csv(data)
        else:
            raise ValueError("input must be pandas DataFrame or CSV path")
        
        df.columns = [col.strip().replace('_', '\\_').replace('%', '\\%').replace('#', '\\#') 
                     for col in df.columns]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('_', '\\_')
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
\\end{tabular}
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