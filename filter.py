import pandas as pd
import re

class KeywordFilter:

    def __init__(self):
        pass

    def filter(self, list_of_df, keyword):
        filtered = []
        for page in list_of_df:
            df_per_page = []
            for df in page:
                row, col = -1, -1

                #check if keyword is in row/col names
                row_names = list(df.index)
                col_names = list(df.columns)
                row_names = [el if isinstance(el, str) else str(el) for el in row_names]
                col_names = [el if isinstance(el, str) else str(el) for el in col_names]

                for row_name in row_names:
                    row_name_lowered = [s.lower() for s in row_name.split()]
                    if keyword.lower() in row_name_lowered:
                        row = row_name_lowered.index(keyword.lower())
                
                for col_name in col_names:
                    col_name_lowered = [s.lower() for s in col_name.split()]
                    if keyword.lower() in col_name_lowered:
                        col = col_name_lowered.index(keyword.lower())
                
                if row != -1 or col != -1:
                    df_per_page.append((row,col))
                else:
                    #TODO: deal with multiple occurrences of the same keyword in a df
                    exit_loop = False
                    for row_idx in range(df.shape[0]):
                        for col_idx in range(df.shape[1]):
                            splitted = [s.lower() for s in re.split(' |\n', str(df.iat[row_idx,col_idx]))]
                            if keyword in [s.lower() for s in splitted]:
                                row = row_idx
                                col = col_idx
                                exit_loop = True
                                break
                        if exit_loop:
                            break
                    df_per_page.append((row,col))
            filtered.append(df_per_page)
        return filtered  
