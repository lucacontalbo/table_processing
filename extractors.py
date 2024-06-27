import tabula
import camelot
import pdfplumber
import pandas as pd

class TabulaExtractor:
    def __init__(self,):
        pass

    def extract(self, pdf_path, out_file=None):
        num_pages = 450
        for page in range(num_pages):
            print(f"*** parsing page {page} ***")
            try:
                dfs = tabula.read_pdf(pdf_path, encoding='utf-8', pages=str(page+1))
            except:
                pass
                #raise ValueError("file path is non existent")
            if len(dfs) > 0:
                for i, df in enumerate(dfs):
                    df.to_csv(f"{out_file}{page}_{i}.csv", index=False)
        

class CamelotExtractor:
    def __init__(self):
        pass

    def extract(self, pdf_path, out_file=None):
        num_pages = 450
        for page in range(num_pages):
            print(f"*** parsing page {page} ***")
            try:
                tables = camelot.read_pdf(pdf_path, pages=str(page))
            except:
                pass
            print(tables)
            if len(tables) > 0:
                for i, table in enumerate(tables):
                    df = table.df
                    df.to_csv(f"{out_file}{page}_{i}.csv", index=False)

class PdfplumberExtractor:
    def __init__(self):
        pass

    def extract(self, pdf_path, out_file=None):
        pdf = pdfplumber.open(pdf_path)
        for i in range(len(pdf.pages)):
            print(f"*** parsing page {i} ***")
            page = pdf.pages[i]
            pages = page.extract_tables()
            for j, p in enumerate(pages):
                if p is not None:
                    df = pd.DataFrame(p)
                    print(df.head())
                    df.to_csv(f"{out_file}{i}_{j}.csv", index=False)

class TabletransformerExtractor:
    def __init__(self):
        pass

    def extract(self, pdf_path, out_file=None):
        pass