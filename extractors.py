import tabula

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
        pass

class PdfplumberExtractor:
    def __init__(self):
        pass

    def extract(self, pdf_path, out_file=None):
        pass

class TabletransformerExtractor:
    def __init__(self):
        pass

    def extract(self, pdf_path, out_file=None):
        pass