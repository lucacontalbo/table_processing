import tabula

class TabulaExtractor:
    def __init__(self,):
        pass

    def extract(self, pdf_path, out_file=None):
        counter = []
        if out_file is not None:
                for page in range(440):
                    print(f"*** parsing page {page} ***")
                    try:
                        dfs = tabula.read_pdf(pdf_path, encoding='utf-8', pages=str(page+1))
                    except:
                        pass
                        #raise ValueError("file path is non existent")
                    print(dfs)
                    if len(dfs) > 0:
                        counter.append(page)
                        for i, df in enumerate(dfs):
                            df.to_csv(f"{out_file}{counter[-1]}_{i}.csv", index=False)

        else:
            raise NotImplementedError("")
        print(f"Pages of tables extracted: {counter}")
        

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