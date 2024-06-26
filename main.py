import argparse
from extractors import TabulaExtractor, CamelotExtractor, PdfplumberExtractor, TabletransformerExtractor

switch = {
    "tabula": TabulaExtractor,
    "camelot": [],
    "pdfplumber": [],
    "tabletransformer": []
}

def init_args():
    parser = argparse.ArgumentParser(prog='BPER Table Extractor', description='Extract tables from companies non financial statements')

    parser.add_argument('--method', choices=["tabula", "camelot", "pdfplumber", "tabletransformer"], default="tabula", type=str,
                        help='extraction method')
    parser.add_argument('--pdf', type=str, required=True,
                        help='relative URI of the pdf file to analyze')
    
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    args = init_args()

    extractor = switch[args["method"]]()
    extractor.extract(args["pdf"], out_file="./output")