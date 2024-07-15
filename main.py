import argparse
from extractors import TabulaExtractor, CamelotExtractor, PdfplumberExtractor, TabletransformerExtractor, TabletransformerExtractorWithPdfPlumber
from stats import StatsCalculator
from filter import KeywordFilter
import os

switch = {
    "tabula": TabulaExtractor,
    "camelot": CamelotExtractor,
    "pdfplumber": PdfplumberExtractor,
    "tabletransformer": TabletransformerExtractor,
    "tt_p": TabletransformerExtractorWithPdfPlumber
}

def init_args():
    parser = argparse.ArgumentParser(prog='BPER Table Extractor', description='Extract tables from companies non financial statements')

    parser.add_argument('--method', choices=["tabula", "camelot", "pdfplumber", "tabletransformer", "tt_p"], default="tabula", type=str,
                        help='extraction method')
    parser.add_argument('--pdf', type=str, required=False, default='',
                        help='relative URI of the pdf file to analyze')
    parser.add_argument('--pdfs', type=str, required=False, default='',
                        help='relative URI of the dir containing the pdf files to analyze')
    parser.add_argument('--stats', action="store_true", default=False,
                        help='Print the statistics of the tables')
    parser.add_argument('--stats_dir', type=str, required=False, default='',
                        help='Path of the directory from where the dataset statistics are calculated')
    parser.add_argument('--keyword', type=str, required=False, default='',
                        help='keyword to search inside the DataFrames')
    
    args = vars(parser.parse_args())
    if args["stats"] ^ (len(args["stats_dir"]) != 0):
        parser.error("--stats and --stats_dir must be given together")

    return args

if __name__ == "__main__":
    args = init_args()

    if args["stats"]:
        stats_calculator = StatsCalculator(args["stats_dir"])
        stats_calculator.run()
    else:
        extractor = switch[args["method"]]()
        keyword_filter = KeywordFilter()
        if args["pdfs"]:
            for file in os.listdir(args["pdfs"]):
                if not file.endswith(".pdf"):
                    continue
                print(f"----------- PARSING PDF {file} -----------")
                new_dir = os.path.join("results", args["method"], '.'.join(file.split(".")[:-1]))
                os.makedirs(new_dir, exist_ok=True)
                extractor.run(os.path.join(args["pdfs"], file), out_file=os.path.join(new_dir,"output"))
        else:
            tables_per_page, tables_per_df, pages_list = extractor.run(args["pdf"], out_file="./output")
            if args["keyword"] != '':
                for page in tables_per_df:
                    for df in page:
                        print(df.head())
                        break
                    break
                filtered_dfs = keyword_filter.filter(tables_per_df, args["keyword"])
                for i in range(10):
                    print(i)
            