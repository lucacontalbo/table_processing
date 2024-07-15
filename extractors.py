import tabula
import camelot
import pdfplumber
import functools
import torch
import csv
import easyocr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection

from PIL import Image, ImageDraw
from torchvision import transforms
from matplotlib.patches import Patch
from tqdm import tqdm
from copy import deepcopy
from pdf2image import convert_from_path, convert_from_bytes, pdfinfo_from_path


class GeneralExtractor:
    def __init__(self,):
        pass

    def get_splits_bbox(self, curve, split_delims, axis=1):
        curr_bbox = {"x0": curve["x0"], "top": curve["top"], "x1": curve["x1"], "bottom": curve["bottom"]}
        if axis == 1:
            splits = []
            for split_val in split_delims:
                curr_bbox["x1"] = split_val
                bbox = [curr_bbox["x0"], curr_bbox["top"], curr_bbox["x1"], curr_bbox["bottom"]]
                splits.append(deepcopy(bbox))
                curr_bbox["x0"] = curr_bbox["x1"]
        return splits

    def get_curve(self, curve):
        bbox = [0,0,0,0]
        bbox[0] = curve["x0"]
        bbox[1] = curve["top"]
        bbox[2] = curve["x1"]
        bbox[3] = curve["bottom"]
        return bbox

    def expand_table(self, bbox, max_width, max_height, direction="top"):
        if direction == "top":
            bbox[1] -= 50
            bbox[0] -= 5
            bbox[2] += 5
            bbox[3] += 5
            bbox[0], bbox[1] = max(0, bbox[0]), max(0, bbox[1])
            bbox[2], bbox[3] = min(bbox[2], max_width), min(bbox[3], max_height)
        return bbox

    def extract_additional_hcols(self, pdf, table):
        bbox = list(table.bbox)
        bbox = self.expand_table(bbox, pdf.width, pdf.height, direction="top")
        new_pdf = pdf.crop(tuple(bbox))
        result = []
        if len(new_pdf.curves) != 0:
            if len(new_pdf.curves) > 1:
                print("the number of curves is higher than 1. Defaulting to the first curve...")
            curve = new_pdf.curves[0]
            extracted_curve = self.get_curve(curve)
            ncols = len(table.rows[0].cells)
            split_size = (curve["x1"] - curve["x0"]) / ncols
            split_delims = []
            for i in range(ncols):
                split_delims.append(curve["x0"] + (i+1)*split_size)
            delimited_bboxes = self.get_splits_bbox(curve, split_delims)
            words_in_pdf = set(pdf.extract_text().lower().split())
            for bbox in delimited_bboxes:
                cropped_pdf = pdf.crop(tuple(bbox))
                for word in cropped_pdf.extract_text().split():
                    if word.lower() not in words_in_pdf:
                        return []
                result.append(cropped_pdf.extract_text())
        return result #test if, when the cell is cropped even with text in between, it gives out wrong words

    def run(self, pdf_path, out_file=None, extract_hcols=True):
        tables_per_page, tables_per_df, pages_list = self.extract(pdf_path, out_file)
        if extract_hcols:
            for i, (tables, tables_df, page) in enumerate(zip(tables_per_page, tables_per_df, pages_list)):
                for j, (table, table_df) in enumerate(zip(tables, tables_df)):
                    new_hcols = self.extract_additional_hcols(page, table)
                    if len(new_hcols) > 0:
                        table_df.columns = new_hcols
                    table_df.to_csv(f"{out_file}{i}_{j}.csv", index=False)
        return tables_per_page, tables_per_df, pages_list

class TabulaExtractor(GeneralExtractor):
    def __init__(self,):
        super(TabulaExtractor, self).__init__()

    def extract(self, pdf_path, out_file=None):
        num_pages = 450
        tables_df = []
        for page in range(num_pages):
            print(f"*** parsing page {page} ***")
            tables = []
            try:
                dfs = tabula.read_pdf(pdf_path, encoding='utf-8', pages=str(page+1))
            except:
                print("error in loading pdf")
                #raise ValueError("file path is non existent")
            if len(dfs) > 0:
                for i, df in enumerate(dfs):
                    tables.append(df)
                    df.to_csv(f"{out_file}{page}_{i}.csv", index=False)
            tables_df.append(tables)
        return tables_df
        

class CamelotExtractor(GeneralExtractor):
    def __init__(self):
        super(CamelotExtractor, self).__init__()

    def extract(self, pdf_path, out_file=None):
        num_pages = 450
        tables_df = []
        for page in range(num_pages):
            print(f"*** parsing page {page} ***")
            tables = []
            try:
                dfs = camelot.read_pdf(pdf_path, pages=str(page))
            except:
                print("error in loading pdf")
                pass
            if len(dfs) > 0:
                for i, df in enumerate(dfs):
                    df = df.df
                    df.to_csv(f"{out_file}{page}_{i}.csv", index=False)
                    tables.append(df)
            tables_df.append(tables)
        return tables_df


class PdfplumberExtractor(GeneralExtractor):
    def __init__(self):
        super(PdfplumberExtractor, self).__init__()

    def read_pdf(self, pdf_path):
        return pdfplumber.open(pdf_path)

    def extract(self, pdf_path, out_file=None):
        pdf = self.read_pdf(pdf_path)
        tables_pdf_tot = []
        tables_df_tot = []
        pages_list = []
        for i in range(len(pdf.pages)):
            if i != 95: continue
            print(f"*** parsing page {i} ***")
            page = pdf.pages[i]
            pages_list.append(page)
            pages, pages_tables = page.extract_tables(), page.find_tables()
            tables_pdf = []
            tables_df = []
            for j, (p, pt) in enumerate(zip(pages, pages_tables)):
                if p is not None:
                    df = pd.DataFrame(p)
                    tables_pdf.append(pt)
                    tables_df.append(df)
                    
                    df.to_csv(f"{out_file}{i}_{j}.csv", index=False)
            tables_pdf_tot.append(tables_pdf)
            tables_df_tot.append(tables_df)
        return tables_pdf_tot, tables_df_tot, pages_list
    
    def get_image_bbox(self, pdf_path):
        pdf = pdfplumber.open(pdf_path)
        images = []

        for i in range(len(pdf.pages)):
            tables = pdf.pages[i].find_tables()
            for j,p in enumerate(tables):
                if p is not None:
                    bbox = list(p.bbox)
                    bbox[1] -= 50
                    bbox[0] -= 5
                    bbox[2] += 5
                    bbox[3] += 5
                    bbox[0], bbox[1] = max(0, bbox[0]), max(0, bbox[1])
                    bbox[2], bbox[3] = min(bbox[2], pdf.pages[i].width), min(bbox[3], pdf.pages[i].height)

                    images.append(pdf.pages[i].crop(tuple(bbox)).to_image().original)
                else:
                    images.append(-1)
        
        return images

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image

class TabletransformerExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_detection = self.load_model(AutoModelForObjectDetection, "microsoft/table-transformer-detection").to(self.device)
        self.model_structure = self.load_model(TableTransformerForObjectDetection, "microsoft/table-structure-recognition-v1.1-all", timm=True).to(self.device)
        self.ocr = OCR()
        self.visualize = False

    @functools.cache
    def load_model(self, arch, model_name, timm=False):
        if not timm:
            return arch.from_pretrained(model_name, revision="no_timm")
        else:
            return arch.from_pretrained(model_name)

    def preprocess_image(self, image, resize=800):
        detection_transform = transforms.Compose([
            MaxResize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        pixel_values = detection_transform(image).unsqueeze(0)
        pixel_values = pixel_values.to(self.device)

        return pixel_values

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def outputs_to_objects(self, outputs, img_size, id2label):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})

        return objects

    def fig2img(self, fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img


    def visualize_detected_tables(self, img, det_tables, out_path=None):
        plt.imshow(img, interpolation="lanczos")
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        ax = plt.gca()

        for det_table in det_tables:
            bbox = det_table['bbox']

            if det_table['label'] == 'table':
                facecolor = (1, 0, 0.45)
                edgecolor = (1, 0, 0.45)
                alpha = 0.3
                linewidth = 2
                hatch='//////'
            elif det_table['label'] == 'table rotated':
                facecolor = (0.95, 0.6, 0.1)
                edgecolor = (0.95, 0.6, 0.1)
                alpha = 0.3
                linewidth = 2
                hatch='//////'
            else:
                continue

            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                        edgecolor='none',facecolor=facecolor, alpha=0.1)
            ax.add_patch(rect)
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                        edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
            ax.add_patch(rect)
            rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                        edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
            ax.add_patch(rect)

        plt.xticks([], [])
        plt.yticks([], [])

        legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                                    label='Table', hatch='//////', alpha=0.3),
                            Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                                    label='Table (rotated)', hatch='//////', alpha=0.3)]
        plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                        fontsize=10, ncol=2)
        plt.gcf().set_size_inches(10, 10)
        plt.axis('off')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', dpi=150)

        return fig


    def objects_to_crops(self, img, tokens, objects, class_thresholds, padding=0):
        """
        Process the bounding boxes produced by the table detection model into
        cropped table images and cropped tokens.
        """

        table_crops = []
        for obj in objects:
            if obj['score'] < class_thresholds[obj['label']]:
                continue

            cropped_table = {}

            bbox = obj['bbox']
            #bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
            bbox = [bbox[0], bbox[1]-padding, bbox[2], bbox[3]]

            cropped_img = img.crop(bbox)

            table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
            for token in table_tokens:
                token['bbox'] = [token['bbox'][0]-bbox[0],
                                token['bbox'][1]-bbox[1],
                                token['bbox'][2]-bbox[0],
                                token['bbox'][3]-bbox[1]]

            # If table is predicted to be rotated, rotate cropped image and tokens/words:
            if obj['label'] == 'table rotated':
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token['bbox']
                    bbox = [cropped_img.size[0]-bbox[3]-1,
                            bbox[0],
                            cropped_img.size[0]-bbox[1]-1,
                            bbox[2]]
                    token['bbox'] = bbox

            cropped_table['image'] = cropped_img
            cropped_table['tokens'] = table_tokens

            table_crops.append(cropped_table)

        return table_crops


    def extract(self, pdf_path, out_file=None):

        print("Processing pdf pages as images. It may take a while...")
        info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=None)

        maxPages = info["Pages"]
        #images = convert_from_path(pdf_path, 10)
        images = []

        for page in range(1, maxPages+1, 10):
            images.extend(convert_from_path(pdf_path, dpi=200, first_page=page, last_page = min(page+10-1,maxPages)))

        print("conversion done.")

        for i, image in enumerate(images):
            #image = self.load_image(image_path)
            pixel_values = self.preprocess_image(image)

            with torch.no_grad():
                outputs = self.model_detection(pixel_values)
            
            id2label = self.model_detection.config.id2label
            id2label[len(self.model_detection.config.id2label)] = "no object"

            objects = self.outputs_to_objects(outputs, image.size, id2label)

            tokens = []
            detection_class_thresholds = {
                "table": 0.5,
                "table rotated": 0.5,
                "no object": 10
            }

            tables_crops = self.objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=500)
            for j in range(len(tables_crops)):
                cropped_table = tables_crops[j]['image'].convert("RGB")

                pixel_values = self.preprocess_image(cropped_table, resize=1000)
                with torch.no_grad():
                    outputs = self.model_structure(pixel_values)

                structure_id2label = self.model_structure.config.id2label
                structure_id2label[len(structure_id2label)] = "no object"

                cells = self.outputs_to_objects(outputs, cropped_table.size, structure_id2label)

                if self.visualize:
                    cropped_table_visualized = cropped_table.copy()
                    draw = ImageDraw.Draw(cropped_table_visualized)

                    for cell in cells:
                        draw.rectangle(cell["bbox"], outline="red")

                cell_coordinates = self.ocr.get_cell_coordinates_by_row(cells)

                data = self.ocr.apply_ocr(cell_coordinates, cropped_table)

                with open(f"{out_file}{i}_{j}.csv",'w') as result_file:
                    wr = csv.writer(result_file, dialect='excel')

                    for _, row_text in data.items():
                        wr.writerow(row_text)

class OCR:

    def __init__(self, lang="eng"):
        self.reader = self.load_model(lang)

    @functools.cache
    def load_model(self, lang):
        return easyocr.Reader(['it'])

    def get_cell_coordinates_by_row(self, table_data):
        rows = [entry for entry in table_data if entry['label'] == 'table row']
        columns = [entry for entry in table_data if entry['label'] == 'table column']

        rows.sort(key=lambda x: x['bbox'][1])
        columns.sort(key=lambda x: x['bbox'][0])

        def find_cell_coordinates(row, column):
            cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
            return cell_bbox

        cell_coordinates = []

        for row in rows:
            row_cells = []
            for column in columns:
                cell_bbox = find_cell_coordinates(row, column)
                row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

            row_cells.sort(key=lambda x: x['column'][0])

            cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

        cell_coordinates.sort(key=lambda x: x['row'][1])

        return cell_coordinates
    
    def apply_ocr(self, cell_coordinates, cropped_table):
        data = dict()
        max_num_columns = 0
        for idx, row in enumerate(tqdm(cell_coordinates)):
            row_text = []
            for cell in row["cells"]:
                cell_image = np.array(cropped_table.crop(cell["cell"]))
                result = self.reader.readtext(np.array(cell_image))
                if len(result) > 0:
                    text = " ".join([x[1] for x in result])
                    row_text.append(text)

            if len(row_text) > max_num_columns:
                max_num_columns = len(row_text)

            data[idx] = row_text

        for row, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
            data[row] = row_data

        return data
    

class TabletransformerExtractorWithPdfPlumber(TabletransformerExtractor):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.visualize = True

        self.pdfplumber = PdfplumberExtractor()

    def extract(self, pdf_path, out_file=None):

        print("Processing pdf pages as images. It may take a while...")
        info = pdfinfo_from_path(pdf_path, userpw=None, poppler_path=None)

        maxPages = info["Pages"]
        #images = convert_from_path(pdf_path, 10)
        images = self.pdfplumber.get_image_bbox(pdf_path)
        """for im in images:
            im.show()"""

        #for page in tqdm(range(1, maxPages+1, 10)):
        #    images.extend(convert_from_path(pdf_path, dpi=200, first_page=page, last_page = min(page+10-1,maxPages)))

        #print(images)

        print("conversion done.")

        for i, image in enumerate(images):
            if image == -1:
                continue
            #image = self.load_image(image_path)
            pixel_values = self.preprocess_image(image)

            with torch.no_grad():
                outputs = self.model_detection(pixel_values)
            
            id2label = self.model_detection.config.id2label
            id2label[len(self.model_detection.config.id2label)] = "no object"

            objects = self.outputs_to_objects(outputs, image.size, id2label)

            tokens = []
            detection_class_thresholds = {
                "table": 0.5,
                "table rotated": 0.5,
                "no object": 10
            }

            tables_crops = self.objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=500)
            for j in range(len(tables_crops)):
                cropped_table = tables_crops[j]['image'].convert("RGB")

                pixel_values = self.preprocess_image(cropped_table, resize=1000)
                with torch.no_grad():
                    outputs = self.model_structure(pixel_values)

                structure_id2label = self.model_structure.config.id2label
                structure_id2label[len(structure_id2label)] = "no object"

                cells = self.outputs_to_objects(outputs, cropped_table.size, structure_id2label)

                if self.visualize:
                    cropped_table_visualized = cropped_table.copy()
                    draw = ImageDraw.Draw(cropped_table_visualized)

                    for cell in cells:
                        draw.rectangle(cell["bbox"], outline="red")

                cell_coordinates = self.ocr.get_cell_coordinates_by_row(cells)

                data = self.ocr.apply_ocr(cell_coordinates, cropped_table)

                with open(f"{out_file}{i}_{j}.csv",'w') as result_file:
                    wr = csv.writer(result_file, dialect='excel')

                    for _, row_text in data.items():
                        wr.writerow(row_text)