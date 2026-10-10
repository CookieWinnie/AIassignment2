"""Build the Word report from the same Markdown used for the editable text report.

Run with the Codex bundled document Python runtime, not the ML environment.
"""
import sys
sys.dont_write_bytecode = True
import re
import argparse
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image

ROOT = Path(__file__).resolve().parent


def formatted_paragraph(doc, text):
    p = doc.add_paragraph()
    if text.startswith('Reflection draft for the student'):
        p.paragraph_format.keep_with_next = True
    # Formula images avoid the delimiter substitution observed in native math rendering.
    expressions = {
        'P(w|D) = P(D|w)P(w) / P(D)': 0,
        'P(y|x,D) = integral P(y|x,w)P(w|D) dw': 1,
        'p_bar = (1/T) sum_t p_t': 2,
        '(1/T) sum_t (p_t - p_bar)^2': 3,
        'H(p_bar) = -sum_k p_bar_k log(p_bar_k)': 4,
        'argmax(p_bar)': 5,
        'max(p_bar)': 6,
    }
    pattern = '(' + '|'.join(re.escape(s) for s in expressions) + ')'
    for part in re.split(pattern, text):
        if part in expressions:
            path = ROOT / f'report_material/equations/equation_{expressions[part]}.png'
            with Image.open(path) as img:
                width = img.width / 300
            shape = p.add_run().add_picture(str(path), width=Inches(width))
            shape._inline.docPr.set('descr', 'Mathematical equation '+part)
        else:
            p.add_run(part)
    return p


def add_table(doc, lines):
    rows = [[s.strip() for s in line.strip('|').split('|')] for line in lines]
    rows = [rows[0]] + rows[2:]
    table = doc.add_table(rows=0, cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    width = 6.45 / len(rows[0])
    if len(rows[0]) == 2:
        widths = [2.0, 4.45]
    else:
        widths = [width] * len(rows[0])
    for i, values in enumerate(rows):
        cells = table.add_row().cells
        for j, value in enumerate(values):
            cell = cells[j]
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = value
            properties = cell._tc.get_or_add_tcPr()
            borders = OxmlElement('w:tcBorders')
            for edge in ['top', 'left', 'bottom', 'right']:
                border = OxmlElement('w:'+edge)
                for attr, val in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
                    border.set(qn('w:'+attr), val)
                borders.append(border)
            properties.append(borders)
            margins = OxmlElement('w:tcMar')
            for edge in ['top', 'left', 'bottom', 'right']:
                margin = OxmlElement('w:'+edge)
                margin.set(qn('w:w'), '60')
                margin.set(qn('w:type'), 'dxa')
                margins.append(margin)
            properties.append(margins)
            if i == 0 or i % 2 == 0:
                shade = OxmlElement('w:shd')
                shade.set(qn('w:fill'), 'E7EDF3' if i == 0 else 'F5F5F5')
                properties.append(shade)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if len(rows[0]) == 2 else WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.bold = i == 0
        props = table.rows[-1]._tr.get_or_add_trPr()
        if i == 0:
            props.append(OxmlElement('w:tblHeader'))
        props.append(OxmlElement('w:cantSplit'))
    doc.add_paragraph().paragraph_format.space_after = Pt(3)


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    source = ROOT / 'report_material/final_report.md'
    if not source.exists():
        raise FileNotFoundError('Run main.py successfully before building the Word report')
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(0.8)
    section.left_margin = section.right_margin = Inches(0.9)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2', 'Caption']:
        style = doc.styles[name]
        style.font.name = 'Calibri'
        style.font.color.rgb = RGBColor(0, 0, 0)
    for border in list(doc.styles.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
    normal = doc.styles['Normal']
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.space_after = Pt(7)
    for name, size in [('Title', 24), ('Heading 1', 16), ('Heading 2', 13)]:
        doc.styles[name].font.size = Pt(size)
        doc.styles[name].paragraph_format.keep_with_next = True
    identity_line = next((line for line in source.read_text(encoding='utf-8').splitlines() if line.startswith('Student Name: ')), '')
    doc.core_properties.author = identity_line.partition(': ')[2]
    doc.core_properties.last_modified_by = ''
    doc.core_properties.title = 'MNIST Classification and Uncertainty with Monte Carlo Dropout'
    doc.core_properties.subject = 'CISC3024 AI Assignment 2'
    lines = source.read_text(encoding='utf-8').splitlines()
    i, first = 0, True
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('# '):
            if first:
                doc.add_paragraph(line[2:], 'Title')
                first = False
            else:
                heading = doc.add_heading(line[2:], level=1)
                if line.startswith('# 1 '):
                    heading.paragraph_format.page_break_before = True
        elif line.startswith('## '):
            doc.add_heading(line[3:], level=2)
        elif line.startswith('|'):
            chunk = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                chunk.append(lines[i].strip())
                i += 1
            add_table(doc, chunk)
            continue
        elif line.startswith('!['):
            match = re.match(r'!\[(.*?)\]\((.*?)\)', line)
            picture = (source.parent / match.group(2)).resolve()
            p = doc.add_paragraph()
            p.paragraph_format.keep_with_next = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run()
            width = 5.5 if picture.name == 'confusion_matrix.png' else 6.4
            shape = run.add_picture(str(picture), width=Inches(width))
            shape._inline.docPr.set('descr', match.group(1))
        elif line.startswith('Figure '):
            doc.add_paragraph(line, 'Caption')
        else:
            formatted_paragraph(doc, line)
        i += 1
    destination = ROOT / 'report_material/final_report.docx'
    doc.save(destination)
    # Verify identity, six main sections, four images, and the pending link.
    check = Document(destination)
    text = '\n'.join(p.text for p in check.paragraphs)
    assert 'Student Name: ' in text and 'Student Number: ' in text
    assert 'https://github.com/' in text or 'To be added after GitHub upload.' in text
    assert len(check.inline_shapes) == 11
    assert len([p for p in check.paragraphs if p.style.name == 'Heading 1']) == 7
    print(f'Created {destination}; identity, sections, images and webpage link verified.')


if __name__ == '__main__':
    main()
