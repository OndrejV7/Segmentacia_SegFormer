"""
Generates a Word (.docx) report for the diploma thesis.

Compares two models on the same test set:
  - main      = "v3" (full dataset incl. Dub_praskliny_a + hrce_mixed)
  - ablation  = "v3_ablation" (only kmen1-10 + Dub_1-10)

Sections:
  1. Introduction -- model and dataset description
  2. Evaluation metrics -- definitions and formulas
  3. Per-class results table -- main model
  4. Ablation study -- comparison main vs ablation
  5. Interpretation commentary

Requires:
    pip install python-docx

Usage:
    python export_report_word.py
    python export_report_word.py --main metrics_v3.json --ablation metrics_v3_ablation.json
    python export_report_word.py --main metrics_v3.json   # bez ablation sekcie
"""

import argparse
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

DATA_DIR = Path(__file__).parent
CLASS_NAMES = ["Drevo", "Kora", "Nezdrava_hrca", "Okolie", "Prasklina"]


# ── Formatting helpers ────────────────────────────────────────────────────
def add_heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def add_paragraph(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p


def add_formula(doc, formula_text, description=""):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(formula_text)
    run.italic = True
    run.font.size = Pt(12)
    if description:
        p2 = doc.add_paragraph()
        run2 = p2.add_run(description)
        run2.font.size = Pt(10)
        run2.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def shade_row(row, hex_color="D9D9D9"):
    for cell in row.cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_color)
        tcPr.append(shd)


def add_results_table(doc, metrics):
    per_class = metrics["per_class"]

    table = doc.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.rows[0].cells
    for i, h in enumerate(["Trieda", "IoU (%)", "Dice / F1 (%)", "Presnosť (%)", "Úplnosť (%)", "Podpora (px)"]):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
    shade_row(table.rows[0], "BDD7EE")

    for name in CLASS_NAMES:
        m = per_class[name]
        row = table.add_row().cells
        row[0].text = name
        row[1].text = f"{m['iou']*100:.2f}"
        row[2].text = f"{m['dice']*100:.2f}"
        row[3].text = f"{m['precision']*100:.2f}"
        row[4].text = f"{m['recall']*100:.2f}"
        row[5].text = f"{m['support']:,}"

    sep = table.add_row().cells
    for c in sep:
        c.text = ""
    shade_row(table.rows[-1], "F2F2F2")

    agg = table.add_row().cells
    agg[0].text = "Priemer (macro)"
    agg[0].paragraphs[0].runs[0].bold = True
    agg[1].text = f"{metrics['mean_iou']*100:.2f}"
    agg[2].text = f"{metrics['mean_dice']*100:.2f}"
    agg[3].text = f"{metrics['mean_precision']*100:.2f}"
    agg[4].text = f"{metrics['mean_recall']*100:.2f}"
    agg[5].text = ""
    shade_row(table.rows[-1], "E2EFDA")

    pa = table.add_row().cells
    pa[0].text = "Pixelová presnosť"
    pa[0].paragraphs[0].runs[0].bold = True
    pa[1].text = f"{metrics['pixel_accuracy']*100:.2f}"
    for i in range(2, 6):
        pa[i].text = ""
    shade_row(table.rows[-1], "E2EFDA")


def add_comparison_table(doc, main, ablation):
    """
    Tabulka porovnania dvoch modelov: pre kazdu triedu a metriku
    ukaze hodnoty oboch modelov a delta.
    """
    # Stlpce: Trieda | Metrika | v3 | v3_ablation | Delta
    metrics_to_show = [
        ("IoU (%)",      "iou"),
        ("Recall (%)",   "recall"),
        ("Precision (%)", "precision"),
    ]

    n_rows_per_class = len(metrics_to_show)
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.rows[0].cells
    for i, h in enumerate(["Trieda", "Metrika", "v3 (plný dataset)", "Ablation", "Δ"]):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
    shade_row(table.rows[0], "BDD7EE")

    for name in CLASS_NAMES:
        for j, (label, key) in enumerate(metrics_to_show):
            row = table.add_row().cells
            row[0].text = name if j == 0 else ""
            if j == 0:
                row[0].paragraphs[0].runs[0].bold = True
            row[1].text = label
            v_main = main["per_class"][name][key] * 100
            v_abl  = ablation["per_class"][name][key] * 100
            delta  = v_main - v_abl
            row[2].text = f"{v_main:.2f}"
            row[3].text = f"{v_abl:.2f}"
            sign = "+" if delta >= 0 else ""
            row[4].text = f"{sign}{delta:.2f}"
            # zvyrazni vyznamne pozitivne delty (ablation HORSI nez main = pridane data pomohli)
            if abs(delta) >= 1.0:
                run = row[4].paragraphs[0].runs[0]
                run.bold = True
                if delta > 0:
                    run.font.color.rgb = RGBColor(0x00, 0x70, 0x30)   # zelena
                else:
                    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)   # cervena
        # oddelenie medzi triedami
        sep = table.add_row().cells
        for c in sep:
            c.text = ""
        shade_row(table.rows[-1], "F8F8F8")

    # Aggregate row
    agg = table.add_row().cells
    agg[0].text = "Priemer (macro)"
    agg[0].paragraphs[0].runs[0].bold = True
    agg[1].text = "mIoU"
    v_main = main["mean_iou"] * 100
    v_abl  = ablation["mean_iou"] * 100
    delta  = v_main - v_abl
    agg[2].text = f"{v_main:.2f}"
    agg[3].text = f"{v_abl:.2f}"
    sign = "+" if delta >= 0 else ""
    agg[4].text = f"{sign}{delta:.2f}"
    shade_row(table.rows[-1], "E2EFDA")

    agg2 = table.add_row().cells
    agg2[0].text = ""
    agg2[1].text = "Mean Recall"
    v_main = main["mean_recall"] * 100
    v_abl  = ablation["mean_recall"] * 100
    delta  = v_main - v_abl
    agg2[2].text = f"{v_main:.2f}"
    agg2[3].text = f"{v_abl:.2f}"
    sign = "+" if delta >= 0 else ""
    agg2[4].text = f"{sign}{delta:.2f}"
    if abs(delta) >= 1.0:
        run = agg2[4].paragraphs[0].runs[0]
        run.bold = True
        if delta > 0:
            run.font.color.rgb = RGBColor(0x00, 0x70, 0x30)
        else:
            run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    shade_row(table.rows[-1], "E2EFDA")


def add_3way_comparison_table(doc, m_v3, m_abl, m_25d):
    """
    3-stlpcove porovnanie pre 2.5D experiment:
    Trieda | Metrika | v3 (full 2D) | v3_abl (2D) | v3_abl_25d (2.5D) | Δ 25D vs 2D
    """
    metrics_to_show = [
        ("IoU (%)",       "iou"),
        ("Recall (%)",    "recall"),
        ("Precision (%)", "precision"),
    ]

    table = doc.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.rows[0].cells
    for i, h in enumerate(["Trieda", "Metrika",
                           "v3 (2D, full)", "v3_abl (2D)", "v3_abl (2.5D)",
                           "Δ 2.5D vs 2D"]):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
    shade_row(table.rows[0], "BDD7EE")

    for name in CLASS_NAMES:
        for j, (label, key) in enumerate(metrics_to_show):
            row = table.add_row().cells
            row[0].text = name if j == 0 else ""
            if j == 0:
                row[0].paragraphs[0].runs[0].bold = True
            row[1].text = label

            v_full = m_v3 ["per_class"][name][key] * 100
            v_2d   = m_abl["per_class"][name][key] * 100
            v_25d  = m_25d["per_class"][name][key] * 100
            delta  = v_25d - v_2d

            row[2].text = f"{v_full:.2f}"
            row[3].text = f"{v_2d:.2f}"
            row[4].text = f"{v_25d:.2f}"
            sign = "+" if delta >= 0 else ""
            row[5].text = f"{sign}{delta:.2f}"
            if abs(delta) >= 1.0:
                run = row[5].paragraphs[0].runs[0]
                run.bold = True
                # 2.5D LEPSI ako 2D = zelena, HORSI = cervena
                if delta > 0:
                    run.font.color.rgb = RGBColor(0x00, 0x70, 0x30)
                else:
                    run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        sep = table.add_row().cells
        for c in sep:
            c.text = ""
        shade_row(table.rows[-1], "F8F8F8")

    # Aggregate rows
    agg = table.add_row().cells
    agg[0].text = "Priemer (macro)"
    agg[0].paragraphs[0].runs[0].bold = True
    agg[1].text = "mIoU"
    v_full = m_v3 ["mean_iou"] * 100
    v_2d   = m_abl["mean_iou"] * 100
    v_25d  = m_25d["mean_iou"] * 100
    delta  = v_25d - v_2d
    agg[2].text = f"{v_full:.2f}"
    agg[3].text = f"{v_2d:.2f}"
    agg[4].text = f"{v_25d:.2f}"
    sign = "+" if delta >= 0 else ""
    agg[5].text = f"{sign}{delta:.2f}"
    if abs(delta) >= 1.0:
        run = agg[5].paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = RGBColor(0x00, 0x70, 0x30) if delta > 0 \
                             else RGBColor(0xC0, 0x00, 0x00)
    shade_row(table.rows[-1], "E2EFDA")

    agg2 = table.add_row().cells
    agg2[0].text = ""
    agg2[1].text = "Mean Recall"
    v_full = m_v3 ["mean_recall"] * 100
    v_2d   = m_abl["mean_recall"] * 100
    v_25d  = m_25d["mean_recall"] * 100
    delta  = v_25d - v_2d
    agg2[2].text = f"{v_full:.2f}"
    agg2[3].text = f"{v_2d:.2f}"
    agg2[4].text = f"{v_25d:.2f}"
    sign = "+" if delta >= 0 else ""
    agg2[5].text = f"{sign}{delta:.2f}"
    if abs(delta) >= 1.0:
        run = agg2[5].paragraphs[0].runs[0]
        run.bold = True
        run.font.color.rgb = RGBColor(0x00, 0x70, 0x30) if delta > 0 \
                             else RGBColor(0xC0, 0x00, 0x00)
    shade_row(table.rows[-1], "E2EFDA")


def add_2d_vs_25d_vs_rgb_table(doc, m_2d, m_25d, m_rgb):
    """
    Tabulka pre porovnanie 2D vs 2.5D 5-kanal vs RGB encoding na rovnakom
    186-img teste. Stlpec Δ porovnava RGB voci 2D.
    """
    metrics_to_show = [
        ("IoU (%)",       "iou"),
        ("Recall (%)",    "recall"),
        ("Precision (%)", "precision"),
    ]

    table = doc.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.rows[0].cells
    for i, h in enumerate(["Trieda", "Metrika",
                           "2D ablation", "2.5D 5-kanál", "RGB encoding",
                           "Δ RGB vs 2D"]):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
    shade_row(table.rows[0], "BDD7EE")

    for name in CLASS_NAMES:
        for j, (label, key) in enumerate(metrics_to_show):
            row = table.add_row().cells
            row[0].text = name if j == 0 else ""
            if j == 0:
                row[0].paragraphs[0].runs[0].bold = True
            row[1].text = label

            v_2d  = m_2d ["per_class"][name][key] * 100
            v_25d = m_25d["per_class"][name][key] * 100
            v_rgb = m_rgb["per_class"][name][key] * 100
            delta = v_rgb - v_2d

            row[2].text = f"{v_2d:.2f}"
            row[3].text = f"{v_25d:.2f}"
            row[4].text = f"{v_rgb:.2f}"
            sign = "+" if delta >= 0 else ""
            row[5].text = f"{sign}{delta:.2f}"
            if abs(delta) >= 1.0:
                run = row[5].paragraphs[0].runs[0]
                run.bold = True
                run.font.color.rgb = RGBColor(0x00, 0x70, 0x30) if delta > 0 \
                                     else RGBColor(0xC0, 0x00, 0x00)
        sep = table.add_row().cells
        for c in sep:
            c.text = ""
        shade_row(table.rows[-1], "F8F8F8")

    # Aggregate
    for label, key in [("mIoU", "mean_iou"),
                       ("Mean Recall", "mean_recall"),
                       ("Pixel Accuracy", "pixel_accuracy")]:
        agg = table.add_row().cells
        agg[0].text = "Priemer (macro)" if label == "mIoU" else ""
        if label == "mIoU":
            agg[0].paragraphs[0].runs[0].bold = True
        agg[1].text = label
        v_2d  = m_2d [key] * 100
        v_25d = m_25d[key] * 100
        v_rgb = m_rgb[key] * 100
        delta = v_rgb - v_2d
        agg[2].text = f"{v_2d:.2f}"
        agg[3].text = f"{v_25d:.2f}"
        agg[4].text = f"{v_rgb:.2f}"
        sign = "+" if delta >= 0 else ""
        agg[5].text = f"{sign}{delta:.2f}"
        if abs(delta) >= 1.0:
            run = agg[5].paragraphs[0].runs[0]
            run.bold = True
            run.font.color.rgb = RGBColor(0x00, 0x70, 0x30) if delta > 0 \
                                 else RGBColor(0xC0, 0x00, 0x00)
        shade_row(table.rows[-1], "E2EFDA")


# ── Report content ────────────────────────────────────────────────────────
def build_report(doc, main, ablation, comp_3trunks=None, comp_3trunks_rgb=None):
    has_ablation = ablation is not None

    # ── Title ──
    title = doc.add_heading("Vyhodnotenie modelu sémantickej segmentácie", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # ── 1. Model a dataset ──
    add_heading(doc, "1  Model a testovacia množina", 1)
    add_paragraph(doc,
        "Na vyhodnotenie bol použitý model SegFormer-B2 trénovaný jednofázovo "
        "s kombinovanou stratou (30 % Cross-Entropy + 70 % Dice) na datasete "
        "prierezov kmeňov stromov. Trénovací dataset zahŕňa: smrek (kmen1–kmen10), "
        "dub (Dub_1–Dub_10), Dub_praskliny_a (50 ručne vybraných snímok cielene "
        "anotovaných pre detekciu pukliín) a hrce_mixed (snímky s nezdravými hrčami). "
        f"Testovanie prebehlo na samostatnej testovacej množine ({main['n_images']} snímok), "
        "ktorá nebola použitá počas tréningu ani validácie. "
        "Test obsahuje celé hold-out kmene (kmen4, kmen9, Dub_3b) a po 15 ručne "
        "vybraných snímok z hrce_mixed a Dub_praskliny_a."
    )
    doc.add_paragraph()
    add_paragraph(doc, f"Súbor váh hlavného modelu: {main['model']}.")
    if has_ablation:
        add_paragraph(doc,
            f"Pre ablation štúdiu (kapitola 4) bol natrénovaný druhý model "
            f"({ablation['model']}) na tých istých hyperparametroch, ale bez "
            f"datasetov Dub_praskliny_a a hrce_mixed v trénovacej množine."
        )
    doc.add_paragraph()

    add_paragraph(doc, "Rozdelenie tried (5 tried):", bold=True)
    class_desc = {
        "Drevo":         "zdravé drevo kmeňa",
        "Kora":          "kôra kmeňa",
        "Nezdrava_hrca": "nezdravá hrča (poškodené tkanivo)",
        "Okolie":        "pozadie / okolie kmeňa",
        "Prasklina":     "trhliny a praskliny v dreve",
    }
    for name, desc in class_desc.items():
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(f"{name}")
        run.bold = True
        p.add_run(f" — {desc}")

    doc.add_paragraph()

    # ── 2. Metriky ──
    add_heading(doc, "2  Hodnotiace metriky", 1)
    add_paragraph(doc,
        "Výkonnosť modelu bola hodnotená pomocou nasledujúcich metrík vypočítaných "
        "z matice zámen (confusion matrix). Pre každú triedu c sa definujú:"
    )
    bullet_items = [
        "TP (true positive)  — pixely správne zaradené do triedy c",
        "FP (false positive) — pixely iných tried nesprávne zaradené do triedy c",
        "FN (false negative) — pixely triedy c nesprávne zaradené do inej triedy",
        "TN (true negative)  — pixely iných tried správne nezaradené do triedy c",
    ]
    for item in bullet_items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(item)
    doc.add_paragraph()

    add_heading(doc, "2.1  IoU — Priesečník nad zjednotením (Jaccard index)", 2)
    add_paragraph(doc,
        "IoU meria mieru prekryvu predikovanej a skutočnej segmentačnej masky. "
        "Hodnota 1,0 znamená dokonalú zhodu, 0,0 znamená žiadny prekryv."
    )
    add_formula(doc, "IoU(c) = TP / (TP + FP + FN)",
                "kde TP, FP, FN sú počty pixelov pre triedu c")
    add_formula(doc, "mIoU = (1/C) · Σ IoU(c)",
                "stredná hodnota cez všetky C triedy (macro priemer)")
    doc.add_paragraph()

    add_heading(doc, "2.2  Dice koeficient (F1 skóre)", 2)
    add_paragraph(doc,
        "Dice koeficient je ekvivalentný F1 skóre pre binárnu klasifikáciu pixelov "
        "každej triedy. Je citlivejší na malé objekty ako IoU."
    )
    add_formula(doc, "Dice(c) = 2·TP / (2·TP + FP + FN)",
                "vzťah k IoU: Dice = 2·IoU / (1 + IoU)")
    doc.add_paragraph()

    add_heading(doc, "2.3  Presnosť (Precision)", 2)
    add_paragraph(doc,
        "Presnosť vyjadruje, aká časť pixelov predikovaných ako trieda c skutočne "
        "patrí do triedy c. Nízka presnosť znamená veľa falošne pozitívnych predikcií."
    )
    add_formula(doc, "Presnosť(c) = TP / (TP + FP)")
    doc.add_paragraph()

    add_heading(doc, "2.4  Úplnosť (Recall / Sensitivity)", 2)
    add_paragraph(doc,
        "Úplnosť vyjadruje, aká časť pixelov skutočne patriacich do triedy c "
        "bola modelom správne detekovaná. Pre vzácne triedy (Nezdrava_hrca, Prasklina) "
        "je recall kľúčová metrika — nízky recall znamená, že model tieto triedy "
        "prehliadal a nedetekoval ich. V kontexte detekcie poškodení dreva je "
        "recall dôležitejší ako precision (neoznačené poškodenie je závažnejšia chyba "
        "než falošný poplach)."
    )
    add_formula(doc, "Úplnosť(c) = TP / (TP + FN)",
                "alias: Sensitivity, True Positive Rate (TPR)")
    doc.add_paragraph()

    add_heading(doc, "2.5  Pixelová presnosť (Pixel Accuracy)", 2)
    add_paragraph(doc,
        "Celkový podiel správne klasifikovaných pixelov. Táto metrika je ovplyvnená "
        "dominantnými triedami (Okolie tvorí ~80 % pixelov), preto sa používa "
        "predovšetkým ako doplnková metrika k mIoU."
    )
    add_formula(doc, "PA = Σ TP(c) / celkový počet pixelov")
    doc.add_paragraph()

    # ── 3. Hlavný model ──
    add_heading(doc, "3  Výsledky hlavného modelu", 1)
    add_paragraph(doc,
        f"Tabuľka obsahuje metriky vypočítané z {main['n_images']} testovacích snímok. "
        f"Hodnoty sú zaokrúhlené na 2 desatinné miesta."
    )
    doc.add_paragraph()
    add_results_table(doc, main)
    doc.add_paragraph()

    # ── Per-class komentár ──
    add_heading(doc, "3.1  Komentár k jednotlivým triedam", 2)

    miou = main["mean_iou"] * 100
    mrec = main["mean_recall"] * 100
    pa   = main["pixel_accuracy"] * 100
    nezdrava  = main["per_class"]["Nezdrava_hrca"]
    prasklina = main["per_class"]["Prasklina"]

    add_paragraph(doc,
        f"Hlavný model dosiahol mIoU {miou:.2f} %, priemernú úplnosť {mrec:.2f} % "
        f"a pixelovú presnosť {pa:.2f} %."
    )
    doc.add_paragraph()

    add_paragraph(doc, "Dominantné triedy (Drevo, Kora, Okolie):", bold=True)
    add_paragraph(doc,
        "Triedy s vysokou frekvenciou pixelov dosahujú vysoké hodnoty všetkých metrík. "
        "Trieda Okolie (pozadie) typicky dosiahne IoU > 99 %, čo výrazne ovplyvňuje "
        "celkovú pixelovú presnosť. Drevo a Kora majú konzistentne vysoký recall, "
        "ich detekcia je prakticky bezproblémová."
    )
    doc.add_paragraph()

    add_paragraph(doc, "Vzácne triedy (Nezdrava_hrca, Prasklina):", bold=True)
    add_paragraph(doc,
        f"Trieda Nezdrava_hrca dosiahla IoU {nezdrava['iou']*100:.1f} %, "
        f"recall {nezdrava['recall']*100:.1f} % a precision {nezdrava['precision']*100:.1f} %. "
        f"Trieda Prasklina dosiahla IoU {prasklina['iou']*100:.1f} %, "
        f"recall {prasklina['recall']*100:.1f} % a precision {prasklina['precision']*100:.1f} %. "
        "Vzácne triedy tvoria menej ako 0,1 % všetkých pixelov v trénovacom datasete, "
        "čo predstavuje extrémnu triednu nerovnováhu. Napriek tomu model dosahuje "
        "klinicky relevantný recall — model deteguje väčšinu pixelov týchto poškodení."
    )
    doc.add_paragraph()

    # ── 4. Ablation štúdia ──
    if has_ablation:
        add_heading(doc, "4  Ablation štúdia: prínos pridaných dát", 1)
        add_paragraph(doc,
            "Pre kvantifikáciu prínosu cielene anotovaných dát "
            "(Dub_praskliny_a a hrce_mixed) bol natrénovaný druhý model na "
            "identických hyperparametroch, ale s redukovanou trénovacou množinou "
            "(iba pôvodné kmen1–kmen10 a Dub_1–Dub_10). Testovacia množina ostáva "
            "identická s hlavným modelom, takže porovnanie je metodologicky čisté. "
            "Tabuľka zobrazuje rozdiel medzi modelmi v triech kľúčových metrikách "
            "(IoU, Recall, Precision)."
        )
        doc.add_paragraph()
        add_comparison_table(doc, main, ablation)
        doc.add_paragraph()

        # Komentar k ablation
        add_heading(doc, "4.1  Diskusia ablation výsledkov", 2)

        prask_main = main["per_class"]["Prasklina"]
        prask_abl  = ablation["per_class"]["Prasklina"]
        d_iou_p    = (prask_main["iou"]    - prask_abl["iou"])    * 100
        d_rec_p    = (prask_main["recall"] - prask_abl["recall"]) * 100
        d_prec_p   = (prask_main["precision"] - prask_abl["precision"]) * 100

        nez_main = main["per_class"]["Nezdrava_hrca"]
        nez_abl  = ablation["per_class"]["Nezdrava_hrca"]
        d_iou_n  = (nez_main["iou"]    - nez_abl["iou"])    * 100
        d_rec_n  = (nez_main["recall"] - nez_abl["recall"]) * 100

        d_miou = (main["mean_iou"] - ablation["mean_iou"]) * 100

        add_paragraph(doc,
            f"Rozdiel celkového mIoU je relatívne malý ({d_miou:+.2f} percentuálneho bodu), "
            "avšak detailný pohľad na jednotlivé metriky vzácnych tried odhaľuje "
            "podstatnejší rozdiel. Pridanie cielene anotovaných dát zvýšilo predovšetkým "
            "úplnosť (recall) detekcie pukliín a hrčí — kľúčových metrík pre "
            "praktické nasadenie modelu v lesníckom priemysle."
        )
        doc.add_paragraph()

        add_paragraph(doc, "Trieda Prasklina:", bold=True)
        add_paragraph(doc,
            f"Recall sa zvýšil o {d_rec_p:+.1f} percentuálneho bodu "
            f"(z {prask_abl['recall']*100:.1f} % na {prask_main['recall']*100:.1f} %), "
            f"pričom precision klesla o {d_prec_p:+.1f} percentuálneho bodu "
            f"(z {prask_abl['precision']*100:.1f} % na {prask_main['precision']*100:.1f} %). "
            "Tento posun znamená, že hlavný model je v detekcii pukliní citlivejší — "
            "deteguje viac skutočných pukliní (vyšší recall) za cenu o niečo nižšej "
            "presnosti predikcií. Z pohľadu praktického využitia je to žiaduci kompromis: "
            "nezistená puklina znamená v dreve nepostrehnutú vadu, kým falošná pozitívna "
            "predikcia môže byť ľahko vylúčená pri následnej kontrole."
        )
        doc.add_paragraph()

        add_paragraph(doc, "Trieda Nezdrava_hrca:", bold=True)
        add_paragraph(doc,
            f"Recall sa zvýšil o {d_rec_n:+.1f} percentuálneho bodu "
            f"(z {nez_abl['recall']*100:.1f} % na {nez_main['recall']*100:.1f} %). "
            f"IoU narástol o {d_iou_n:+.1f} percentuálneho bodu, čo potvrdzuje, "
            "že prínos hrce_mixed datasetu sa prejavil predovšetkým v lepšej detekcii "
            "hrčí, nielen v upravenej presnosti predikcií."
        )
        doc.add_paragraph()

        add_paragraph(doc, "Záver ablation štúdie:", bold=True)
        add_paragraph(doc,
            "Ablation štúdia kvantifikuje prínos cielene anotovaných dát na úrovni "
            f"+{d_rec_p:.1f} percentuálneho bodu recall pre triedu Prasklina a "
            f"+{d_rec_n:.1f} percentuálneho bodu recall pre triedu Nezdrava_hrca. "
            "Hoci celkový mIoU zostáva podobný, pridaním 50 cielene anotovaných snímok "
            "pukliní (Dub_praskliny_a) a podmnožiny hrce_mixed datasetu sa významne "
            "zlepšila schopnosť modelu detegovať vady, čo je hlavný cieľ aplikácie "
            "v lesníckom priemysle. Z metodologického hľadiska tento výsledok demonštruje, "
            "že priemerné agregované metriky (mIoU) môžu zakrývať podstatu zmeny — "
            "pre vzácne triedy je dôležitejšie sledovať per-class recall, nie iba IoU."
        )

    # ── 5. Experiment 2.5D ──
    has_25d = comp_3trunks is not None
    if has_25d:
        m_v3_3t  = comp_3trunks["v3"]
        m_abl_3t = comp_3trunks["v3_ablation"]
        m_25d_3t = comp_3trunks["v3_ablation_25d"]

        add_heading(doc, "5  Experiment 2.5D — vplyv 3D priestorového kontextu", 1)
        add_paragraph(doc,
            "Pre overenie, či priestorová kontinuita medzi konsekutívnymi CT rezmi "
            "môže pomôcť pri detekcii pukliín (ktoré majú geometrický charakter v 3D), "
            "bol implementovaný 2.5D variant modelu. Vstupom modelu nie je jeden "
            "CT rez (3 kanály opakovaného grayscale), ale 5 konsekutívnych rezov "
            "z toho istého kmena (5 kanálov). Centrálny rez je predikovaný, "
            "okolité 4 rezy slúžia iba ako vstupný kontext (žiadna supervízia)."
        )
        doc.add_paragraph()
        add_paragraph(doc, "Postup experimentu:", bold=True)
        for line in [
            "Architektúra a hyperparametre identické s ablation modelom (CE 30 % + Dice 70 %, "
              "ReduceLROnPlateau, oversample, alpha clamp 20).",
            "Trénovacia množina rovnaká ako pre 2D ablation (kmen1–kmen10 + Dub_1–Dub_10 "
              "minus testovacie trunky).",
            "in_channels=5 — pretrained ImageNet váhy prvej konvolučnej vrstvy "
              "boli adaptované z 3 na 5 kanálov priemerovaním RGB.",
            "Augmentácia (flip, rotate) sa aplikuje rovnako na všetkých 5 kanálov.",
        ]:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(line)
        doc.add_paragraph()

        add_heading(doc, "5.1  Špeciálna testovacia množina (3 trunky)", 2)
        add_paragraph(doc,
            f"Pre férové porovnanie 2D vs 2.5D bol vytvorený špeciálny test set "
            f"obsahujúci LEN 3 celé hold-out trunky: kmen4 + kmen9 + Dub_3b "
            f"({m_v3_3t['n_images']} snímok). Tento test sa nepoužíva v ablation "
            f"štúdii (kapitola 4) — bol vytvorený špecificky pre 2.5D experiment, "
            f"pretože iba na týchto kompletných CT sekvenciách má 2.5D priestorový "
            f"kontext zmysel. Snímky z Dub_praskliny_a (ručne vybrané, nie spatial "
            f"neighbors) a hrce_mixed[:15] (out-of-distribution pre ablation tréning) "
            f"boli vylúčené, aby výsledok porovnával skutočnú schopnosť modelov "
            f"využiť priestorový kontext."
        )
        doc.add_paragraph()

        add_heading(doc, "5.2  Výsledky 3-trunks testu", 2)
        add_paragraph(doc,
            "Tabuľka porovnáva tri modely na rovnakej špeciálnej testovacej množine. "
            "Stĺpec „v3 (2D, full)\" je referenčný hlavný model trénovaný na celom "
            "datasete vrátane Dub_praskliny_a a hrce_mixed; ostatné dva modely "
            "boli trénované iba na pôvodných 20 trunkoch. Stĺpec Δ porovnáva 2.5D "
            "voči 2D pri rovnakej tréningovej množine."
        )
        doc.add_paragraph()
        add_3way_comparison_table(doc, m_v3_3t, m_abl_3t, m_25d_3t)
        doc.add_paragraph()

        # Diskusia
        add_heading(doc, "5.3  Diskusia výsledkov 2.5D experimentu", 2)

        d_miou_25d = (m_25d_3t["mean_iou"] - m_abl_3t["mean_iou"]) * 100
        prask_2d  = m_abl_3t["per_class"]["Prasklina"]
        prask_25d = m_25d_3t["per_class"]["Prasklina"]
        nez_2d    = m_abl_3t["per_class"]["Nezdrava_hrca"]
        nez_25d   = m_25d_3t["per_class"]["Nezdrava_hrca"]
        d_iou_p   = (prask_25d["iou"] - prask_2d["iou"]) * 100
        d_iou_n   = (nez_25d["iou"]   - nez_2d["iou"])   * 100

        add_paragraph(doc,
            f"Pôvodná hypotéza, že priestorová kontinuita medzi rezmi pomôže pri "
            f"detekcii geometricky 3D útvarov (najmä pukliín), sa nepotvrdila. "
            f"2.5D model dosiahol o {abs(d_miou_25d):.2f} percentuálneho bodu nižšie "
            f"mIoU než ekvivalentný 2D model ({m_25d_3t['mean_iou']*100:.2f} % vs "
            f"{m_abl_3t['mean_iou']*100:.2f} %). Najväčší pokles bol pozorovaný "
            f"u triedy Nezdrava_hrca (IoU {d_iou_n:+.1f} pp) a Prasklina ({d_iou_p:+.1f} pp)."
        )
        doc.add_paragraph()

        add_paragraph(doc, "Pravdepodobné príčiny zlyhania:", bold=True)
        for cause in [
            "Adaptácia ImageNet pretrained váh prvej konvolučnej vrstvy z 3 na 5 "
              "vstupných kanálov priemerovaním RGB znižuje kvalitu inicializácie. "
              "Model prakticky stráca výhodu pretrained encoderu v prvej vrstve.",
            "Konsekutívne CT rezy sú vizuálne veľmi podobné — neprinášajú "
              "dostatok nového informačného obsahu, aby kompenzovali stratu "
              "z bodu vyššie. Centrálny rez má v 5-kanálovom vstupe iba 1/5 váhu.",
            "Veľkosť trénovacej množiny (17 trunkov ≈ 924 trénovacích snímok) "
              "nie je dostatočná na to, aby sa model naučil zmysluplné 3D vzory "
              "z hrubých 5-rezových stackov.",
        ]:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(cause)
        doc.add_paragraph()

        add_paragraph(doc, "Predbežný záver experimentu 2.5D (5-kanál):", bold=True)
        add_paragraph(doc,
            "Naivné rozšírenie 2D na 2.5D pomocou 5-kanálového stackingu konsekutívnych "
            "CT rezov nepriniesol očakávaný prínos. Hlavnou príčinou je strata "
            "ImageNet pretrained signálu pri adaptácii prvej konvolučnej vrstvy. "
            "Otázkou ostáva: zlyháva 2.5D priestup principiálne, alebo len kvôli "
            "tomuto špecifickému technickému problému? V kapitole 5.4 testujeme "
            "alternatívny RGB encoding ktorý túto otázku zodpovedá."
        )
        doc.add_paragraph()

    # ── 5.4 RGB adjacent-slice encoding ──
    has_rgb = comp_3trunks_rgb is not None
    if has_rgb and has_25d:
        m_2d_rgb  = comp_3trunks_rgb["v3_ablation"]
        m_25d_rgb = comp_3trunks_rgb["v3_ablation_25d"]
        m_rgb     = comp_3trunks_rgb["v3_ablation_rgb"]

        add_heading(doc, "5.4  RGB adjacent-slice encoding (alternatívny 2.5D prístup)", 2)
        add_paragraph(doc,
            "Aby sme oddelili dva potenciálne dôvody zlyhania 5-kanálového 2.5D "
            "(strata ImageNet pretrain vs. nedostatočný informačný obsah susedov), "
            "implementovali sme alternatívny 2.5D prístup známy ako "
            "adjacent-slice RGB encoding (Roth et al., 2014; Christ et al., 2016): "
            "tri konsekutívne CT rezy sú zakódované ako jednotlivé kanály "
            "RGB obrázka — R = (n−1), G = (n), B = (n+1). Anotácia ostáva "
            "pre centrálny rez n."
        )
        doc.add_paragraph()
        add_paragraph(doc, "Kľúčová výhoda RGB encodingu:", bold=True)
        for line in [
            "Vstup ostáva 3-kanálový — ImageNet pretrained encoder funguje natívne, "
              "žiadna adaptácia prvej konvolučnej vrstvy.",
            "Centrálny rez je v G kanáli, ktorý má v ImageNet luminosity formulácii "
              "(0.299·R + 0.587·G + 0.114·B) najvyššiu váhu — model „vidí\" centrálny "
              "rez najsilnejšie.",
            "Žiadny memory overhead oproti 2D — rovnaké výpočtové nároky, rovnaké "
              "tréningové časy.",
        ]:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(line)
        doc.add_paragraph()
        add_paragraph(doc,
            "Z každého kmena bol vyhodený prvý a posledný rez (nemajú obojstranných "
            f"susedov pre RGB encoding), test set sa redukoval z {m_25d_3t['n_images']} "
            f"na {m_rgb['n_images']} snímok. Všetky tri modely (2D, 2.5D 5-kanál, "
            "RGB encoding) boli vyhodnotené na tomto identickom 186-snímkovom teste, "
            "aby porovnanie bolo metodologicky čisté."
        )
        doc.add_paragraph()

        add_heading(doc, "5.5  Výsledky porovnania troch prístupov (186 snímok)", 2)
        add_paragraph(doc,
            "Tabuľka zobrazuje per-class metriky pre tri modely vyhodnotené na "
            "rovnakej redukovanej 3-trunks testovacej množine. Stĺpec Δ porovnáva "
            "RGB encoding voči 2D baseline-u."
        )
        doc.add_paragraph()
        add_2d_vs_25d_vs_rgb_table(doc, m_2d_rgb, m_25d_rgb, m_rgb)
        doc.add_paragraph()

        # Diskusia RGB
        add_heading(doc, "5.6  Diskusia výsledkov RGB encodingu", 2)

        d_miou_rgb_2d  = (m_rgb["mean_iou"] - m_2d_rgb["mean_iou"]) * 100
        d_miou_rgb_25d = (m_rgb["mean_iou"] - m_25d_rgb["mean_iou"]) * 100

        add_paragraph(doc,
            f"RGB encoding dosiahol mIoU {m_rgb['mean_iou']*100:.2f} %, čo je "
            f"o {d_miou_rgb_25d:+.2f} percentuálneho bodu lepší výsledok než "
            f"5-kanálové 2.5D ({m_25d_rgb['mean_iou']*100:.2f} %), avšak stále "
            f"o {d_miou_rgb_2d:+.2f} percentuálneho bodu nižší než 2D baseline "
            f"({m_2d_rgb['mean_iou']*100:.2f} %)."
        )
        doc.add_paragraph()

        add_paragraph(doc, "Interpretácia:", bold=True)
        for line in [
            f"Pozitívny zistok: RGB encoding prekonal 5-kanálovú variantu o "
              f"{d_miou_rgb_25d:+.2f} pp mIoU. Hypotéza, že zachovanie 3-kanálovej "
              f"štruktúry je dôležité pre využitie ImageNet pretrained váh, sa potvrdila.",
            f"Negatívny zistok: ani RGB encoding nedosiahol výkon 2D baseline-u "
              f"({d_miou_rgb_2d:+.2f} pp). Strata pretrained signálu teda nebola "
              f"jediným dôvodom zlyhania 2.5D — aj plne kompatibilný 2.5D prístup "
              f"nepriniesol skutočný zisk z 3D priestorového kontextu.",
            "Konzistencia: dva nezávislé 2.5D prístupy zlyhali prekonať plain 2D — "
              "ide o robustný negatívny výsledok, nie o náhodu.",
        ]:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(line)
        doc.add_paragraph()

        add_paragraph(doc, "Pravdepodobné príčiny zlyhania 2.5D ako konceptu:", bold=True)
        for cause in [
            "Susedné CT rezy v dataste prierezov stromov sú anatomicky veľmi "
              "podobné — neprinášajú dostatok nového informačného obsahu, "
              "ktorý by model dokázal využiť na zlepšenie segmentácie centrálneho rezu.",
            "Anotácie sú výlučne 2D — model sa supervizuje len pre stredný rez, "
              "takže nemá ako naučiť 3D štruktúru pukliny alebo hrčí.",
            "Pre G kanál vidí RGB model presne ten istý vstup ako 2D model. "
              "Dodatočná informácia v R/B kanáloch je z pohľadu modelu skôr "
              "drobný šum než užitočný signál.",
        ]:
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(cause)
        doc.add_paragraph()

        add_paragraph(doc, "Záver experimentov 2.5D:", bold=True)
        add_paragraph(doc,
            "Dva nezávislé pokusy o rozšírenie 2D na 2.5D segmentation "
            "(5-kanálový stack a RGB adjacent-slice encoding) konzistentne nedosiahli "
            "zlepšenie nad 2D baseline na tomto datasete. Ide o robustný "
            "negatívny výsledok, ktorý naznačuje, že priestorová informácia zo "
            "susedných CT rezov nie je v dataste prierezov stromov modelom "
            "dostatočne využiteľná. Ak by malo zmysel pokračovať s 3D prístupom, "
            "vyžadovalo by si to úplne odlišnú architektúru (3D U-Net s patch "
            "trénovaním, V-Net, Swin UNETR) v kombinácii s 3D anotáciami "
            "kontinuálnymi naprieč rezmi. Pre účely tejto diplomovej práce je "
            "2D variant ponechaný ako finálny model."
        )
        doc.add_paragraph()

    # ── Záver ──
    section_num = 6 if has_25d else (5 if has_ablation else 4)
    add_heading(doc, f"{section_num}  Záver", 1)
    add_paragraph(doc,
        f"Hlavný model SegFormer-B2 (váhy {main['model']}) dosiahol na testovacej "
        f"množine výsledok mIoU {miou:.2f} %, priemerný recall {mrec:.2f} % "
        f"a pixelovú presnosť {pa:.2f} %. Pre kľúčové triedy z hľadiska detekcie "
        f"poškodení dosahuje recall hodnoty {prasklina['recall']*100:.1f} % "
        f"(Prasklina) a {nezdrava['recall']*100:.1f} % (Nezdrava_hrca), "
        "čo predstavuje praktický signál pre využiteľnosť modelu v lesníckom priemysle."
    )


# ── Main ──────────────────────────────────────────────────────────────────
def main(main_path: Path, ablation_path: Path | None,
         comp_3trunks_paths: dict | None,
         comp_3trunks_rgb_paths: dict | None,
         out_path: Path):
    if not main_path.exists():
        print(f"ERROR: {main_path} not found. Run evaluate_test.py first.")
        return

    with open(main_path) as f:
        main_metrics = json.load(f)
    print(f"Main      : {main_path.name}     -> mIoU {main_metrics['mean_iou']*100:.2f}%")

    ablation_metrics = None
    if ablation_path is not None and ablation_path.exists():
        with open(ablation_path) as f:
            ablation_metrics = json.load(f)
        print(f"Ablation  : {ablation_path.name} -> mIoU {ablation_metrics['mean_iou']*100:.2f}%")
    else:
        print(f"Ablation  : not provided -- section 4 will be skipped")

    comp_3trunks = None
    if comp_3trunks_paths is not None:
        loaded = {}
        all_ok = True
        for key, path in comp_3trunks_paths.items():
            if path is not None and path.exists():
                with open(path) as f:
                    loaded[key] = json.load(f)
                print(f"  3trunks {key:<18}: {path.name} -> mIoU {loaded[key]['mean_iou']*100:.2f}%")
            else:
                print(f"  3trunks {key:<18}: {path} CHYBA -> sekcia 5 sa preskoci")
                all_ok = False
        if all_ok:
            comp_3trunks = loaded

    comp_3trunks_rgb = None
    if comp_3trunks_rgb_paths is not None:
        loaded_rgb = {}
        all_ok_rgb = True
        for key, path in comp_3trunks_rgb_paths.items():
            if path is not None and path.exists():
                with open(path) as f:
                    loaded_rgb[key] = json.load(f)
                print(f"  3t-rgb  {key:<18}: {path.name} -> mIoU {loaded_rgb[key]['mean_iou']*100:.2f}%")
            else:
                print(f"  3t-rgb  {key:<18}: {path} CHYBA -> sekcia 5.4 sa preskoci")
                all_ok_rgb = False
        if all_ok_rgb:
            comp_3trunks_rgb = loaded_rgb

    doc = Document()
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.0)
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(12)

    build_report(doc, main_metrics, ablation_metrics, comp_3trunks, comp_3trunks_rgb)

    doc.save(out_path)
    print(f"\nReport saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main",     type=Path, default=DATA_DIR / "metrics_v3.json",
                        help="Hlavny model metrics JSON")
    parser.add_argument("--ablation", type=Path, default=DATA_DIR / "metrics_v3_ablation.json",
                        help="Ablation model metrics JSON (optional)")
    parser.add_argument("--v3-3trunks",          type=Path,
                        default=DATA_DIR / "metrics_v3_3trunks.json")
    parser.add_argument("--ablation-3trunks",    type=Path,
                        default=DATA_DIR / "metrics_v3_ablation_3trunks.json")
    parser.add_argument("--ablation-25d-3trunks", type=Path,
                        default=DATA_DIR / "metrics_v3_ablation_25d_3trunks.json")
    # 3-trunks RGB (186 imgs) -- pre sekciu 5.4
    parser.add_argument("--ablation-3trunks-rgb", type=Path,
                        default=DATA_DIR / "metrics_v3_ablation_3trunks_rgb.json")
    parser.add_argument("--ablation-25d-3trunks-rgb", type=Path,
                        default=DATA_DIR / "metrics_v3_ablation_25d_3trunks_rgb.json")
    parser.add_argument("--ablation-rgb-3trunks", type=Path,
                        default=DATA_DIR / "metrics_v3_ablation_rgb_3trunks.json")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "report.docx")
    args = parser.parse_args()

    abl = args.ablation if args.ablation.exists() else None

    # 3-trunks sekcia sa zaradi iba ak su dostupne VSETKY 3 metric subory
    comp = {
        "v3":              args.v3_3trunks,
        "v3_ablation":     args.ablation_3trunks,
        "v3_ablation_25d": args.ablation_25d_3trunks,
    }
    comp_rgb = {
        "v3_ablation":     args.ablation_3trunks_rgb,
        "v3_ablation_25d": args.ablation_25d_3trunks_rgb,
        "v3_ablation_rgb": args.ablation_rgb_3trunks,
    }

    main(args.main, abl, comp, comp_rgb, args.out)
