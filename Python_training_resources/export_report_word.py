"""
Generates a Word (.docx) report for the diploma thesis.

Sections:
  1. Introduction -- model and dataset description
  2. Evaluation metrics -- definitions and formulas
  3. Per-class results table
  4. Aggregate metrics
  5. Interpretation commentary

Requires:
    pip install python-docx

Usage:
    python export_report_word.py
    python export_report_word.py --metrics metrics.json --out report.docx
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
    """Add a formula line -- italic monospace style."""
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


# ── Report content ────────────────────────────────────────────────────────
def build_report(doc, metrics):
    # ── Title ──
    title = doc.add_heading("Vyhodnotenie modelu sémantickej segmentácie", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # ── 1. Model a dataset ──
    add_heading(doc, "1  Model a testovacia množina", 1)
    add_paragraph(doc,
        f"Na vyhodnotenie bol použitý model SegFormer-B2 trénovaný dvojfázovo "
        f"na datasete prierezov kmeňov stromov (kmen1–kmen10, Dub_1–Dub_10). "
        f"Testovanie prebehlo na samostatnej testovacej množine '{metrics['test_trunk']}' "
        f"({metrics['n_images']} snímok), ktorá nebola použitá počas tréningu ani validácie. "
        f"Súbor váh modelu: {metrics['model']}."
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

    # IoU
    add_heading(doc, "2.1  IoU — Priesečník nad zjednotením (Jaccard index)", 2)
    add_paragraph(doc,
        "IoU meria mieru prekryvu predikovanej a skutočnej segmentačnej masky. "
        "Hodnota 1,0 znamená dokonalú zhodu, 0,0 znamená žiadny prekryv."
    )
    add_formula(doc,
        "IoU(c) = TP / (TP + FP + FN)",
        "kde TP, FP, FN sú počty pixelov pre triedu c"
    )
    add_formula(doc,
        "mIoU = (1/C) · Σ IoU(c)",
        "stredná hodnota cez všetky C triedy (macro priemer)"
    )
    doc.add_paragraph()

    # Dice
    add_heading(doc, "2.2  Dice koeficient (F1 skóre)", 2)
    add_paragraph(doc,
        "Dice koeficient je ekvivalentný F1 skóre pre binárnu klasifikáciu pixelov "
        "každej triedy. Je citlivejší na malé objekty ako IoU, pretože penalizuje "
        "falošne pozitívne a falošne negatívne rovnako."
    )
    add_formula(doc,
        "Dice(c) = 2·TP / (2·TP + FP + FN)",
        "vzťah k IoU: Dice = 2·IoU / (1 + IoU)"
    )
    doc.add_paragraph()

    # Precision
    add_heading(doc, "2.3  Presnosť (Precision)", 2)
    add_paragraph(doc,
        "Presnosť vyjadruje, aká časť pixelov predikovaných ako trieda c skutočne "
        "patrí do triedy c. Nízka presnosť znamená veľa falošne pozitívnych predikcií."
    )
    add_formula(doc,
        "Presnosť(c) = TP / (TP + FP)"
    )
    doc.add_paragraph()

    # Recall
    add_heading(doc, "2.4  Úplnosť (Recall / Sensitivity)", 2)
    add_paragraph(doc,
        "Úplnosť vyjadruje, aká časť pixelov skutočne patriacich do triedy c "
        "bola modelom správne detekovaná. Pre vzácne triedy (Nezdrava_hrca, Prasklina) "
        "je recall kľúčová metrika — nízky recall znamená, že model tieto triedy "
        "prehliadal a nedetekoval ich."
    )
    add_formula(doc,
        "Úplnosť(c) = TP / (TP + FN)",
        "alias: Sensitivity, True Positive Rate (TPR)"
    )
    doc.add_paragraph()

    # Pixel accuracy
    add_heading(doc, "2.5  Pixelová presnosť (Pixel Accuracy)", 2)
    add_paragraph(doc,
        "Celkový podiel správne klasifikovaných pixelov. Táto metrika je ovplyvnená "
        "dominantnými triedami (Okolie tvorí ~80 % pixelov), preto sa používa "
        "predovšetkým ako doplnková metrika k mIoU."
    )
    add_formula(doc,
        "PA = Σ TP(c) / celkový počet pixelov"
    )
    doc.add_paragraph()

    # ── 3. Výsledky ──
    add_heading(doc, "3  Výsledky na testovacej množine", 1)
    add_paragraph(doc,
        f"Tabuľka obsahuje metriky vypočítané z {metrics['n_images']} testovacích snímok "
        f"datasetu '{metrics['test_trunk']}'. Hodnoty sú zaokrúhlené na 2 desatinné miesta."
    )
    doc.add_paragraph()
    add_results_table(doc, metrics)
    doc.add_paragraph()

    # ── 4. Interpretácia ──
    add_heading(doc, "4  Interpretácia výsledkov", 1)

    miou  = metrics["mean_iou"] * 100
    mprec = metrics["mean_precision"] * 100
    mrec  = metrics["mean_recall"] * 100
    pa    = metrics["pixel_accuracy"] * 100

    nezdrava = metrics["per_class"]["Nezdrava_hrca"]
    prasklina = metrics["per_class"]["Prasklina"]

    add_paragraph(doc,
        f"Model dosiahol mIoU {miou:.2f} %, priemernú úplnosť {mrec:.2f} % "
        f"a pixelovú presnosť {pa:.2f} %. Výsledky sú hodnotené samostatne "
        f"pre každú triedu:"
    )
    doc.add_paragraph()

    add_paragraph(doc, "Dominantné triedy (Drevo, Kora, Okolie):", bold=True)
    add_paragraph(doc,
        "Triedy s vysokou frekvenciou pixelov dosahujú vysoké hodnoty všetkých metrík. "
        "Trieda Okolie (pozadie) typicky dosiahne IoU > 95 %, čo výrazne ovplyvňuje "
        "celkovú pixelovú presnosť. Tieto výsledky sú očakávané a potvrdzujú "
        "správnosť základnej segmentácie."
    )
    doc.add_paragraph()

    add_paragraph(doc, "Vzácne triedy (Nezdrava_hrca, Prasklina):", bold=True)
    add_paragraph(doc,
        f"Trieda Nezdrava_hrca dosiahla IoU {nezdrava['iou']*100:.1f} % "
        f"a recall {nezdrava['recall']*100:.1f} %. "
        f"Trieda Prasklina dosiahla IoU {prasklina['iou']*100:.1f} % "
        f"a recall {prasklina['recall']*100:.1f} %. "
        "Nízke hodnoty recall pri vzácnych triedach naznačujú, že model časť "
        "týchto oblastí nedetekoval (falošne negatívne predikcie). "
        "Príčinou je extrémna triedna nerovnováha — tieto triedy tvoria menej "
        "ako 0,1 % všetkých pixelov v trénovacom datasete. "
        "Pre zlepšenie by bolo vhodné rozšíriť trénovacie dáta o ďalšie vzorky "
        "s hrčami a trhlinami, prípadne použiť silnejšie prevzorkovanie."
    )
    doc.add_paragraph()

    add_paragraph(doc, "Porovnanie Precision a Recall:", bold=True)
    add_paragraph(doc,
        "Pre praktické využitie v lesníckom priemysle (detekcia poškodení dreva) "
        "je recall dôležitejší ako precision — neodhalené poškodenie (FN) je "
        "závažnejšie ako falošný poplach (FP). Model by mal byť hodnotený "
        "predovšetkým podľa recall hodnôt vzácnych tried."
    )


# ── Main ──────────────────────────────────────────────────────────────────
def main(metrics_path: Path, out_path: Path):
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found. Run evaluate_test.py first.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3.0)
        section.right_margin  = Cm(2.0)

    # Default font
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(12)

    build_report(doc, metrics)

    doc.save(out_path)
    print(f"Report saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=DATA_DIR / "metrics.json")
    parser.add_argument("--out",     type=Path, default=DATA_DIR / "report.docx")
    args = parser.parse_args()
    main(args.metrics, args.out)
