from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import os
from collections import Counter

def generate_attack_report(results, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("XIDS - Intrusion Detection Summary Report", styles["Title"]))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Paragraph(f"Total Records Analysed: {len(results)}", styles["Normal"]))
    story.append(Spacer(1, 12))

    if not results:
        story.append(Paragraph("No prediction data available.", styles["Normal"]))
        doc = SimpleDocTemplate(
            save_path,
            pagesize=A4,
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30
        )
        doc.build(story)
        return save_path

    labels = [r.get("prediction_label", "UNKNOWN") for r in results]
    total = len(labels)
    counter = Counter(labels)

    attack_data = [["Attack Type", "Count", "Percentage"]]

    for label, count in counter.items():
        percent = round((count / total) * 100, 2)
        attack_data.append([label, count, f"{percent}%"])

    story.append(Paragraph("Attack Distribution", styles["Heading2"]))

    attack_table = Table(attack_data, repeatRows=1)

    attack_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#111827")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#1f2937")),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))

    story.append(attack_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Key Observed Features", styles["Heading2"]))

    features = []
    for item in results:
        if item.get("explanation"):
            for f in item["explanation"]:
                features.append(f.get("feature"))

    if features:
        feature_counter = Counter(features).most_common(5)

        feature_data = [["Feature", "Occurrences"]]
        for feat, cnt in feature_counter:
            feature_data.append([feat, cnt])

        feature_table = Table(feature_data)

        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#111827")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#1f2937")),
            ('TEXTCOLOR', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))

        story.append(feature_table)
    else:
        story.append(Paragraph("No feature explanation data available.", styles["Normal"]))

    story.append(Spacer(1, 20))

    benign_count = counter.get("BENIGN", 0)
    attack_count = total - benign_count

    if total == 0:
        threat_level = "LOW"
    else:
        threat_ratio = (attack_count / total) * 100

        if threat_ratio > 70:
            threat_level = "CRITICAL"
        elif threat_ratio > 40:
            threat_level = "HIGH"
        elif threat_ratio > 20:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"

    story.append(Paragraph("Overall Threat Level", styles["Heading2"]))
    story.append(Paragraph(f"{threat_level}", styles["Title"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Security Recommendation", styles["Heading2"]))

    if threat_level == "CRITICAL":
        rec = "Immediate mitigation required. Block malicious IPs and enforce strict monitoring."
    elif threat_level == "HIGH":
        rec = "High level of malicious activity detected. Apply rate-limiting and firewall rules."
    elif threat_level == "MEDIUM":
        rec = "Potential risk detected. Monitor traffic and investigate abnormal patterns."
    else:
        rec = "System is operating under safe limits. Continue regular monitoring."

    story.append(Paragraph(rec, styles["Normal"]))

    doc = SimpleDocTemplate(
        save_path,
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    doc.build(story)

    return save_path
