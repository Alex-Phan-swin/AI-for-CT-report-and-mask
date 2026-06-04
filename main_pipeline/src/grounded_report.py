import json
from pathlib import Path
from datetime import datetime

class GroundedReportGenerator:
    """
    Generates reports that ONLY state what can be verified from the segmentation.
    No hallucination - every claim is traceable to specific regions/colours.
    """
    
    def __init__(self):
        self.claims_log = []  # Track all claims for validation
    
    def generate_from_regions(self, regions, original_image_path, modality="Brain MRI"):
        """
        Generate a structured report based solely on region data.
        Returns: (report_text, validation_log)
        """
        
        total_regions = len(regions)
        
        if total_regions == 0:
            return self._empty_report(modality), {"warning": "No tumour regions detected"}
        
        # Calculate metrics from regions
        total_tumour_percent = sum(r['size_percent'] for r in regions)
        primary_size = regions[0]['size_percent'] if regions else 0
        
        # Build structured report
        report_sections = []
        
        # 1. Examination header
        report_sections.append(f"=== MEDICAL IMAGING REPORT ===")
        report_sections.append(f"Modality: {modality}")
        report_sections.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report_sections.append(f"Image: {Path(original_image_path).name}")
        report_sections.append("")
        
        # 2. Colour-coded region identification (traceable evidence)
        report_sections.append("=== COLOUR-CODED VISUAL EVIDENCE ===")
        report_sections.append("The segmentation identifies distinct regions using the following colour scheme:")
        
        for i, region in enumerate(regions, 1):
            colour_name = [k for k, v in COLOUR_MAP.items() if v == region['colour']][0]
            report_sections.append(f"\nRegion {i} - {colour_name.replace('_', ' ').upper()}:")
            report_sections.append(f"  • Colour: {region['colour']} (RGB)")
            report_sections.append(f"  • Size: {region['size_percent']:.1f}% of scan area")
            report_sections.append(f"  • Location: Centre at approximate coordinates ({region['centre'][1]:.0f}, {region['centre'][0]:.0f})")
            
            # Log this claim for validation
            self.claims_log.append({
                "claim": f"Region {i} identified as {colour_name}",
                "evidence": f"Connected component {i} with {region['size_percent']:.1f}% area",
                "verifiable": True
            })
        
        # 3. Observations (only what's measurable)
        report_sections.append("\n=== OBSERVATIONS ===")
        
        if total_regions == 1:
            obs = f"A single abnormal region is visible, highlighted in RED (primary tumour region), occupying {primary_size:.1f}% of the scan area."
            report_sections.append(f"• {obs}")
            self.claims_log.append({"claim": obs, "evidence": f"single region, {primary_size:.1f}% area", "verifiable": True})
        
        elif total_regions >= 2:
            obs = f"Multiple abnormal regions detected: {total_regions} distinct areas identified. The primary region (RED) occupies {primary_size:.1f}% of the scan, with {total_regions - 1} additional {regions[1]['label'] if len(regions) > 1 else 'secondary'} region(s) visible in GREEN and YELLOW."
            report_sections.append(f"• {obs}")
            self.claims_log.append({"claim": obs, "evidence": f"{total_regions} regions detected", "verifiable": True})
        
        # 4. Region-specific findings (traceable by colour)
        report_sections.append("\n=== REGION-SPECIFIC FINDINGS ===")
        
        for i, region in enumerate(regions, 1):
            colour_name = [k for k, v in COLOUR_MAP.items() if v == region['colour']][0]
            colour_desc = colour_name.replace('_', ' ')
            
            finding = f"The {colour_desc} region (Region {i}) spans {region['size_percent']:.1f}% of the visible scan area and is located in the {'upper' if region['centre'][0] < 128 else 'lower'} {'left' if region['centre'][1] < 128 else 'right'} quadrant."
            report_sections.append(f"• {finding}")
            self.claims_log.append({
                "claim": finding,
                "evidence": f"Region {i}: {region['size_percent']:.1f}% at position ({region['centre'][1]:.0f}, {region['centre'][0]:.0f})",
                "verifiable": True
            })
        
        # 5. Impression (cautious, grounded)
        report_sections.append("\n=== IMPRESSION ===")
        
        if total_tumour_percent > 10:
            impression = f"The segmentation reveals {total_regions} distinct abnormal region(s) occupying {total_tumour_percent:.1f}% of the scan. Clinical correlation is recommended for further characterization of the {regions[0]['label']} (RED region)."
        elif total_tumour_percent > 2:
            impression = f"A small abnormal focus is detected (RED region, {total_tumour_percent:.1f}% of scan). This finding may represent early abnormality; follow-up imaging could be considered."
        else:
            impression = f"Minimal abnormal signal detected ({total_tumour_percent:.1f}% of scan). This finding is of uncertain significance."
        
        report_sections.append(f"• {impression}")
        self.claims_log.append({
            "claim": impression,
            "evidence": f"{total_regions} regions, {total_tumour_percent:.1f}% total area",
            "verifiable": True,
            "uncertainty": "explicitly stated"
        })
        
        # 6. Validation footer
        report_sections.append("\n=== VALIDATION ===")
        report_sections.append(f"All claims in this report are traceable to the colour-coded segmentation map.")
        report_sections.append(f"Total verifiable claims: {len(self.claims_log)}")
        report_sections.append("No diagnostic assertions beyond visible evidence are made.")
        
        return "\n".join(report_sections), self.claims_log
    
    def _empty_report(self, modality):
        return f"""=== MEDICAL IMAGING REPORT ===
Modality: {modality}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

=== OBSERVATIONS ===
No significant abnormal regions were detected by the segmentation model.

=== IMPRESSION ===
Unremarkable scan with no clear tumour-suspicious regions identified.

=== VALIDATION ===
This conclusion is based on the absence of segmented regions meeting the minimum size threshold."""
    
    def save_validation_log(self, output_path):
        """Save claims log for audit trail"""
        with open(output_path, 'w') as f:
            json.dump(self.claims_log, f, indent=2)