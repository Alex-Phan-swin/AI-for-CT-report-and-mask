import json
import os
import re
from pathlib import Path

PROHIBITED_CLAIMS = [
    "tumour type",
    "tumour grade",
    "prognosis",
    "treatment",
    "malignant",
    "benign",
    "diagnosis",
]

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def yes_no(value):
    return "Yes" if value else "No"


def build_supported_claims(evidence):
    findings = evidence["mask_derived_findings"]
    visual = evidence["visual_evidence"]
    regions = evidence.get("evidence_regions", [])
    primary_region = regions[0] if regions else None
    supported_claims = []

    supported_claims.append(
        {
            "claim": (
                "This report is generated from structured segmentation evidence, not "
                "directly from the image alone."
            ),
            "evidence_key": "visual_evidence.source",
            "evidence_value": visual["source"],
        }
    )
    supported_claims.append(
        {
            "claim": f"Finding present: {yes_no(findings['finding_present'])}.",
            "evidence_key": "mask_derived_findings.finding_present",
            "evidence_value": findings["finding_present"],
        }
    )
    supported_claims.append(
        {
            "claim": f"The predicted mask covers {findings['mask_area_percent']:.2f}% of the image.",
            "evidence_key": "mask_derived_findings.mask_area_percent",
            "evidence_value": findings["mask_area_percent"],
        }
    )
    supported_claims.append(
        {
            "claim": f"Primary region: {findings['primary_region']}.",
            "evidence_key": "mask_derived_findings.primary_region",
            "evidence_value": findings["primary_region"],
        }
    )
    supported_claims.append(
        {
            "claim": (
                f"The mean probability inside the mask is "
                f"{findings['mean_mask_probability']:.2f}."
            ),
            "evidence_key": "mask_derived_findings.mean_mask_probability",
            "evidence_value": findings["mean_mask_probability"],
        }
    )

    if findings["finding_present"]:
        supported_claims.append(
            {
                "claim": (
                    "A suspicious region was identified in evidence region "
                    f"{primary_region['id'] if primary_region else 'MASK'}."
                ),
                "evidence_key": "evidence_regions[0]",
                "evidence_value": primary_region or visual["overlay"],
            }
        )
        if primary_region:
            supported_claims.append(
                {
                    "claim": (
                        f"{primary_region['id']} is shown as the "
                        f"{primary_region['color']['name']} highlighted region."
                    ),
                    "evidence_key": f"visual_evidence.colour_key.{primary_region['id']}",
                    "evidence_value": primary_region,
                }
            )
    else:
        supported_claims.append(
            {
                "claim": (
                    "No suspicious region exceeded the configured visual-evidence "
                    "thresholds."
                ),
                "evidence_key": "mask_derived_findings.area_threshold_percent",
                "evidence_value": findings["area_threshold_percent"],
            }
        )

    return supported_claims


def build_sentence_evidence_map(evidence):
    findings = evidence["mask_derived_findings"]
    regions = evidence.get("evidence_regions", [])
    primary = regions[0] if regions else None
    primary_colour = primary["color"]["name"] if primary else "mask"
    rows = [
        {
            "sentence": "The report is generated from structured segmentation evidence.",
            "evidence_id": "PIPELINE",
            "support": "visual_evidence.source",
            "status": "supported",
        },
        {
            "sentence": f"Finding present: {yes_no(findings['finding_present'])}.",
            "evidence_id": "MASK",
            "support": "mask_derived_findings.finding_present",
            "status": "supported",
        },
        {
            "sentence": f"The predicted mask covers {findings['mask_area_percent']:.2f}% of the image.",
            "evidence_id": "MASK",
            "support": "mask_derived_findings.mask_area_percent",
            "status": "supported",
        },
        {
            "sentence": (
                f"The finding is localised primarily in the {findings['primary_region']} "
                f"and corresponds to the {primary_colour} visual evidence region."
            ),
            "evidence_id": primary["id"] if primary else "MASK",
            "support": "mask_derived_findings.primary_region + visual_evidence.colour_key",
            "status": "supported",
        },
        {
            "sentence": "Unsupported clinical attributes are blocked rather than inferred.",
            "evidence_id": "GUARDRAIL",
            "support": "report_constraints",
            "status": "blocked",
        },
    ]
    return rows


def generate_report(evidence):
    findings = evidence["mask_derived_findings"]
    supported_claims = build_supported_claims(evidence)
    sentence_evidence_map = build_sentence_evidence_map(evidence)
    regions = evidence.get("evidence_regions", [])
    primary = regions[0] if regions else None
    colour_lines = (
        "\n".join(
            f"- {region['id']} is shown as the {region['color']['name']} highlighted region."
            for region in regions
        )
        if regions
        else "- No colour-coded evidence region exceeded the evidence thresholds."
    )

    if findings["finding_present"]:
        evidence_id = primary["id"] if primary else "MASK"
        colour_name = primary["color"]["name"] if primary else "red"
        findings_text = (
            f"A suspicious region was identified in evidence region {evidence_id}, "
            f"shown as the {colour_name} highlighted region on the overlay. "
            f"The predicted mask covers {findings['mask_area_percent']:.2f}% of the image "
            f"and is localised primarily in the {findings['primary_region']}. "
            f"The mean probability inside the mask is {findings['mean_mask_probability']:.2f}."
        )
        impression_text = (
            f"The {colour_name} highlighted region is consistent with a possible tumour-like abnormality. "
            "This statement is grounded only in the predicted mask, overlay, and structured evidence."
        )
    else:
        findings_text = (
            "No suspicious region exceeded the configured visual-evidence thresholds. "
            f"The candidate abnormal mask area was {findings['mask_area_percent']:.2f}% "
            f"with mean mask probability {findings['mean_mask_probability']:.2f}."
        )
        impression_text = (
            "No clear tumour-like abnormality was localised by the current model under "
            "the current thresholds. This does not rule out disease and requires clinical review."
        )

    report = (
        "Evidence-Constrained Segmentation Report\n"
        "========================================\n\n"
        "Grounding method:\n"
        "This report is generated from structured segmentation evidence, not directly from the image alone.\n\n"
        "Visual evidence:\n"
        f"- Predicted mask: {evidence['visual_evidence']['predicted_mask']}\n"
        f"- Overlay: {evidence['visual_evidence']['overlay']}\n\n"
        "Colour key:\n"
        f"{colour_lines}\n\n"
        "Structured evidence:\n"
        f"- Finding present: {yes_no(findings['finding_present'])}\n"
        f"- Mask area: {findings['mask_area_percent']:.2f}%\n"
        f"- Primary region: {findings['primary_region']}\n"
        f"- Mean mask probability: {findings['mean_mask_probability']:.2f}\n\n"
        "Evidence regions:\n"
        + (
            "\n".join(
                f"- {region['id']} ({region['color']['name']}): {region['area_percent']:.2f}% area, "
                f"{region['mean_probability']:.2f} mean probability, "
                f"{region['region_label']}, bbox {region['bounding_box_pixels']}"
                for region in regions
            )
            if regions
            else "- No region exceeded the evidence thresholds."
        )
        + "\n\n"
        f"Findings:\n{findings_text}\n\n"
        f"Impression:\n{impression_text}\n\n"
        "Claim grounding map:\n"
        + "\n".join(
            f"- [{row['status'].upper()}] {row['sentence']} "
            f"(evidence: {row['evidence_id']}, source: {row['support']})"
            for row in sentence_evidence_map
        )
        + "\n\n"
        "Safety note:\n"
        "Prototype only. Not a diagnostic tool. Requires clinical review.\n"
    )

    return report, supported_claims, sentence_evidence_map


def qwen_enabled():
    return os.environ.get("MEDISCAN_USE_QWEN", "").strip().lower() in {"1", "true", "yes"}


def build_qwen_prompt(evidence, draft_report):
    findings = evidence["mask_derived_findings"]
    regions = evidence.get("evidence_regions", [])
    region_summary = (
        "\n".join(
            f"- {region['id']} is {region['color']['name']}, area {region['area_percent']:.2f}%, "
            f"mean probability {region['mean_probability']:.2f}, location {region['region_label']}, "
            f"bbox {region['bounding_box_pixels']}"
            for region in regions
        )
        if regions
        else "- No colour-coded evidence region exceeded the evidence thresholds."
    )

    return f"""You are writing a concise medical imaging prototype report grounded in segmentation evidence.

                Use ONLY the structured segmentation evidence below. Do not infer tumour type, tumour grade, prognosis, malignancy, treatment, or diagnosis. Mention that this is a prototype and requires clinical review.

                Report structure to follow:

                1. Opening summary: Write 3 lines in detail describing what the AI model identified based on the {findings}, mentioning key regions affected and their locations.

                2. Detailed Regional Impact section: For each colour-coded evidence region, provide a numbered point (1, 2, 3, etc.) that includes:
                    Write 3 lines describing
                - The region ID and colour
                - The percentage of that region affected
                - The functional/anatomical significance of that region
                - Potential impact on brain function or patient presentation

                3. Clinical Implications section: Provide numbered points describing:
                    Write 3 lines describing
                - How the lesion locations may affect cognitive and motor functions
                - Specific functional impacts based on the regions identified
                - The need for clinical correlation

                4. Include {findings}, {region_summary}, and any other relevant structured evidence in the report.
                """


def clean_report_formatting(report):
    cleaned_lines = []
    for line in report.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = line.replace("**", "")
        line = line.replace("__", "")
        line = re.sub(r"^\s*[-*]\s+", "- ", line)
        cleaned_lines.append(line.rstrip())

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned + "\n"


def generate_qwen_report(evidence, draft_report):
    model_name = os.environ.get("MEDISCAN_QWEN_MODEL", DEFAULT_QWEN_MODEL)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Qwen report generation needs transformers and torch installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    messages = [
        {
            "role": "system",
            "content": (
                "You generate grounded prototype medical imaging reports from structured "
                "segmentation evidence. You must not add unsupported clinical claims."
            ),
        },
        {"role": "user", "content": build_qwen_prompt(evidence, draft_report)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=650,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    generated_ids = outputs[:, inputs.input_ids.shape[1] :]
    return clean_report_formatting(
        tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    ).strip()


def validate_report(report, supported_claims, sentence_evidence_map):
    lower_report = report.lower()
    unsupported_terms = [
        term for term in PROHIBITED_CLAIMS if term in lower_report and term != "diagnosis"
    ]
    contains_diagnosis = "diagnostic tool" in lower_report or "diagnosis" in lower_report
    if contains_diagnosis:
        unsupported_terms = [term for term in unsupported_terms if term != "diagnosis"]

    claim_checks = []
    for claim in supported_claims:
        claim_checks.append(
            {
                "claim": claim["claim"],
                "supported": claim["claim"].lower().rstrip(".") in lower_report,
                "evidence_key": claim["evidence_key"],
                "evidence_value": claim["evidence_value"],
            }
        )

    validation = {
        "is_valid": not unsupported_terms,
        "unsupported_terms": unsupported_terms,
        "claim_checks": claim_checks,
        "sentence_evidence_map": sentence_evidence_map,
        "blocked_claims": [
            "tumour type",
            "tumour grade",
            "malignancy status",
            "prognosis",
            "treatment recommendation",
        ],
        "guardrails": [
            "Report must be generated from evidence.json.",
            "Report must not infer tumour type, grade, prognosis, or treatment.",
            "Report must state prototype limitation and clinical review requirement.",
        ],
    }
    return validation


def write_report_outputs(evidence, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report, supported_claims, sentence_evidence_map = generate_report(evidence)
    qwen_error = None
    if qwen_enabled():
    #if True:
        try:
            qwen_report = generate_qwen_report(evidence, report)
            qwen_validation = validate_report(qwen_report, supported_claims, sentence_evidence_map)
            if qwen_validation["is_valid"]:
                report = clean_report_formatting(qwen_report)
            else:
                qwen_error = (
                    "Qwen output was rejected by validation; deterministic report was used."
                )
        except Exception as exc:
            print('qwen fail')
            qwen_error = f"Qwen unavailable; deterministic report was used. Reason: {exc}"

    validation = validate_report(report, supported_claims, sentence_evidence_map)
    validation["qwen"] = {
        "enabled": qwen_enabled(),
        "model": os.environ.get("MEDISCAN_QWEN_MODEL", DEFAULT_QWEN_MODEL),
        "error": qwen_error,
    }

    (output_path / "report.txt").write_text(report, encoding="utf-8")
    (output_path / "report_validation.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    validation_lines = [
        "Report Validation",
        "=================",
        "",
        f"Valid: {yes_no(validation['is_valid'])}",
        f"Qwen enabled: {yes_no(validation['qwen']['enabled'])}",
        f"Qwen model: {validation['qwen']['model']}",
        f"Qwen note: {validation['qwen']['error'] or 'None'}",
        f"Unsupported terms: {validation['unsupported_terms'] or 'None'}",
        "",
        "Supported claims:",
    ]
    for check in validation["claim_checks"]:
        validation_lines.append(
            f"- {yes_no(check['supported'])}: {check['claim']} "
            f"(source: {check['evidence_key']})"
        )
    validation_lines.extend(["", "Sentence-to-evidence map:"])
    for row in validation["sentence_evidence_map"]:
        validation_lines.append(
            f"- {row['status'].upper()}: {row['sentence']} "
            f"(evidence: {row['evidence_id']})"
        )

    (output_path / "report_validation.txt").write_text(
        "\n".join(validation_lines) + "\n",
        encoding="utf-8",
    )

    return report, validation
