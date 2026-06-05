import torch
from config import load_data, device
from models import CLIPVLM, text_encoder, text_tokenizer, report_generator
import os

# =========================
# DATA
# =========================
vision_features, labels = load_data()

vision_dim = vision_features.shape[1]
text_dim = text_encoder.config.hidden_size

# Define detailed, structured prompts for report generation
class_prompts = [
    "ABDOMINAL SCAN DIAGNOSIS: Normal Healthy Liver\n\nFindings: The liver demonstrates normal echogenicity and size. No fatty infiltration detected. Liver parenchyma appears homogeneous. Hepatic vasculature is normal. No focal lesions identified.\n\nConclusion: Normal abdominal scan with no pathological findings.",
    "ABDOMINAL SCAN DIAGNOSIS: Hepatic Steatosis\n\nFindings: The liver demonstrates increased echogenicity consistent with hepatic steatosis. There is diffuse fatty infiltration of the liver parenchyma. No focal lesions or cirrhotic features identified.\n\nConclusion: Imaging findings consistent with hepatic steatosis."
]

# =========================
# MODEL PATH
# =========================
model_path = os.path.join(
    os.path.dirname(__file__),
    "projection",
    "clip_vlm.pt"
)

# =========================
# MODEL LOAD (FIXED)
# =========================
model = CLIPVLM(vision_dim, text_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =========================
# REPORT GENERATION FUNCTION
# =========================
def generate_medical_report(diagnosis, confidence):
    """Generate a structured medical imaging report based on diagnosis."""
    
    if diagnosis == "Normal Healthy Liver":
        findings = [
            "Normal liver size and contour with homogeneous parenchymal echogenicity",
            "No evidence of fatty infiltration or steatosis",
            "Normal hepatic vasculature with preserved portal venous flow",
            "No focal hepatic lesions, masses, or abnormal collections",
            "No signs of cirrhosis, portal hypertension, or cholestasis"
        ]
        # Use LLM to generate conclusion
        prompt = "Generate a brief clinical conclusion for normal liver imaging findings in an abdominal scan:"
        try:
            generated = report_generator(
                prompt,
                max_length=60,
                num_return_sequences=1,
                temperature=0.6,
                top_p=0.8,
                do_sample=True
            )
            conclusion = generated[0]['generated_text'].replace(prompt, '').strip()
            # Clean up and limit length
            conclusion = conclusion.split('.')[0] + '.' if '.' in conclusion else conclusion
            if len(conclusion) < 10 or not conclusion[0].isupper():
                conclusion = "The liver demonstrates normal imaging characteristics with no evidence of pathology. No follow-up imaging is required."
        except:
            conclusion = "The liver demonstrates normal imaging characteristics with no evidence of pathology. No follow-up imaging is required."
            
    else:  # Hepatic Steatosis
        findings = [
            "Diffuse increased hepatic echogenicity consistent with steatosis (fatty infiltration)",
            "Hepatic parenchyma demonstrates heterogeneous echotexture",
            "Hepatic borders are preserved, no evidence of cirrhosis",
            "Portal venous radicles are partially obscured by increased echogenicity",
            "No focal lesions or signs of acute liver disease detected"
        ]
        # Use LLM to generate conclusion
        prompt = "Generate a brief clinical conclusion for hepatic steatosis imaging findings in an abdominal scan:"
        try:
            generated = report_generator(
                prompt,
                max_length=80,
                num_return_sequences=1,
                temperature=0.6,
                top_p=0.8,
                do_sample=True
            )
            conclusion = generated[0]['generated_text'].replace(prompt, '').strip()
            # Clean up and limit length
            conclusion = '.'.join(conclusion.split('.')[:2]) + '.' if '.' in conclusion else conclusion
            if len(conclusion) < 15 or not conclusion[0].isupper():
                conclusion = "The imaging findings are consistent with hepatic steatosis. Clinical correlation with liver function tests and metabolic markers is recommended. Patient should consider lifestyle modifications."
        except:
            conclusion = "The imaging findings are consistent with hepatic steatosis. Clinical correlation with liver function tests and metabolic markers is recommended. Patient should consider lifestyle modifications."
    
    return findings, conclusion
def retrieve_best_match(image_feat, text_bank):
    """Classify image and return predicted class and confidence score."""
    image_feat = image_feat.to(device)

    with torch.no_grad():
        # image embedding
        v = model.vision_proj(image_feat)
        v = v / v.norm(dim=-1, keepdim=True)

        best_score = -float("inf")
        best_index = None

        for i, t in enumerate(text_bank):
            # SUPPORT BOTH STRING OR TOKENIZED INPUT
            if isinstance(t, str):
                inputs = text_tokenizer(
                    t,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(device) for k, v in t.items()}

            outputs = text_encoder(**inputs)
            t_emb = outputs.last_hidden_state[:, 0, :]
            t_emb = model.text_proj(t_emb)
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)

            score = torch.matmul(v, t_emb.T).item()

            if score > best_score:
                best_score = score
                best_index = i

    return best_index, best_score

# =========================
# RUN
# =========================
if __name__ == "__main__":

    for i in range(10):
        print(f"\n{'='*75}")
        print(f"ABDOMINAL IMAGING REPORT {i+1}")
        print(f"{'='*75}")
        
        predicted_class, confidence = retrieve_best_match(vision_features[i:i+1], class_prompts)
        diagnosis = "Normal Healthy Liver" if predicted_class == 0 else "Hepatic Steatosis"
        
        # Generate structured report
        findings, conclusion = generate_medical_report(diagnosis, confidence)
        
        # Format the report output
        print(f"\nDIAGNOSIS: {diagnosis}")
        print(f"Confidence Score: {confidence:.4f}")
        print(f"\nKEY IMAGING FINDINGS:")
        for idx, finding in enumerate(findings, 1):
            print(f"  {idx}. {finding}")
        print(f"\nCLINICAL CONCLUSION:")
        print(f"  {conclusion}")
        print(f"{'='*75}")