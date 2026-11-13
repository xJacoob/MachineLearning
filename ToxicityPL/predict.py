from pathlib import Path
import torch
from transformers import pipeline
import gradio as gr

ROOT = Path(__file__).parent
MODEL_DIR = str(ROOT / "checkpoints" / "herbert" / "checkpoint-2400")
MAX_LENGTH = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = pipeline(
    task="text-classification",
    model=MODEL_DIR,
    device=device,
)

def predict(text: str) -> tuple[str, float]:
    result = classifier(text, truncation=True, max_length=MAX_LENGTH)
    label = result[0]["label"]
    score = result[0]["score"]
    return label, round(score, 3)

if __name__ == "__main__":
    EXAMPLES = [
        # wyraźnie harmful: przemoc
        "przestępca powinien być kopany w ryj tak długo aż straci przytomność i będzie grzecznie leżał do przyjazdu policji.",
        # wyraźnie harmful: obelga wobec rozmówcy
        "{USERNAME} ty debilu, nikt cię nie pytał o zdanie, wracaj pod swój kamień",
        # ostra krytyka, ale nie hejt
        "Ten rząd to banda nieudaczników, kolejna ustawa napisana na kolanie i znowu zapłacimy za to my.",
        # neutralny
        "Dzięki za polecenie tej książki, przeczytałem w weekend i naprawdę warto. Ktoś zna coś podobnego?",
        # sarkazm (trudny przypadek)
        "No brawo, genialny pomysł, na pewno tym razem się uda",
        # para do testu shortcutu: ten sam tekst bez i z {USERNAME}
        "nie zgadzam się z tobą, ale rozumiem twój punkt widzenia.",
        "{USERNAME} nie zgadzam się z tobą, ale rozumiem twój punkt widzenia.",
    ]

    in_components = gr.Textbox(lines=4, label="Enter harmful or non-harmful text: ")
    out_components = [gr.Textbox(label="Class"), gr.Number(label="Confidence")]
    demo = gr.Interface(predict, in_components, out_components, flagging_mode="never", examples=EXAMPLES)
    demo.launch()