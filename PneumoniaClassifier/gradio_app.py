from app import predict_image
import gradio as gr

in_components = gr.Image(type="pil")
out_components = [gr.Textbox(label="Class"), gr.Number(label="Confidence")]
demo = gr.Interface(predict_image, in_components, out_components)
demo.launch()


