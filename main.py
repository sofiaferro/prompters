from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

import torch
from diffusers import StableDiffusionPipeline

import base64
from io import BytesIO

# load model for colab gpu
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16
)

pipe.to("cuda")

# start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)


@app.toute("/")
def initial():
    return render_template("index.html")


@app.route("/generate_image", methods=['POST'])
def generate_image():
    prompt = request.form["prompt-input"]
    image = pipe(prompt).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = "data:image/png;base64," + str(img_str)[2:-1]

    print("Sending image ...")
    return render_template('index.html', generated_image=img_str)


if __name__ == '__main__':
    app.run()
