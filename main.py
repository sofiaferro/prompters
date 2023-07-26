# flask
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

# model
import torch
from diffusers import StableDiffusionPipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# utilities
import base64
from io import BytesIO

# load model for colab gpu
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

# start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)


@app.route('/')
def initial():
    # renders a template
    return render_template('index.html')


@app.route("/generate_image", methods=['POST'])
def generate_image():
    # takes the prompt from the input form
    prompt = request.form['prompt-input']
    print(f"Generating an image of {prompt}")

    # generates the image
    image = pipe(prompt).images[0]
    print("Image generated! Converting image ...")

    # creates an in-memory buffer using to hold the binary data
    buffered = BytesIO()

    # saves the generated image into the buffered object as a PNG format image
    image.save(buffered, format="PNG")

    # encodes the image data from the buffer into a base64-encoded string
    img_str = base64.b64encode(buffered.getvalue())

    # prepares the image data as a data uri to embed the image directly into the HTML page
    img_str = "data:image/png;base64," + str(img_str)[2:-1]

    print("Sending image ...")
    # the data uri of the image is passed to the html template
    return render_template('index.html', generated_image=img_str)


@app.route("/random_prompt", methods=['POST'])
def generate_random_prompt():
    # Load the pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"  # You can also use "EleutherAI/gpt-neo-1.3" for a more powerful model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the model to generate text
    model.eval()

    # Generate a random verse
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(
            " ", add_special_tokens=False)).unsqueeze(0)  # No initial input
        output = model.generate(
            input_ids=input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text.strip()


random_prompt = generate_random_prompt()


def generate_random_image():
    # generates the image
    image = pipe(random_prompt).images[0]
    print("Image generated! Converting image ...")

    # creates an in-memory buffer using to hold the binary data
    buffered = BytesIO()

    # saves the generated image into the buffered object as a PNG format image
    image.save(buffered, format="PNG")

    # encodes the image data from the buffer into a base64-encoded string
    img_str = base64.b64encode(buffered.getvalue())

    # prepares the image data as a data uri to embed the image directly into the HTML page
    img_str = "data:image/png;base64," + str(img_str)[2:-1]

    print("Sending image ...")
    # the data uri of the image is passed to the html template
    return render_template('index.html', generated_random_image=img_str)


if __name__ == '__main__':
    prompt = generate_random_prompt()
    print("Random prompt:")
    print(prompt)
    app.run()
