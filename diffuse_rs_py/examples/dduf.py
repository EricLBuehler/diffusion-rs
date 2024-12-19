from diffuse_rs import DiffusionGenerationParams, ModelSource, Pipeline
from PIL import Image
import io

pipeline = Pipeline(source=ModelSource.DdufFile("FLUX.1-schnell-Q4-bnb.dduf"))

image_bytes = pipeline.forward(
    prompts=["Draw a picture of a sunrise."],
    params=DiffusionGenerationParams(
        height=720, width=1280, num_steps=4, guidance_scale=0.0
    ),
)

image = Image.open(io.BytesIO(image_bytes[0]))
image.show()
