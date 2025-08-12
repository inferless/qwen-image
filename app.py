import torch
from diffusers import DiffusionPipeline
from typing import Optional
import base64, io
from pydantic import BaseModel, Field
import inferless
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="A coffee shop entrance features a chalkboard sign reading 'Qwen Coffee 😊 $2 per cup,' with a neon light beside it displaying 'HELLO'. Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written 'π≈3.1415926-53589793-23846264-33832795-02384197'. Ultra HD, 4K, cinematic composition")
    negative_prompt: Optional[str] = " "
    true_cfg_scale: Optional[float] = 4.0
    num_inference_steps: Optional[int] = 50
    height: Optional[int] = 1328
    width: Optional[int] = 1328
    seed: Optional[int] = 42

@inferless.response
class ResponseObjects(BaseModel):
    image_base64: str = Field(default="Test Output")

class InferlessPythonModel:
    def initialize(self):
        model_name = "Qwen/Qwen-Image"
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
      
    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        image = self.pipe(
                          prompt=inputs.prompt,
                          negative_prompt=inputs.negative_prompt,
                          width=inputs.width,
                          height=inputs.height,
                          num_inference_steps=inputs.num_inference_steps,
                          true_cfg_scale=inputs.true_cfg_scale,
                          generator=torch.Generator(device="cuda").manual_seed(inputs.seed)
        ).images[0]
      
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()
        return ResponseObjects(image_base64=encoded)

    def finalize(self):
        self.pipe = None
