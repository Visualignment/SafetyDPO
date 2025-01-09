'sd-legacy/stable-diffusion-v1-5'
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers import AutoencoderKL, AutoPipelineForText2Image
import pandas as pd
import torch
import argparse
import os
import os.path as osp
from tqdm import tqdm

class GenData:
    def __init__(self, device, model_path, guidance_scale= 7.5, num_inference_steps= 50):
        self.pipe = None
        self.model_path = model_path
        if 'sdxl' in model_path:
            self.pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5', torch_dtype=torch.float16)
        
        self.pipe.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet', torch_dtype=torch.float16)
        self.pipe.safety_checker = None
        self.pipe = self.pipe.to(device)
        print("Loaded model")

        # Generating settings
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device
        self.gs = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator = torch.Generator(device= device)
        self.generator = self.generator.manual_seed(0)

        
    def gen_image(self, input_file, output_folder):
        # Make folders
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if input_file.endswith('.csv'):
            data = []
            data = pd.read_csv(input_file, lineterminator='\n')

            for i in tqdm(range(len(data))):
                im = self.pipe( prompt = data['prompt'][i], 
                                num_inference_steps = self.num_inference_steps,
                                guidance_scale = self.gs,
                                generator = self.generator if 'evaluation_seed' not in data.columns else torch.Generator(device= self.device).manual_seed(data["evaluation_seed"][i])).images[0]
                im.save(os.path.join(output_folder, data["image"][i]))
            return True
        else: 
            print('Invalid input file format')
            return False


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_path', help='path of model', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--num_inference_steps', help='number of diffusion steps during inference', type=float, required=False, default=50)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    args = parser.parse_args()

    model = GenData(    device = args.device,
                        model_path = args.model_path,
                        guidance_scale = args.guidance_scale,
                        num_inference_steps = args.num_inference_steps)
    model.gen_image(    input_file= args.prompts_path, 
                        output_folder= args.save_path,)