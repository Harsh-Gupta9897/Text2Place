import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import torch
import os
from PIL import Image
import math

#### ---- textual Inversion ----- ####
from diffusers import DiffusionPipeline,DDIMScheduler,DDPMScheduler

import torch
from PIL import Image,ImageEnhance
import torchvision.transforms as T
from tqdm import auto
import random
import numpy as np

def save_XLembedding(emb,embedding_file="myToken.pt",path="./Embeddings/"):
    torch.save(emb,path+embedding_file)

def set_XLembedding(base,emb,token="my"):
    with torch.no_grad():            
        # Embeddings[tokenNo] to learn
        tokens=base.components["tokenizer"].encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer'"
        tokenNo=tokens[1]
        tokens=base.components["tokenizer_2"].encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer_2'"
        tokenNo2=tokens[1]
        embs=base.components["text_encoder"].text_model.embeddings.token_embedding.weight
        embs2=base.components["text_encoder_2"].text_model.embeddings.token_embedding.weight
        assert embs[tokenNo].shape==emb["emb"].shape, "different 'text_encoder'"
        assert embs2[tokenNo2].shape==emb["emb2"].shape, "different 'text_encoder_2'"
        embs[tokenNo]=emb["emb"].to(embs.dtype).to(embs.device)
        embs2[tokenNo2]=emb["emb2"].to(embs2.dtype).to(embs2.device)

def load_XLembedding(base,token="my",embedding_file="myToken.pt",path="./Embeddings/"):
    emb=torch.load(path+embedding_file)
    set_XLembedding(base,emb,token)

#### ---- textual Inversion ----- ####




def dilate_mask(mask, radii):
    '''
        Desc: given a mask, dilate it with different radii
    '''
    dilated_masks = []
    for radius in radii:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius + 1, 2*radius + 1))
        dilated = cv2.dilate(np.array(mask), kernel, iterations=1)
        dilated_masks.append(Image.fromarray(dilated))
    return dilated_masks

# Function to create a two-row grid with the initial image and text in the first cell
def create_row_grid(init_image_with_text, dilated_masks, inpainted_images, output_filename,n_seeds,person_name, output_dir, individual_file_name):
    '''
        Desc: Create a grid image with the initial image and text in the first cell, dilated masks in the first row, and inpainted images in the second row
    '''
    rows = n_seeds+1
    cols = max(len(dilated_masks), len(inpainted_images[0])) + 1  # +1 for the initial image with text
    grid_image = Image.new('RGB', ((cols - 1) * 512 + init_image_with_text.width, rows * 512))

    # Place the initial image with text in the first column
    grid_image.paste(init_image_with_text, (0, 0))

    # Create column-wise folders
    for col in range(cols):
        column_dir = os.path.join(output_dir, f'column_{col}')
        os.makedirs(column_dir, exist_ok=True)


    # Place the dilated masks in the first row and inpainted images in the second row
    for i, (mask, inpainted) in enumerate(zip(dilated_masks, inpainted_images[0])):

        column_dir = os.path.join(output_dir, f'column_{i+1}')

        grid_image.paste(mask, ((i + 1) * 512, 0))  # Offset by one to account for the initial image
        for j in range(n_seeds):
            inpainted = inpainted_images[j][i]
            ###.........###
            filename = f'image_{person_name}_{j}_{individual_file_name}'
            output_path = os.path.join(column_dir, filename)
            inpainted.save(output_path)

            grid_image.paste(inpainted, ((i + 1) * 512, (j+1)*512))

    grid_image.save(output_filename)

def add_text_to_init_image(init_image, text):
    '''
        Desc: Add text to the initial image
    '''
    image_with_text = init_image.copy()  # Create a copy of the original image
    draw = ImageDraw.Draw(image_with_text)
    font = ImageFont.load_default()
    draw.text((20, 20), text, (255, 255, 255), font=font)
    return image_with_text



# Function to process a batch of inpainting tasks
def process_inpainting_batch(pipe, image, dilated_masks, prompt,mask_file,seed):
    '''
        Desc: Process a batch of inpainting images
    '''
    batch_results = []
    try:
        # Prepare the batch for the pipeline
        batch = [{'prompt': prompt, 'init_image': image, 'mask_image': mask} for mask in dilated_masks]
        init_images = [init_image for _ in range(len(batch))]
        prompts = [prompt for _ in range(len(batch))]
        
        # Process the batch through the pipeline
        results = pipe(prompt=prompts, image=init_images, mask_image=dilated_masks).images

        # Resize results to match mask dimensions and append to the batch_results list
        resized_results = []
        for result, mask in zip(results, dilated_masks):
            resized_result = result.resize(mask.size)
            resized_results.append(resized_result)
            batch_results.append(resized_result)

        # Save individual results with masks
        for idx, (resized_result, mask) in enumerate(zip(resized_results, dilated_masks)):
            combined_width = resized_result.width + mask.width
            combined_image = Image.new('RGB', (combined_width, resized_result.height))
            combined_image.paste(resized_result, (0, 0))
            combined_image.paste(mask, (resized_result.width, 0))
            
            individual_image_path = os.path.join(individual_results_directory, f'{mask_file.split(".")[0]}_seed{seed}_mask{idx}.png')
            combined_image.save(individual_image_path)

    except Exception as e:
        print('Error processing batch:', e)
    
    return batch_results


def get_prompt(text,pipe,person="ed_sheeren",embs_path="path/to/embs_directory"):
    '''
        Desc: Get the prompt for the inpainting task from general prompt to prompt with special token for subject
    '''

    learned="sks"
    emb_file=f"{person}.pt"
    text = text.replace("a person","{} person")

    load_XLembedding(pipe,token=learned,embedding_file=emb_file,path=embs_path)
    prompt=text.format(learned)
    return prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--base_dir', type=str, default='output_with_All_parameters_learnable2_0.2')
    parser.add_argument('--image_dir', type=str, default='final_images/')
    parser.add_argument('--mask_type', type=str, default='blob')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument("--person", type=str, default="yann_lecun")
    parser.add_argument('--base_path', type=str, required=True, help='Base path for the project')
    parser.add_argument('--image_directory', type=str, required=True, help='Directory for images')
    parser.add_argument('--mask_directory', type=str, required=True, help='Directory for masks')
    parser.add_argument('--embs_path', type=str, required=True, help='Path for embeddings')

    args = parser.parse_args()
    
    base_path = args.base_path
    image_directory = args.image_directory
    mask_directory = args.mask_directory
    embs_path = args.embs_path
    # Define radii for dilation 
    radii = [1, 5, 15, 30, 50]

    # List all images and masks
    mask_files = [f for f in os.listdir(mask_directory) if f.endswith('.png')]

    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    persons = [ f.split('.') for f in os.listdir(embs_path)]  # should be defined based on embedding_dir_name

    for person in persons:
        try:
            output_directory = f'{base_path}/inpainted_dilated_{person}_{args.mask_type}_results'
            individual_results_directory = f'{base_path}/individual_results_{person}_{args.mask_type}'

            os.makedirs(individual_results_directory, exist_ok=True)
            for mask_file in mask_files:
                try: 
                    mask_image_path = os.path.join(mask_directory, mask_file)
                    base_arr = mask_image_path.split("/")[-1].split("_")
                    image_path = os.path.join(image_directory, base_arr[1] + ".png")
                    prompt = get_prompt(base_arr[0], pipe=pipe, person=person, embs_path=embs_path)
                    init_image = load_image(image_path).resize((512, 512))
                    original_mask = load_image(mask_image_path).resize((512, 512))
                    dilated_masks = dilate_mask(original_mask, radii)
                    init_image_with_text = add_text_to_init_image(init_image, prompt)

                    print(base_arr)
                    os.makedirs(output_directory, exist_ok=True)

                    for i in range(args.n_seeds):
                        if i == 0:
                            inpainted_images = [process_inpainting_batch(pipe, init_image, dilated_masks, prompt, mask_file, seed=i)]
                        else:
                            inpainted_images.append(process_inpainting_batch(pipe, init_image, dilated_masks, prompt, mask_file, seed=i))
                        
                    result_image_path = os.path.join(output_directory, f'{base_arr[0]}_{base_arr[1]}_{args.mask_type}.png')
                    create_row_grid(init_image_with_text, dilated_masks, inpainted_images, result_image_path, args.n_seeds)
                    print(f'Inpainted grid image saved to {result_image_path}')
                except Exception as e:
                    print('Error processing mask:', mask_file, e)
        except Exception as e:
            print('Error processing person:', person, e)

# CUDA_VISIBLE_DEVICES=4 python inpainting_pipeline/inpainting.py  --base_dir g200_campuses_scenes_0.2 --image_dir example_images --mask_type blob --n_seeds 4 --person yann_lecun
