import sys, io, gc, time, os, math, random, requests
from tqdm import tqdm
import numpy as np
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image, ExifTags
from torchvision import transforms
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Output
from IPython.display import display, clear_output
import threestudio
threestudio.utils.base.get_device = lambda: torch.device('cuda')


#### ----------------- Initial Config ----------------- ####
dump_path = '/path/to/dump_dir'
input_path = '/path/to/folder/images_folder_name_present'   ## input_dir
output_path = 'path/to/output_dir'   ## output_dir
output_folder_name = 'output_dir_name'               ## output_dir_name
input_folder_name = 'images_folder_name'                                     ### input_bg_image_folder_path
save_video_frames = False                                                  ### to save video frames in case to visualisize it.
n_iters = 1000                                                             ### # of iterations to generate mask
thresold = 0.2                                                            
guidance_scale = 200.
one_person = True
os.makedirs(dump_path, exist_ok=True)
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def save_frames(img_rgb, fg_img_rgb, mask,frame_path,frame_image_name):
    '''
        Desc: To save frames of foreground,target and mask image'
    '''
    mask_path = f'{frame_path}/masks/'
    fg_target_path = f'{frame_path}/fg_target/'
    target_path = f'{frame_path}/target/'
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(fg_target_path, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)

    cv2.imwrite(f'{target_path}/{frame_image_name}_target.png',img_rgb[:,:,::-1]*255)
    cv2.imwrite(f'{fg_target_path}/{frame_image_name}_fg_target.png',fg_img_rgb[:,:,::-1]*255)
    cv2.imwrite(f'{mask_path}/{frame_image_name}_mask.png',mask[:,:,::-1])
    print(f"\n\n\nFrames saved at {target_path}/{frame_image_name}_target.png\n\n\n")


def get_latent(img_path):
    '''
        To get the latent tensor and image from given image path
    '''

    init_image = Image.open(img_path).convert("RGB") 

    init_image = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize the image to 64x64
        transforms.ToTensor()         # Convert the resized image to tensor
    ])
    image_tensor = transform(init_image)
    # image_tensor = image_tensor.permute(1, 2, 0)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, init_image

##------------ Stable Diffusion Config -----------------##
config = {
    'max_iters': 1000,
    'seed': 12,
    'scheduler': 'cosine',
    'mode': 'latent',
    'prompt_processor_type': 'stable-diffusion-prompt-processor',
    'prompt_processor': {
        'prompt': None,
    },
    'dds_prompt_processor': {
        'prompt': None,
    },
    'text_embedding_perturb_variance' : None,
    'guidance_type': 'stable-diffusion-guidance',
    'guidance': {
        'half_precision_weights': False,
        'guidance_scale': guidance_scale,
        'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
        'grad_clip': None,
        'view_dependent_prompting': False,
    },
    'image': {
        'width': 64,
        'height': 64,
        'image': None
    },
    'title': None,
    'given_steps': 0,
}

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    



seed_everything(config['seed'])


def figure2image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')
    return img

def configure_other_guidance_params_manually(guidance, config):
    # avoid reloading guidance every time change these params
    guidance.cfg.grad_clip = config['guidance']['grad_clip']
    guidance.cfg.guidance_scale = config['guidance']['guidance_scale']


def convert_img_to_video_format(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def find_largest_gradient_box(gradient_image):
    # Convert gradient image to grayscale if it's in color
    if len(gradient_image.shape) == 3 and gradient_image.shape[2] == 3:
        gradient_image = cv2.cvtColor(gradient_image, cv2.COLOR_BGR2GRAY)

    # Step 1: Thresholding - adjust the threshold value as needed
    _, binary_image = cv2.threshold(gradient_image, 50, 255, cv2.THRESH_BINARY)

    # Step 2: Morphological Operations - to disconnect weak components
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    # Step 3: Finding Contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if not contours:
        return None,None,None,None

    # Step 4: Selecting the Largest Contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Step 5: Bounding Box around the Largest Contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h


def save_masks_with_colors(masks, dump_path, guidance, threshold=0.2):
    '''
        Save the masks with colors
    '''
    print("---\n\n\n\n----- running  Blob Function  ----\n\n\n-----")
    colored_masks = []
    fg_images = []
    # Convert each mask to a colored mask
    for i, mask in enumerate(masks):
        mask = mask.to(guidance.device)
        mask = mask.reshape(1, 64, 64, 1)


        
        fg_image = nn.Parameter(torch.randn(1,8,8,3, device=guidance.device)*255)
        fg_images.append(fg_image)

        # Convert the mask to numpy array
        mask_array = mask.permute(1, 2, 0, 3).cpu().detach().numpy()[:, :, :, 0]

        # Apply thresholding
        binary_mask = (mask_array > threshold).astype(np.uint8) * 255

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours and fill with random color
        colored_mask = np.zeros_like(mask_array, dtype=np.uint8)
        for contour in contours:
            color = (np.random.randint(40, 200), np.random.randint(40, 200), np.random.randint(40, 200))
            cv2.drawContours(colored_mask, [contour], -1, color, thickness=cv2.FILLED)

        # Save the colored mask
        cv2.imwrite(f"{dump_path}/mask_{i}.png", colored_mask)

        # Append the colored mask for combining
        colored_masks.append(colored_mask)

    # Combine colored masks
    combined_mask = np.zeros_like(colored_masks[0])
    for colored_mask in colored_masks:
        combined_mask = np.where(colored_mask > 0, colored_mask, combined_mask)

    # Save the combined mask
    cv2.imwrite(f"{dump_path}/combined_mask.png", combined_mask)


def run(config, guidance, prompt_processor, dds_prompt_processor=None,root_seed=1):
    
    # clear gpu memory
    losses=[]
    rgb = None
    grad = None
    vis_grad = None
    vis_grad_norm = None
    loss = None
    optimizer = None
    target = None
    title = config['title']
    given_steps = config['given_steps']
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    title = f'{title}_{root_seed}'
    configure_other_guidance_params_manually(guidance, config)
    mode = config['mode']
    
    w, h = config['image']['width'], config['image']['height']


    if mode == 'rgb':
        target = config['image']['image']
        target = target.to(guidance.device)
        target.requires_grad = True
    else:
        # --------------------------------------------- New logic to validate --------------------------------------------- # 
        # BG target images which should not be optimized 
        bg_target = guidance.encode_images(config['image']['image'].to(guidance.device))
        bg_target = bg_target.to(guidance.device)
        bg_target = bg_target.permute(0,2,3,1)


        # FG target images which should be optimized
        fg_target = guidance.encode_images(config['image']['image'].to(guidance.device))
        fg_target = fg_target.to(guidance.device).permute(0,2,3,1)
        fg_target.requires_grad = True 

        # Mask image which should be optimized along with the foreground image in training 
        mask_shape = [64,64]
        #### ------ for n masks ---------########

        masks = guidance.compute_gaussian_blobs(mask_shape)
        save_masks_with_colors(masks, dump_path, guidance, thresold)
        for i,mask in enumerate(masks):
            mask = mask.to(guidance.device)
            mask = mask.reshape(1, 64, 64, 1)
            cv2.imwrite(f"{dump_path}/mask_{i}.png",mask.permute(1,2,0,3)[:,:,:,0].cpu().detach().numpy()*255)
        
        masks = torch.stack(masks)
        combined_mask = masks.mean(dim=0, keepdim=True)
        combined_mask = combined_mask/combined_mask.max()
        combined_mask = combined_mask.reshape(1, 64, 64, 1)

        combined_mask[combined_mask<thresold] = 0
        combined_mask = combined_mask.to(guidance.device)
    
        cv2.imwrite(f"{dump_path}/combined_mask2.png",combined_mask.permute(1,2,0,3)[:,:,:,0].cpu().detach().numpy()*255)
        print("\n\n\n Masks Saved \n\n\n")

        # Combine the foreground and background target image using the combined mask

        ### ---- normalization ---- #####
        fg_target_normalized = fg_target / 255.0
        bg_target_normalized = bg_target / 255.0        
        # Combine the foreground and background target image using the combined mask
        target = combined_mask * fg_target_normalized + (1 - combined_mask) * bg_target_normalized

        # If you need to convert 'target' back to the standard range [0, 255] for saving or displaying:
        target = (target * 255.0)


    optimizer = torch.optim.AdamW([fg_target],#guidance.scale, guidance.aspect_ratio
                                     lr=2e-1, weight_decay=0)
    center_optimizer = torch.optim.AdamW([guidance.center,guidance.relative_angles],#guidance.scale, guidance.aspect_ratio
                                     lr=1e-1, weight_decay=0)
    param_optimizer = torch.optim.AdamW([guidance.angle],#, 
                                     lr=1e-1, weight_decay=0)
    scale_optimizer = torch.optim.AdamW([guidance.scale,guidance.aspect_ratio],lr=2e-2, weight_decay=0)
    


    num_steps = config['max_iters']
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(num_steps*1.5)) if config['scheduler'] == 'cosine' else None
    center_scheduler = torch.optim.lr_scheduler.StepLR(center_optimizer, step_size=100, gamma=0.2)
    param_scheduler = get_cosine_schedule_with_warmup(param_optimizer, 100, int(num_steps*1.5)) if config['scheduler'] == 'cosine' else None
    rgb = None
    plt.axis('off')

    img_array = []
    # ptp_utils.register_attention_control(guidance.pipe, guidance.controller)
    
    # Initialize accumulated gradients and their norms
    acc_grad = torch.zeros_like(target)
    acc_grad_norm = torch.zeros_like(target[..., :1])  # Assuming the last dimension is for RGB
    
    try:
        for step in tqdm(range(num_steps + 1)):

            ### ------- n blobs ---------- #####
            # masks = guidance.compute_opacity_map(mask_shape)
            masks = guidance.compute_gaussian_blobs(mask_shape)
            masks = torch.stack(masks)
            combined_mask = masks.mean(dim=0, keepdim=True)
            combined_mask = combined_mask/combined_mask.max()
            combined_mask = combined_mask.reshape(1, 64, 64, 1)
            combined_mask[combined_mask<thresold] = 0
            combined_mask = combined_mask.to(guidance.device)


            ### -- normalization -- #####
            fg_target_normalized = fg_target / 255.0
            bg_target_normalized = bg_target / 255.0        
            # Combine the foreground and background target image using the combined mask
            target = combined_mask * fg_target_normalized + (1 - combined_mask) * bg_target_normalized

            # If you need to convert 'target' back to the standard range [0, 255] for saving or displaying:
            target = (target * 255.0)

            optimizer.zero_grad()
            param_optimizer.zero_grad()
            center_optimizer.zero_grad()
            scale_optimizer.zero_grad()
            batch = {
                'elevation': torch.Tensor([1]),
                'azimuth': torch.Tensor([0]),
                'camera_distances': torch.Tensor([5]),
            }
            if dds_prompt_processor:
                loss = guidance(target, 
                                prompt_processor(), 
                                **batch, 
                                rgb_as_latents=(mode != 'rgb'),
                                prompt_utils_paired=dds_prompt_processor(),
                                text_embed_perturb_variance=config['text_embedding_perturb_variance'],
                               )
            else:
                loss = guidance(target, 
                                prompt_processor(), 
                                **batch, 
                                rgb_as_latents=(mode != 'rgb'),
                               )
            losses.append(loss['loss_sds'].item())

            loss['loss_sds'].backward() 

            if dds_prompt_processor:
                grad =  loss['grad_paired']- loss['grad_unpaired']
                grad = grad.permute(0,2,3,1)
            
            else:
                # For visualization changing for now
                grad = fg_target.grad 

            
            optimizer.step() 
            param_optimizer.step() 
            center_optimizer.step()

            scale_optimizer.step()
            grad = torch.abs(grad)
            
            if scheduler is not None:
                scheduler.step()

            if param_scheduler is not None:
                param_scheduler.step()
            
            if center_scheduler is not None:
                center_scheduler.step()

            

            print(f"\n\nGradients iters: {step}\n")
            print("Guidance center Grad: {}".format(guidance.center.grad))
            print("Guidance angle Grad: {}".format(guidance.angle.grad))
            print("Guidance aspect ratio Grad: {}".format(guidance.aspect_ratio.grad))
            print("Guidance scale Grad: {}".format(guidance.scale.grad))
            print("Guidance relative angles Grad: {}\n".format(guidance.relative_angles.grad))
            
            print(f"\nParameters iters: {step}\n")
            print("Guidance center: {}".format(torch.sigmoid(guidance.center)))
            print("Guidance angle: {}".format(torch.tanh(guidance.angle)*torch.pi))
            print("Guidance aspect ratio: {}".format(guidance.aspect_ratio))
            print("Guidance scale: {}".format(guidance.scale))
            print("Guidance relative angles: {}\n".format(torch.tanh(guidance.relative_angles)*torch.pi/4))


            # Accumulate gradients and their norms
            if step>given_steps:
                acc_grad += grad
                acc_grad_norm += grad.norm(dim=-1, keepdim=True)

            if step % 5 == 0:
                if mode == 'rgb':
                    rgb = target
                    vis_grad = grad[..., :3]
                    vis_acc_grad = acc_grad[..., :3]
                else:
                    rgb = guidance.decode_latents(target.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                    fg_rgb = guidance.decode_latents(fg_target.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                    vis_grad = grad
                    vis_acc_grad = acc_grad
                

                vis_grad_norm = grad.norm(dim=-1)
                vis_acc_grad_norm = acc_grad_norm.squeeze(-1)
                segmented_img = process_segmentation(acc_grad_norm)
                # Normalize for visualization
                vis_grad_norm = vis_grad_norm / vis_grad_norm.max()
                vis_acc_grad_norm = vis_acc_grad_norm / vis_acc_grad_norm.max()
                vis_grad = vis_grad / vis_grad.max()
                vis_acc_grad = vis_acc_grad / vis_acc_grad.max()
    
                img_rgb = rgb.clamp(0, 1).detach().squeeze(0).cpu().numpy()
                fg_img_rgb = fg_rgb.clamp(0, 1).detach().squeeze(0).cpu().numpy()
                img_grad = vis_grad.clamp(0, 1).detach().squeeze(0).cpu().numpy()
                img_grad_norm = vis_grad_norm.clamp(0, 1).detach().squeeze(0).cpu().numpy()
                img_acc_grad = vis_acc_grad.clamp(0, 1).detach().squeeze(0).cpu().numpy()
                img_acc_grad_norm = vis_acc_grad_norm.clamp(0, 1).detach().squeeze(0).cpu().numpy()
                original_img = config['image']['image'].clamp(0, 1).detach().squeeze(0).cpu().numpy()
                original_img = np.transpose(original_img, (1, 2, 0)) 
                # segmented_img, rectangular_box_img = process_segmentation(acc_grad_norm, folder_path, config, step)
                fig, ax = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid
                
                ####------ Visualize the images  ----------#######
                frame_path = f'{output_path}/{output_folder_name}_{thresold}/frames/{title}'
                os.makedirs(frame_path, exist_ok=True)
                frame_mask = combined_mask.permute(1,2,0,3)[:,:,:,0].cpu().detach().numpy()*255
                frame_image_path = f'{title}_{step}'
                if save_video_frames and step%5==0:
                    fmasks = guidance.compute_gaussian_blobs([512,512])
                    fmasks = torch.stack(fmasks)
                    fcombined_mask = fmasks.mean(dim=0, keepdim=True)
                    fcombined_mask = fcombined_mask/fcombined_mask.max()
                    fcombined_mask = fcombined_mask.reshape(1, 512, 512, 1)
                    fcombined_mask[fcombined_mask<thresold] = 0
                    fcombined_mask = fcombined_mask.to(guidance.device)
                    frame_mask = fcombined_mask.permute(1,2,0,3)[:,:,:,0].cpu().detach().numpy()*255
                    save_frames(img_rgb, fg_img_rgb,frame_mask,frame_path,frame_image_path)

                ### save the mask and image pair in the folder #####

                # Plot the images in the grid
                ax[0, 0].imshow(original_img)
                ax[0, 0].set_title(f"step={step}  Original Image")
                
                ax[0, 1].imshow(img_rgb)
                ax[0, 1].set_title("Current Image")
                
                ax[0, 2].imshow(fg_img_rgb)
                ax[0, 2].set_title("Foreground Image")

                ax[1, 0].imshow(img_grad_norm)
                ax[1, 0].set_title("Gradient Norm")
                
                ax[1, 1].imshow(segmented_img)
                ax[1, 1].set_title("Segmented Image from Acc Grad Norm")
                # ax[1, 2].imshow(mask[0].permute(1,2,0)[:,:,0].cpu().detach().numpy()*255)
                ax[1, 2].imshow(combined_mask.permute(1,2,0,3)[:,:,:,0].cpu().detach().numpy()*255)
                
                ax[1, 2].set_title("Rectangular Box Image")

                for a in ax:
                    for b in a:
                        b.axis('off')
                clear_output(wait=True)
                plt.tight_layout()  # Ensure proper spacing between subplots

                plt.show()
                img_array.append(figure2image(fig))
    except KeyboardInterrupt:
        pass
    finally:
        #####------ saving the mask,image pair in  inital config dir ----------#######
        
        # Browse the result
        print("Optimizing process:")
        images = img_array
        
        if len(images) > 0:
            slider = IntSlider(min=0, max=len(images)-1, step=1, value=1)
            output = Output()
    
            def display_image(index):
                with output:
                    output.clear_output(wait=True)
                    display(images[index])
    
            interact(display_image, index=slider)


    init_path = f'{output_path}/{output_folder_name}_{thresold}/videos'
    segmented_img_path = f'{output_path}/{output_folder_name}_{thresold}/segmented_mask'
    blobs_independent_path = f'{output_path}/{output_folder_name}_{thresold}/blobs_independent'
    blob_img_path = f'{output_path}/{output_folder_name}_{thresold}/blob_mask'
    folder_path = os.path.join(f'{output_path}/{output_folder_name}_{thresold}', title)
    os.makedirs(init_path, exist_ok=True)
    # os.makedirs(folder_path, exist_ok=True)
    os.makedirs(segmented_img_path, exist_ok=True)
    os.makedirs(blobs_independent_path, exist_ok=True)
    os.makedirs(blob_img_path, exist_ok=True)
    
    ### saving whole mask #####
    combined_mask[combined_mask>=thresold] = 1 
    cv2.imwrite(f'{blob_img_path}/{title}_blob.png',combined_mask.permute(1,2,0,3)[:,:,:,0].cpu().detach().numpy()*255)
        
    # masks = guidance.compute_opacity_map(mask_shape)
    masks = guidance.compute_gaussian_blobs(mask_shape)
    gap = guidance.n//guidance.n_persons
    for i in range(guidance.n_persons):
        if i!=guidance.n_persons-1:
            combined_mask = masks[i*gap:(i+1)*gap]
        else:
            combined_mask = masks[i*gap:]

        combined_mask = torch.stack(combined_mask)
        combined_mask = combined_mask.mean(dim=0, keepdim=True)
        combined_mask = combined_mask/combined_mask.max()
        combined_mask = combined_mask.reshape(1, 64, 64, 1)
        combined_mask[combined_mask<thresold] = 0
        combined_mask = combined_mask.to(guidance.device)
        combined_mask[combined_mask>=thresold] = 1
        print("combined mask shape: {}".format(combined_mask.shape))

        cv2.imwrite(f'{blobs_independent_path}/{title.replace("two","one")}_{guidance.n_persons}_{i}_blob.png',combined_mask.permute(1,2,0,3)[:,:,:,0].cpu().detach().numpy()*255)
        

    if len(img_array) > 0:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = f'{init_path}/{title}_{given_steps}_optimization_video.mp4'
        # print(img_array[0].size)
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (img_array[0].size[0], img_array[0].size[1]))

        for step, img in enumerate(img_array):
            video_img = convert_img_to_video_format(img)
            # Add text to the frame
            step_text = f'Step: {step * 5}'  # Assuming each image corresponds to 5 steps
            cv2.putText(video_img, step_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            video_writer.write(video_img)

        video_writer.release()
        print("Video saved at:", video_path)
     # Don't forget to close the log file when your program is finished

    # Process acc_grad_norm for segmentation
    acc_grad_norm = acc_grad_norm.cpu().detach().numpy()
    acc_grad_norm = acc_grad_norm[0, :, :, :]
    acc_grad_norm = (acc_grad_norm - acc_grad_norm.min()) / (acc_grad_norm.max() - acc_grad_norm.min()) * 255
    acc_grad_norm = acc_grad_norm.astype(np.uint8)
    
    # Threshold to create a binary mask
    threshold = acc_grad_norm.mean() * 2.1
    binary_mask = (acc_grad_norm > threshold).astype(np.uint8) * 255
    
    # Apply the mask to get the segmented image
    segmented_img = acc_grad_norm.copy()
    segmented_img[binary_mask == 0] = 0
    
   # Squeeze the singleton channel dimension
    binary_mask_2d = np.squeeze(binary_mask, axis=-1)  # From (64, 64, 1) to (64, 64)
    segmented_img_2d = np.squeeze(segmented_img, axis=-1)  # From (64, 64, 1) to (64, 64)
    
    # Convert to PIL images
    binary_mask_img = Image.fromarray(binary_mask_2d, mode='L')
    segmented_img_pil = Image.fromarray(segmented_img_2d, mode='L')


    # original_width, original_height = original_img.size
    segmented_img_pil = segmented_img_pil.resize((512, 512), Image.Resampling.LANCZOS)
    binary_mask_img = binary_mask_img.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Save the images
    # binary_mask_path = f'{segmented_img_path}/{title}_blob.png'
    # segmented_img_path = f'{folder_path}/{title}_{given_steps}_segmented_image_acc_grad.png'
    # binary_mask_img.save(binary_mask_path)
    # segmented_img_pil.save(segmented_img_path)
    


def process_segmentation(acc_grad_norm):
    # Process acc_grad_norm for segmentation
    acc_grad_norm = acc_grad_norm.cpu().detach().numpy()
    acc_grad_norm = acc_grad_norm[0, :, :, :]
    acc_grad_norm = (acc_grad_norm - acc_grad_norm.min()) / (acc_grad_norm.max() - acc_grad_norm.min()) * 255
    acc_grad_norm = acc_grad_norm.astype(np.uint8)
    
    # Threshold to create a binary mask
    threshold = acc_grad_norm.mean() * 2.1
    binary_mask = (acc_grad_norm > threshold).astype(np.uint8) * 255
    
    # Apply the mask to get the segmented image
    segmented_img = acc_grad_norm.copy()
    segmented_img[binary_mask == 0] = 0

    return segmented_img
    
def update_config_for_image(image_path, prompt, num_iters, initial_steps, config):
    # Load image tensor
    image_tensor, _ = get_latent(image_path)

    # Update config
    config['image']['image'] = image_tensor
    config['prompt_processor']['prompt'] = prompt
    config['max_iters'] = num_iters
    config['title'] = f"{prompt}_{os.path.basename(image_path).split('.')[0]}_{num_iters}iters_{initial_steps}steps"
    config['given_steps'] = initial_steps


def run_main():
    # List of iterations to run experiments on
    # iterations = [0,50,100,150]
    iterations = [50,]

    # Directory containing images
    src_image_dir = f'{input_path}/{input_folder_name}'

    guidance = None
    prompt_processor = None
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    guidance = threestudio.find(config['guidance_type'])(config['guidance'])


    dds_prompt_processor = None
    os.makedirs(f'{output_path}/{output_folder_name}_{thresold}/blob_mask', exist_ok=True)
    images_already_processed = [ f for f in os.listdir(f'{output_path}/{output_folder_name}_{thresold}/blob_mask') if f.endswith('.png') ]
    print("Images already processed: ",images_already_processed)
    
    # Iterate over each image in the directory
    for file_name in os.listdir(src_image_dir):
        image_name = file_name.split('.')[0]
        try:
            print("file name: {}".format(file_name))
            image_path = os.path.join(src_image_dir, file_name)
            already = False
            for image_already in images_already_processed:
                if image_name in image_already:   
                    already = True
                    print(f"Skipping {image_name} as it already exists")
                    break

            if already: continue

            # Determine the prompt based on the image name
            if 'chair' in file_name:
                if one_person:
                    prompt = 'a person sitting on chair'
                else:
                    prompt = 'two person sitting on chair'
            elif 'bed' in file_name:
                if one_person:
                    prompt = 'a person sitting on bed'
                else:
                    prompt = 'two person sitting on bed'
            elif 'sofa' in file_name:
                if one_person:
                    prompt = 'a person sitting on sofa'
                else:
                    prompt = 'two person sitting on sofa'
            elif 'cycle' in file_name:
                if one_person:
                    prompt = 'a person riding a cycle'
                else:
                    prompt = 'two person, one riding a cycle and other standing'
            elif 'car' in file_name:
                if one_person:
                    prompt = 'a person sitting in a car'
                else:
                    prompt = 'two person sitting in a car'
            elif 'bike' in file_name:
                if one_person:
                    prompt = 'a person riding a motorbike'
                else:
                    prompt = 'two person riding a bike'
            elif 'outdoor' in file_name:
                if one_person:
                    prompt = 'a person standing in a outdoor'
                else:
                    prompt = 'two person standing in a outdoor'
            elif 'road' in file_name:
                if one_person:
                    prompt = 'a person walking on a road'
                else:
                    prompt = 'two person walking on a road'
            elif 'tajmahal' in file_name:
                if one_person:
                    prompt = 'a person walking there'
                else:
                    prompt = 'two person walking there'
            elif 'bench' in file_name:
                if one_person:
                    prompt = 'a person sitting on a bench'
                else:
                    prompt = 'two person sitting on a bench'
            elif 'auto'  in file_name:
                prompt = 'a person sitting in a auto rickshaw'
            elif 'pool' in file_name:
                prompt = 'a person playing pool ball'
            elif 'podium' in file_name:
                prompt = 'a person standing on a podium'
            elif 'stairs' in file_name:
                prompt = 'a person sitting on stairs'
            elif 'side' in file_name:
                prompt = 'a person walking on sidewalk'
            elif 'walk' in file_name:
                prompt = 'a person walking'
            elif 'running' in file_name:
                prompt = 'a person running there'
            elif 'effil' in file_name:
                prompt = 'a person standing near effil tower'
            else:
                if one_person:
                    prompt = 'a person sitting'
                else:
                    prompt = 'two person sitting'
        
            print(prompt)     
            # prompt = 'a person standing in a room'
            print("\n\n\n----",file_name,"----\n\n")
            # Run experiments for different iterations
            for initial_steps in iterations:

                with torch.no_grad():
                    torch.cuda.empty_cache()
                guidance = threestudio.find(config['guidance_type'])(config['guidance'])
                update_config_for_image(image_path, prompt, n_iters, initial_steps, config)
                prompt_processor = threestudio.find(config['prompt_processor_type'])(config['prompt_processor'])
                prompt_processor.configure_text_encoder()
                run(config, guidance, prompt_processor,root_seed=0)  # Your run function should handle different iterations
        except Exception as e:
            print(e)
            
        # break

if __name__ == "__main__":
    # Replace sys.stdout with the custom Logger
    log_filename = f'{output_path}/{output_folder_name}_{thresold}/log.txt'
    sys.stdout = Logger(log_filename)
    run_main()
    sys.stdout.logfile.close()








