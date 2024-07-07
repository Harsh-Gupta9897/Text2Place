
import os
from diffusers import DiffusionPipeline,DDPMScheduler
import torch
from PIL import Image,ImageEnhance
import torchvision.transforms as T
from tqdm import auto
import random

# mitigate CUDA memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:50"
os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "0"

# A single token to be used during the learning process; should NOT be used in "prompts" below
learn_token="sks"
# start learning with an embedding of single token or "randn_like" 
start_token="randn_like"
# list of learning rates [(#steps,learning_rate)] ; 4 gradient accumulation steps per step
learning_rates=[(4,1e-3),(8,9e-4),(13,8e-4),(20,7e-4),(35,6e-4),(60,5e-4),(100,4e-4),(160,3e-4)]

# Templates for training: {} defines the token to be learned (learn_token)
template_prompts_for_objects=["a SLR photo of a {}","a photo of a {}","a rendering of a {}","a cropped photo of a {}","the rendering of a {}","a photo of a small {}","a photo of a fat {}","a rendering of a dirty {}","a dark photo of the {}","a rendering of a big {}","a 3D rendering of a {}","a close-up photo of a {}","a bright photo of the {}","a cropped photo of a {}","a rendering of the {}","an award winning photo of a {}","a photo of one {}","a close-up photo of the {}","a photo of the clean {}","a rendering of a nice {}","a good photo of a {}","a full body photo of a cute {}","a 3D rendering of the small {}","a photo of the weird {}","a photo of the large {}","a rendering of a cool {}","a SLR photo of a small {}"]
template_prompts_for_faces=["a color photo of {}","a national geograhic photo of {}","a national geograhic shot of {}","a shot of {}","a studio shot of {}", "a selfie of {}","a SLR photo of {}","a photo of {}","a studio photo of {}","a cropped photo of {}","a close-up photo of {}","an award winning photo of {}","a good photo of {}","a portrait photo of {}","a portrait shot of {}","a SLR photo of a cool {}","a SLR photo of the face of {}","a funny portrait of {}","{}, portrait shot","{}, studio lighting","{}, bokeh","{}, professional photo"]
template_prompts_for_styles=["a face in {} style","a portrait, {}","A {} portrait","{} showing a face","a portrait of a person depicted in a {}","{} showing a person","in style of {}","person ,{} style"]

# Define prompts for training
prompts=template_prompts_for_faces
#prompts=template_prompts_for_styles
negative_prompt="deformed, ugly, disfigured, blurry, pixelated, hideous, indistinct, old, malformed, extra hands, extra arms, joined misshapen, collage, grainy, low, poor, monochrome, huge, extra fingers, mutated hands, cropped off, out of frame, poorly drawn hands, mutated hands, fused fingers, too many fingers, fused finger, closed eyes, cropped face, blur, long body, people, watermark, text, logo, signature, text, logo, writing, heading, no text, logo, wordmark, writing, heading, signature, 2 heads, 2 faces, b&w, nude, naked"

prompt_variations=["men, white background","beard guy, white background","white t-shirt, white background"]

# INPUT images
#imgs_wh=(1024,1024) # 25 min for 500 steps (3090TI) -> noisy when used with lower INPUT image resolution
imgs_wh=(768,768) # 15 min for 500 steps (3090TI) -> good results
#imgs_wh=(512,512) # 10 min for 500 steps (3090TI) -> fastest
imgs_flip=True # additionally use horizontally mirrored INPUT images

# OUTPUT embedding
embs_path="/data/home/harshg/rishubh_person_scene/Embeddings/"
os.makedirs(embs_path,exist_ok=True)
os.makedirs("/data/home/harshg/rishubh_person_scene/inpainting_results/Samples",exist_ok=True)
emb_file="lambo.pt"

# Visualize intermediate optimization steps
test_prompt="a {} standing near effil tower"
intermediate_steps=9
outGIF="/data/home/harshg/rishubh_person_scene/inpainting_results/Samples/lambo.gif"



base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.bfloat16,
    variant="fp16", 
    # use_safetensors=False,
    add_watermarker=False,
    scheduler = DDPMScheduler(num_train_timesteps=1000,prediction_type="epsilon",beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)
)
base.disable_xformers_memory_efficient_attention()
torch.set_grad_enabled(True)
_=base.to("cuda")


def force_training_grad(model,bT=True,bG=True):
    model.training=bT
    model.requires_grad_=bG
    for module in model.children():
        force_training_grad(module,bT,bG)
        
def load_imgs(path,wh=(1024,1024),flip=True,preview=(64,64)):
    files=list()
    imgs=list()
    PILimgs=list()
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in [f for f in filenames if (f.endswith(".jpg") or f.endswith(".JPG") or f.endswith(".png") or f.endswith(".JPEG") or f.endswith(".jpeg"))]:
            fname = os.path.join(dirpath, filename)
            files.append(fname)
    for f in files:
        img = Image.open(f).convert("RGB")
        img = T.RandomAutocontrast(p=1.0)(img)
        img = T.Resize(wh, interpolation=T.InterpolationMode.LANCZOS)(img)
        PILimgs.append(T.Resize(preview, interpolation=T.InterpolationMode.LANCZOS)(img))
        img0 = T.ToTensor()(img)
        img0 = img0 *2.- 1.0
        imgs.append(img0[None].clip(-1.,1.))
        if flip:
            img0 = T.RandomHorizontalFlip(p=1.0)(img0)  
            imgs.append(img0[None].clip(-1.,1.)) 
            img = T.RandomHorizontalFlip(p=1.0)(img)
            PILimgs.append(T.Resize(preview, interpolation=T.InterpolationMode.LANCZOS)(img))
    return imgs,PILimgs

def make_grid(imgs):
    n=len(imgs)
    cols=1
    while cols*cols<n:
        cols+=1
    rows=n//cols+int(n%cols>0)
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))  
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def save_XLembedding(emb,embedding_file="myToken.pt",path="./Embeddings/"):
    torch.save(emb,path+embedding_file)

def set_XLembedding(base,emb,token="my"):
    with torch.no_grad():            
        # Embeddings[tokenNo] to learn
        tokens=base.components["tokenizer"].encode(token)
        print(tokens)
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


def XL_textual_inversion(base,imgs,prompts,prompt_variations=None,token="my",start_token=None,negative_prompt=None,learning_rates=[(5,1e-3),(10,9e-4),(20,8e-4),(35,7e-4),(55,6e-4),(80,5e-4),(110,4e-4),(145,3e-4)],intermediate_steps=9):
    
    XLt1=base.components["text_encoder"]
    XLt2=base.components["text_encoder_2"]
    XLtok1=base.components["tokenizer"]
    XLtok2=base.components["tokenizer_2"]
    XLunet=base.components["unet"]
    XLvae=base.components['vae']
    XLsch=base.components['scheduler']
    base.upcast_vae() # vae does not work correctly in 16 bit mode -> force fp32
    
    # Check Scheduler
    schedulerType=XLsch.config.prediction_type
    assert schedulerType in ["epsilon","sample"], "{} scheduler not supported".format(schedulerType)

    # Embeddings to Finetune
    embs=XLt1.text_model.embeddings.token_embedding.weight
    embs2=XLt2.text_model.embeddings.token_embedding.weight

    with torch.no_grad():       
        # Embeddings[tokenNo] to learn
        tokens=XLtok1.encode(token)
        print(tokens)
        assert len(tokens)==3, "token is not a single token in 'tokenizer'"
        tokenNo=tokens[1]
        tokens=XLtok2.encode(token)
        assert len(tokens)==3, "token is not a single token in 'tokenizer_2'"
        tokenNo2=tokens[1]            

        # init Embedding[tokenNo] with noise or with a copy of an existing embedding
        if start_token=="randn_like" or start_token==None:
            # Original value range: [-0.5059,0.6538] # regular [-0.05,+0.05]
            embs[tokenNo]=(torch.randn_like(embs[tokenNo])*.01).clone() # start with [-0.04,+0.04]
            # Original value range 2: [-0.6885,0.1948] # regular [-0.05,+0.05]
            embs2[tokenNo2]=(torch.randn_like(embs2[tokenNo2])*.01).clone() # start [-0.04,+0.04]
            startNo="~"
            startNo2="~"
        else:  
            tokens=XLtok1.encode(start_token)
            assert len(tokens)==3, "start_token is not a single token in 'tokenizer'"
            startNo=tokens[1]
            tokens=XLtok2.encode(start_token)
            assert len(tokens)==3, "start_token is not a single token in 'tokenizer_2'"
            startNo2=tokens[1]
            embs[tokenNo]=embs[startNo].clone()
            embs2[tokenNo2]=embs2[startNo2].clone()

        # Make a copy of all embeddings to keep all but the embedding[tokenNo] constant 
        index_no_updates = torch.arange(len(embs)) != tokenNo
        orig=embs.clone()
        index_no_updates2 = torch.arange(len(embs2)) != tokenNo2
        orig2=embs2.clone()
 
        print("Begin with '{}'=({}/{}) for '{}'=({}/{})".format(start_token,startNo,startNo2,token,tokenNo,tokenNo2))

        # Create all combinations [prompts] X [promt_variations]
        if prompt_variations:
            token=token+" "
        else:
            prompt_variations=[""]            

        txt_prompts=list()
        for p in prompts:
            for c in prompt_variations:
                txt_prompts.append(p.format(token+c))
        noPrompts=len(txt_prompts)
        
        # convert imgs to latents
        samples=list()
        for img in imgs:
            samples.append(((XLvae.encode(img.to(XLvae.device)).latent_dist.sample(None))*XLvae.config.scaling_factor).to(XLunet.dtype)) # *XLvae.config.scaling_factor=0.13025:  0.18215    
        noSamples=len(samples)
           
        # Training Parameters
        batch_size=1
        acc_size=4
        total_steps=sum(i for i, _ in learning_rates)
        # record_every_nth step is recorded in the progression list
        record_every_nth=(total_steps//(intermediate_steps+1)+1)*acc_size
        total_steps*=acc_size

        # Prompt Parametrs
        lora_scale = [0.6]  
        time_ids = torch.tensor(list(imgs[0].shape[2:4])+[0,0]+[1024,1024]).to(XLunet.dtype).to(XLunet.device)

    
    with torch.enable_grad():
        # Switch Models into training mode
        force_training_grad(XLunet,True,True)
        force_training_grad(XLt1,True,True)
        force_training_grad(XLt2,True,True)
        XLt1.text_model.train()
        XLt2.text_model.train()
        XLunet.train()
        XLunet.enable_gradient_checkpointing()
       
        # Optimizer Parameters        
        learning_rates=iter(learning_rates+[(0,0.0)]) #dummy for last update
        sp,lr=next(learning_rates)
        optimizer = torch.optim.AdamW([embs,embs2], lr=lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)   # 1e-7
        optimizer.zero_grad()
                
        # Progrssion List collects intermediate and final embedding
        progression=list()
        emb=embs[tokenNo].clone()
        emb2=embs2[tokenNo2].clone()
        progression.append({"emb":emb,"emb2":emb2})
        
        # Display [min (mean) max] of embeddings & current learning rate during training
        desc="[{0:2.3f} ({1:2.3f}) +{2:2.3f}] [{3:2.3f} ({4:2.3f}) +{5:2.3f}] lr={6:1.6f}".format(
                        torch.min(emb.to(float)).detach().cpu().numpy(),
                        torch.mean(emb.to(float)).detach().cpu().numpy(),
                        torch.max(emb.to(float)).detach().cpu().numpy(),
                        torch.min(emb2.to(float)).detach().cpu().numpy(),
                        torch.mean(emb2.to(float)).detach().cpu().numpy(),
                        torch.max(emb2.to(float)).detach().cpu().numpy(),
                        lr)

        # Training Loop
        t=auto.trange(total_steps, desc=desc,leave=True)
        for i in t:
            # use random prompt, random time stepNo, random input image sample
            prompt=txt_prompts[random.randrange(noPrompts)]
            stepNo=torch.tensor(random.randrange(XLsch.config.num_train_timesteps)).unsqueeze(0).long().to(XLunet.device)
            sample=samples[random.randrange(noSamples)].to(XLunet.device)

            ### Target
            noise = torch.randn_like(sample).to(XLunet.device)
            target = noise
            noised_sample=XLsch.add_noise(sample,noise,stepNo)

            # Prediction
            (prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds) = base.encode_prompt(
                prompt=prompt,prompt_2=prompt,
                negative_prompt=negative_prompt,negative_prompt_2=negative_prompt,
                do_classifier_free_guidance=True,lora_scale=lora_scale)
            cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
            pred = XLunet.forward(noised_sample,stepNo,prompt_embeds,added_cond_kwargs=cond_kwargs)['sample']
                        
            # Loss
            loss = torch.nn.functional.mse_loss((pred).float(), (target).float(), reduction="mean")                  
            loss/=float(acc_size)
            loss.backward() 
            
            # One Optimization Step for acc_size gradient accumulation steps
            if ((i+1)%acc_size)==0:
                # keep Embeddings in normal value range
                torch.nn.utils.clip_grad_norm_(XLt1.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(XLt2.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                
                with torch.no_grad():                    
                    # keep Embeddings for all other tokens stable      
                    embs[index_no_updates]= orig[index_no_updates]
                    embs2[index_no_updates2]= orig2[index_no_updates2]      
                        
                    # Current Embedding
                    emb=embs[tokenNo].clone()        
                    emb2=embs2[tokenNo2].clone()        
                            
                    if ((i+1)%(record_every_nth))==0:
                        progression.append({"emb":emb,"emb2":emb2})
                        
                    # adjust learning rate?
                    sp-=1
                    if sp<1:
                        sp,lr=next(learning_rates)
                        for g in optimizer.param_groups:
                            g['lr'] = lr
                            
                    # update display
                    t.set_description("[{0:2.3f} ({1:2.3f}) +{2:2.3f}] [{3:2.3f} ({4:2.3f}) +{5:2.3f}] lr={6:1.6f}".format(
                        torch.min(emb.to(float)).detach().cpu().numpy(),
                        torch.mean(emb.to(float)).detach().cpu().numpy(),
                        torch.max(emb.to(float)).detach().cpu().numpy(),
                        torch.min(emb2.to(float)).detach().cpu().numpy(),
                        torch.mean(emb2.to(float)).detach().cpu().numpy(),
                        torch.max(emb2.to(float)).detach().cpu().numpy(),
                        lr))

        # append final Embedding
        progression.append({"emb":emb,"emb2":emb2})
        
        return progression
    
def create_training_progression_gif(base, progression, test_prompt, learn_token, negative_prompt, outGIF):
    prompt = test_prompt.format(learn_token)
    frames = []
    for emb in progression:
        set_XLembedding(base, emb, token=learn_token)
        with torch.no_grad():    
            torch.manual_seed(46)
            image = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=40,
                guidance_scale=7.5
            ).images
        frames.append(image[0])
    import imageio
    imageio.mimsave(outGIF, frames + [frames[-1]] * 2, format='GIF', duration=1.0)


def train_multiple_concepts(base, main_folder_path, embs_path, prompts, prompt_variations, learning_rates, intermediate_steps, test_prompt, outGIF_base, learn_token="sks", start_token="randn_like", negative_prompt=None, imgs_wh=(768,768), imgs_flip=True):
    for concept_folder in os.listdir(main_folder_path):
        concept_path = os.path.join(main_folder_path, concept_folder)
        if os.path.isdir(concept_path):
            try:
                print(f"Training on concept: {concept_folder}")

                # Load images for the concept
                imgs, PILimgs = load_imgs(concept_path, wh=imgs_wh, flip=imgs_flip)

                # Perform XL Textual Inversion
                progression = XL_textual_inversion(base, imgs=imgs[:3], prompts=prompts, prompt_variations=prompt_variations, token=learn_token, start_token=start_token, negative_prompt=negative_prompt, learning_rates=learning_rates, intermediate_steps=intermediate_steps)

                # Save final and intermediate embeddings
                save_XLembedding(progression[-1], embedding_file=f"{concept_folder}.pt", path=embs_path)
                save_XLembedding(progression, embedding_file=f"all_{concept_folder}.pt", path=embs_path)

                # Generate and save a GIF to visualize training progression
                outGIF = os.path.join(outGIF_base, f"{concept_folder}.gif")
                create_training_progression_gif(base, progression, test_prompt, learn_token, negative_prompt, outGIF)
                print("Saved GIF to: ", outGIF, "concept: ", concept_folder)
            except:
                pass



# Example usage
    
if __name__ == "__main__":
    main_folder_path = "/data/home/harshg/rishubh_person_scene/data/child_dataset/"
    embs_path = "/data/home/harshg/rishubh_person_scene/data/new_embeddings_child/"
    outGIF_base = "/data/home/harshg/rishubh_person_scene/data/gifs/"
    os.makedirs(outGIF_base, exist_ok=True)
    os.makedirs(embs_path, exist_ok=True)
    train_multiple_concepts(base, main_folder_path, embs_path, prompts, prompt_variations, learning_rates, intermediate_steps, test_prompt, outGIF_base)


