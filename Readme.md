# Text2Place

## Setup Environment

1. Clone the repository:
    ```bash
    git clone https://github.com/Harsh-Gupta9897/Text2Place
    ```

2. Create a virtual environment (vnv) and activate it:
    ```bash
    python3 -m venv vnv
    source vnv/bin/activate
    ```

3. Install the required dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Generating Mask

In this section, we will generate a mask using the modified code from `stable_diffusion_guidance.py` and `mask_generator.py` files.

1. Open `stable_diffusion_guidance.py` located at `Text2Place/threestudio/threestudio/models/guidance/stable_diffusion_guidance.py` and make the following changes (it is optional):
    ```python
    # Add your code changes here
    # for multiple persons(n_persons), 
    # n= total number of blobs required, default=5 for each person
    self.n_persons = 1
    self.n = 5 * self.n_persons      

    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()

    # Centers for each blob, shape: n x 2, in case for static fixed centers
    # self.center = nn.Parameter(torch.tensor([[-1.0,-3.0],[-1.0,-1.0],[-1.0,-1.0],[-1.0,-3.0],[-1.0,2.0]]))

    # Centers for each blob, shape: 1 x 2, in case for dynamic centers
    self.center = nn.Parameter(torch.tensor([[-1.0,-3.0]]))

    # parameters for each blob
    self.scale = nn.Parameter(torch.ones(self.n) * 0.6)
    self.aspect_ratio = nn.Parameter(torch.ones(self.n) * 2.0)  # a ∈ R for each blob
    self.angle = nn.Parameter(torch.ones(self.n))  # θ ∈ [−π, π] for each blob

    # relative angles 
    self.relative_angles = nn.Parameter(torch.ones(self.n))
    self.radius = 0.1  # 0.07, 0.216, 0.13 - change this to play with the blob parameters
    ```

2. Open `mask_generator.py` and make the following changes:
    ```python
    dump_path = '/path/to/dump_dir'
    input_path = '/path/to/folder/images_folder_name_present'   # input_dir
    output_path = 'path/to/output_dir'   # output_dir
    output_folder_name = 'output_dir_name'  # output_dir_name
    input_folder_name = 'images_folder_name'  # input_bg_image_folder_path
    save_video_frames = False  # to save video frames in case to visualize it
    n_iters = 1000  # number of iterations to generate mask
    thresold = 0.2
    guidance_scale = 200.
    one_person = True
    ```

3. Run the modified code to generate the mask:
    ```bash
    python mask_generator.py
    ```

That's it! You have successfully generated a mask using the modified code.


## Inpainting with the above Generated mask.

Go Inside the `inpainting_pipeline` folder.
    ```cd inpainting_pipeline```


1. Generate the textual_embedding for particular subject by using `textual_inversion.py` file:
    ```
        bash generate_ti_run.sh
    ```

2. after getting the embedding, provide the path in `inpaint_run.sh` file, and run it

    ```
        bash inpaint_run.sh
    ```

After running the file , you will get the result in same output folder
