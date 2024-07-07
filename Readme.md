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

1. Open `stable_diffusion_guidance.py` and make the following changes:
    ```python
    # Add your code changes here
    ```

2. Open `mask_generator.py` and make the following changes:
    ```python
    # Add your code changes here
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
