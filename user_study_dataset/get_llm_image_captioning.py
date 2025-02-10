import openai
import base64
import shutil
import random
import os

# Set your OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

raw_prompt = """
Provide a caption for this image.
"""
focus_prompt = """
Provide a caption for this image. Focus on the person's activity.
"""
focus_few_shot_prompt = lambda x: f"""
Provide a caption for this image. Focus on the person's activity.\n
Follow the example captions:\n
"{x[0]}"\n
"{x[1]}"\n
"{x[2]}"
"""

with open("flicker_images/annotation.txt", "r") as f:
    annotations = [line.split(",")[1].strip() for line in f.readlines() if line.strip()]

def encode_image(image_path):
    """Convert image file to base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_caption(image_path, prompt_text):
    """Send image to OpenAI's vision model and get a caption."""
    base64_image = encode_image(image_path)
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # Use GPT-4 Vision model
        messages=[
            {"role": "system", "content": "You are an AI that describes images with a focus on people's activities."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        temperature=0.0,
        max_tokens=100
    )
    
    return response["choices"][0]["message"]["content"]

def main():
    image_dir = "flicker_images"
    save_dir = "flicker_human_gpt4o_captions"
    os.makedirs(save_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        try:
            image_id, traditional_label, annotation = filename.split(".")[0].split("_")
            image_path = os.path.join(image_dir, filename)
        except ValueError:
            print(f"Invalid filename: {filename}")
            continue

        raw_caption = get_caption(image_path, prompt_text=raw_prompt)
        focus_caption = get_caption(image_path, prompt_text=focus_prompt)
        random_annotations = random.sample(annotations, 3)
        focus_few_shot_caption = get_caption(image_path, prompt_text=focus_few_shot_prompt(random_annotations))

        os.makedirs(os.path.join(save_dir, traditional_label), exist_ok=True)
        new_filename = f"{image_id}.jpg"
        renamed_image_path = os.path.join(save_dir, traditional_label, new_filename)
        caption_file_path = os.path.join(save_dir, traditional_label, f"{image_id}_captions.txt")
        with open(caption_file_path, "w") as caption_file:
            caption_file.write(f"Image ID: {image_id}\n")
            caption_file.write(f"Traditional Label: {traditional_label}\n")
            caption_file.write(f"Human Annotation: {annotation}\n")
            caption_file.write(f"Raw Caption: {raw_caption}\n")
            caption_file.write(f"Only Activity Caption: {focus_caption}\n")
            caption_file.write(f"Only Activity & Few-Shot Caption: {focus_few_shot_caption}\n")

        shutil.copy(image_path, renamed_image_path)

if __name__ == "__main__":
    main()