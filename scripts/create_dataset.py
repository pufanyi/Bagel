import os
import glob
import json
from datasets import load_dataset, Dataset, DatasetDict, Image
from PIL import Image as PILImage
import argparse

def create_bagel_dataset(args):
    """
    Processes the Bagel example data, creates a Hugging Face DatasetDict,
    and optionally pushes it to the Hugging Face Hub.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    base_path = args.data_path
    output_path = args.output_path

    print(f"Processing data from: {base_path}")

    # --- 1. Process t2i data ---
    t2i_path = os.path.join(base_path, 't2i')
    t2i_files = glob.glob(os.path.join(t2i_path, '*.parquet'))
    if not t2i_files:
        raise FileNotFoundError(f"No parquet files found in {t2i_path}")
    print(f"Found {len(t2i_files)} parquet files for t2i dataset.")
    t2i_dataset = load_dataset('parquet', data_files=t2i_files, split='train')
    print("Loaded t2i dataset:")
    print(t2i_dataset)

    # --- 2. Process editing data ---
    editing_path = os.path.join(base_path, 'editing', 'seedxedit_multi')
    editing_files = glob.glob(os.path.join(editing_path, '*.parquet'))
    if not editing_files:
        raise FileNotFoundError(f"No parquet files found in {editing_path}")
    print(f"\nFound {len(editing_files)} parquet files for editing dataset.")
    editing_dataset = load_dataset('parquet', data_files=editing_files, split='train')
    print("Loaded editing dataset:")
    print(editing_dataset)

    # --- 3. Process vlm data ---
    vlm_path = os.path.join(base_path, 'vlm')
    vlm_jsonl_path = os.path.join(vlm_path, 'llava_ov_si.jsonl')
    vlm_images_path = os.path.join(vlm_path, 'images')

    if not os.path.exists(vlm_jsonl_path):
        raise FileNotFoundError(f"Metadata file not found: {vlm_jsonl_path}")
    if not os.path.exists(vlm_images_path):
        raise FileNotFoundError(f"Images directory not found: {vlm_images_path}")

    print(f"\nProcessing VLM data from {vlm_jsonl_path}")

    def vlm_generator():
        with open(vlm_jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                image_path = os.path.join(vlm_images_path, data['image'])
                if os.path.exists(image_path):
                    yield {
                        "image": image_path,
                        "conversations": data['conversations'],
                        "id": data.get('id')
                    }
                else:
                    print(f"Warning: Image not found {image_path}")

    vlm_dataset = Dataset.from_generator(vlm_generator).cast_column("image", Image())
    print("Loaded vlm dataset:")
    print(vlm_dataset)

    # --- 4. Save to disk or push to Hub ---
    if args.push_to_hub:
        print(f"\nPushing 't2i' dataset to Hugging Face Hub repository: {args.repo_id}...")
        t2i_dataset.push_to_hub(repo_id=args.repo_id, config_name="t2i", private=args.private, split="train")
        print("'t2i' dataset pushed successfully!")

        print(f"\nPushing 'editing' dataset to Hugging Face Hub repository: {args.repo_id}...")
        editing_dataset.push_to_hub(repo_id=args.repo_id, config_name="editing", private=args.private, split="train")
        print("'editing' dataset pushed successfully!")

        print(f"\nPushing 'vlm' dataset to Hugging Face Hub repository: {args.repo_id}...")
        vlm_dataset.push_to_hub(repo_id=args.repo_id, config_name="vlm", private=args.private, split="train")
        print("'vlm' dataset pushed successfully!")

        print("\nAll datasets have been pushed to the Hub.")

    else:
        # Combine into a DatasetDict for local saving
        final_dataset = DatasetDict({
            't2i': t2i_dataset,
            'editing': editing_dataset,
            'vlm': vlm_dataset
        })
        print("\nFinal combined dataset for local saving:")
        print(final_dataset)
        print(f"\nSaving dataset to {output_path}...")
        final_dataset.save_to_disk(output_path)
        print("Dataset saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Create a Hugging Face dataset from the Bagel example data and optionally push to the Hub.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="temp/data/bagel_example",
        help="Path to the source bagel_example directory."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="bagel_hf_dataset",
        help="Path to save the processed Hugging Face dataset locally. (Ignored if --push_to_hub is used)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, push the dataset to the Hugging Face Hub instead of saving locally."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="pufanyi/bagel-example",
        help="The ID of the repository on the Hugging Face Hub (e.g., 'username/dataset-name')."
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If set, creates a private repository on the Hub."
    )
    args = parser.parse_args()

    # Check for required libraries
    try:
        import datasets
        import PIL
        import huggingface_hub
    except ImportError as e:
        print("="*50)
        print(f"Error: Missing required library -> {e.name}")
        print("Please install the required packages by running:")
        print("uv pip install datasets Pillow huggingface_hub")
        print("="*50)
        exit(1)

    create_bagel_dataset(args)

if __name__ == '__main__':
    main()
