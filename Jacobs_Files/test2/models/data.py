#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from PIL import Image
import imagehash
from torchvision.datasets import ImageFolder

# def get_all_image_paths(dataset_root):
#     # ImageFolder requires transform even if not used
#     dataset = ImageFolder(root=dataset_root)
#     return [path for path, _ in dataset.samples]

# def compute_hash(image_path):
#     image = Image.open(image_path).convert("RGB")
#     return imagehash.average_hash(image)  # or use phash/dhash/whash

# # def image_exists_in_dataset(query_image_path, dataset_root):
# #     query_hash = compute_hash(query_image_path)
# #     all_paths = get_all_image_paths(dataset_root)

# #     for path in all_paths:
# #         if query_hash == compute_hash(path):
# #             print(f"Duplicate found: {path}")
# #             return True
# #     print("No duplicate found in dataset.")
# #     return False

# def image_exists_in_dataset(query_image_path, hash_list):
#     compute_hash(query_image_path)
#     if query_image_path in hash_list:
#         return True
#     return False
    
# Example usage
# query_image = "1.jpg"
# dataset_dir = "data/"
# image_exists_in_dataset(query_image, dataset_dir)


# In[ ]:


import os
import pickle
from PIL import Image
import imagehash
from torchvision.datasets import ImageFolder

HASH_FILE = "image_hashes.pkl"

def get_all_image_paths(dataset_root):
    dataset = ImageFolder(root=dataset_root)
    return [path for path, _ in dataset.samples]

def compute_hash(image_path):
    image = Image.open(image_path).convert("RGB")
    return str(imagehash.average_hash(image))

def save_dataset_hashes(dataset_root, hash_file=HASH_FILE):
    print("Computing and caching image hashes...")
    image_paths = get_all_image_paths(dataset_root)
    hashes = {path: compute_hash(path) for path in image_paths}
    with open(hash_file, "wb") as f:
        pickle.dump(hashes, f)
    print(f"Hashes saved to {hash_file}")

def load_hashes(hash_file=HASH_FILE):
    with open(hash_file, "rb") as f:
        return pickle.load(f)

def image_exists_in_dataset(query_image_path, dataset_root, hash_file=HASH_FILE):
    if not os.path.exists(hash_file):
        save_dataset_hashes(dataset_root, hash_file)

    dataset_hashes = load_hashes(hash_file)
    query_hash = compute_hash(query_image_path)

    for path, h in dataset_hashes.items():
        if h == query_hash:
            print(f"Duplicate found: {path}")
            return True

    print("No duplicate found in dataset.")
    return False

# Example usage



# In[4]:


# query_image = "1.jpg"
# dataset_dir = "data/"
# image_exists_in_dataset(query_image, dataset_dir)


# In[7]:


# query_image = "Untitled.jpeg"
# dataset_dir = "data/"
# image_exists_in_dataset(query_image, dataset_dir)

