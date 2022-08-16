from torchvision import transforms

def get_image_resize_transformers(new_image_size):
    return transforms.Compose([
        transforms.Resize(new_image_size),
    ])