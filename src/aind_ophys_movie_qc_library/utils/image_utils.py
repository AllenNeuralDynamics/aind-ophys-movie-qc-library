from PIL import Image


def combine_images_vertically(
    images: list[Image.Image], padding: int = 10
) -> Image.Image:
    """Combine images vertically with padding

    Parameters
    ----------
    images: list[Image.Image]
        List of images
    padding: int
        Padding between images

    Returns
    -------
    Image.Image
        Combined image
    """

    print(f"Combining {len(images)} images")

    total_height = sum(img.height for img in images) + (len(images) - 1) * padding
    max_width = max(img.width for img in images)

    combined_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))
    y_offset = 0

    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height + padding

    return combined_image
