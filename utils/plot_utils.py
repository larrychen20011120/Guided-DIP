import matplotlib.pyplot as plt
import os

def plot_single_image(image, plot_method="plot", store_dir="assets"):
    
    if plot_method == "plot":
        plt.imshow(image)
        plt.show()
    else:
        path = os.path.join(store_dir, "single_image.jpg")
        plt.imsave(path, image)

def plot_snapshots(snapshots, row_count=3, plot_method="plot", store_dir="assets"):

    count = len(snapshots[-9:])
    column_count = (count-1) // row_count + 1
    if count < row_count:
        row_count = count
    
    if plot_method == "plot":
        plt.figure(figsize=(row_count*2, column_count*2))
        for i, image in enumerate(snapshots):
            plt.subplot(int(f"{row_count}{column_count}{i+1}"))
            plt.title(f"snapshot = {i+1}")
            plt.imshow(image)
        plt.tight_layout()
        plt.show()
    else:
        path = os.path.join(store_dir, "snapshots.jpg")
        plt.figure(figsize=(row_count*2, column_count*2))
        for i, image in enumerate(snapshots):
            plt.subplot(int(f"{row_count}{column_count}{i+1}"))
            plt.title(f"snapshot = {i+1}")
            plt.imshow(image)
        plt.tight_layout()
        plt.savefig(path)
