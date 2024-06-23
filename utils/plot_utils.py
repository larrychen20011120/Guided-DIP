import matplotlib.pyplot as plt
import os

def plot_single_image(image, plot_method="plot", store_dir="assets", filename="single_image"):
    
    if plot_method == "plot":
        plt.imshow(image)
        plt.show()
    else:
        path = os.path.join(store_dir, f"{filename}.jpg")
        plt.imsave(path, image)

def plot_snapshots(snapshots, plot_method="plot", store_dir="assets", filename="snapshots"):

    snapshots = snapshots[-9:]
    if plot_method == "plot":
        plt.figure(figsize=(6, 6))
        for i, image in enumerate(snapshots):
            plt.subplot(int(f"{3}{3}{i+1}"))
            plt.title(f"snapshot{i+1}")
            plt.imshow(image)
        plt.tight_layout()
        plt.show()
    else:
        path = os.path.join(store_dir, f"{filename}.jpg")
        plt.figure(figsize=(6, 6))
        for i, image in enumerate(snapshots):
            plt.subplot(int(f"{3}{3}{i+1}"))
            plt.title(f"snapshot = {i+1}")
            plt.imshow(image)
        plt.tight_layout()
        plt.savefig(path)

def plot_sequence(snapshots, plot_method="plot", store_dir="assets", filename="sequence"):
    
    snapshots = snapshots[-9:]
    if plot_method == "plot":
        plt.figure(figsize=(12, 4))
        for i, image in enumerate(snapshots):
            plt.subplot(int(f"{1}{9}{i+1}"))
            #plt.title(f"snapshot{i+1}")
            plt.imshow(image)
        plt.tight_layout()
        plt.show()
    else:
        path = os.path.join(store_dir, f"{filename}.jpg")
        plt.figure(figsize=(6, 6))
        for i, image in enumerate(snapshots):
            plt.subplot(int(f"{1}{9}{i+1}"))
            #plt.title(f"snapshot = {i+1}")
            plt.imshow(image)
        plt.tight_layout()
        plt.savefig(path)

def plot_psnr(psnrs, plot_method="plot", store_dir="assets", filename="psnr"):

    if plot_method == "plot":
        plt.figure(figsize=(12, 4))
        plt.plot(psnrs, 'ro-')
        plt.tight_layout()
        plt.show()
    else:
        path = os.path.join(store_dir, f"{filename}.jpg")
        plt.figure(figsize=(12, 4))
        plt.plot(psnrs, 'ro-')
        plt.tight_layout()
        plt.savefig(path)