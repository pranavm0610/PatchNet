import argparse
from torch.utils.data import DataLoader
from data import GraphImageDataset  

def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--image_folder", type=str, required=True, help="Path to image folder")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to labels CSV/XLSX file")

    # Optional hyperparameters
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size for cropping")
    parser.add_argument("--num_patches", type=int, default=30, help="Max number of SURF patches")
    parser.add_argument("--overlap", type=float, default=0.3, help="Max allowed overlap ratio")
    parser.add_argument("--hessian_thresh", type=int, default=400, help="SURF hessian threshold")

    args = parser.parse_args()

    # Create dataset
    dataset = GraphImageDataset(
        image_folder=args.image_folder,
        csv_path=args.csv_path,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        overlap=args.overlap,
        hessian_thresh=args.hessian_thresh
    )

    # DataLoader example
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for grid_graph, surf_graph in loader:
        print("Grid graph nodes:", grid_graph.x.shape)
        print("Surf graph nodes:", surf_graph.x.shape)
        break

if __name__ == "__main__":
    main()
