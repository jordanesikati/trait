from utils import read_pcd, visualize_pcd, segment_pcd, get_pcd_radius

if __name__ == "__main__":
    
    # Read the point cloud file
    pcd  = read_pcd(filename='./samples/sample.ply')

    # Compute the min, max, mean and standard deviation of pcd radius
    min_r, max_r, mean_r, std_r = get_pcd_radius(pcd)

    # Segmentation Ops
    seg = segment_pcd(pcd, r=min_r)

    # Show the result
    visualize_pcd(seg, save_image=False, output_filename="./result/segmentation2.png")