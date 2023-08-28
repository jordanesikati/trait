#=================================================================================
# Libraries 
#=================================================================================

import open3d as o3d
import numpy as np
import time
import torch


#=================================================================================
# Read
#=================================================================================

def read_pcd(filename=''):
    
    '''
        This function read points clouds file using Open3D
        filename: points cloud file (.ply) 
    '''
    
    try:
        return o3d.io.read_point_cloud(str(filename))
    
    except Exception as e:
        print("An error occurred:", e)


#=================================================================================
# Visualization
#=================================================================================

def visualize_pcd(pcd, save_image=False, output_filename=None):
    '''
    This function shows the visual representation of a set of point clouds.
    pcd: set of points clouds
    save_image: flag to indicate whether to save the image or not
    output_filename: filename to save the image (required if save_image is True)
    '''
    
    # Check that the point cloud is not empty
    if len(pcd.points) == 0:
        print("Error: Point cloud is empty")
        
    else:
        # Compute the axis-aligned bounding box (AABB) of the point cloud
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)

        # Oriented bounding box
        obb = pcd.get_oriented_bounding_box()
        obb.color = (0, 1, 0)
        
        # Assign the color to the point cloud
        color = np.array([0, 1, 0])  # green
        
        # Create a mesh frame for XYZ axes
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.075)
        
        # Visualize point clouds
        if save_image and output_filename:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            # vis.add_geometry(aabb)
            vis.update_renderer()
            vis.poll_events()
            vis.capture_screen_image(output_filename)
            vis.destroy_window()
            
            # Rotate the visualization
            ctr = vis.get_view_control()
            ctr.rotate(45.0, 450.0)  # Example rotation angles, adjust as needed

            # Render the scene
            vis.update_renderer()
            
        else:
            # o3d.visualization.draw_geometries([pcd, aabb, mesh_frame])
            o3d.visualization.draw_geometries([pcd])


#=================================================================================
# Radius
#=================================================================================

def get_pcd_radius(pcd):

    # Get the center of the point cloud
    center = pcd.get_center()

    # Calculate the distance from the center to the farthest point in the point cloud
    distances = np.linalg.norm(np.asarray(pcd.points) - center, axis=1)
    min_radius = np.min(distances)
    max_radius = np.max(distances)
    mean_radius = np.mean(distances)
    std_radius = np.std(distances)

    return min_radius, max_radius, mean_radius, std_radius


#=================================================================================
# Experimental Find Optimal Radius
#=================================================================================

def find_optimal_radius(pcd, min_radius=0.01, max_radius=0.1, step=0.01, use_gpu=False):
    '''
    This function finds the optimal radius for segmenting the provided point clouds.
    
    pcd: point cloud file (.ply)
    min_radius: minimum radius to search for (default: 0.01)
    max_radius: maximum radius to search for (default: 0.1)
    step: step size to use for searching (default: 0.01)
    use_gpu: boolean to indicate if GPU should be used for computations (default: False)
    
    Returns the optimal radius and the segmented point clouds.
    '''
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Create list of radii to search within
    radii = np.arange(min_radius, max_radius+step, step)
    num_clusters = []
    
    # Loop over all radii and segment the point cloud
    for r in radii:
        segmented_pcd = segment_pcd(pcd, r, use_gpu)
        num_clusters.append(len(segmented_pcd.points))
    
    # Calculate the mean and standard deviation of the number of clusters for each radius
    mean_clusters = np.mean(num_clusters)
    std_clusters = np.std(num_clusters)
    
    # Find the radius with the minimum number of clusters, which is the optimal radius
    optimal_radius = radii[np.argmin(num_clusters)]
    
    # Segment the point cloud using the optimal radius
    segmented_pcd = segment_pcd(pcd, optimal_radius, use_gpu)
    
    print("Optimal radius:", optimal_radius)
    print("Minimum number of clusters:", min(num_clusters))
    print("Maximum number of clusters:", max(num_clusters))
    print("Mean number of clusters:", mean_clusters)
    print("Standard deviation of number of clusters:", std_clusters)
    
    return optimal_radius


#=================================================================================
# Segmentation
#=================================================================================

def segment_pcd(pcd, r, use_gpu=False):
    '''
        This function returns the segmentation result of the plant point clouds provided
        pcd: points cloud file (.ply)
        r: the radius of the point cloud
        use_gpu: boolean to indicate if GPU should be used for computations
    '''
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    start_time = time.time()

    # Convert point cloud to numpy array
    P = np.asarray(pcd.points)
    P = torch.from_numpy(P).to(device)

    segmented_cloud = []
    colors = []
    while P.shape[0] > 0:
        seed_point = P[0] # step 2, select seed point from P
        Q = torch.unsqueeze(seed_point, 0) # step 3, move seed point to Q
        Cn = torch.unsqueeze(seed_point, 0) # step 4, create new cluster Cn
        color = torch.rand(3, device=device) # step 5, assign random color to cluster
        i = 0
        while i < Q.shape[0]:
            pj = Q[i] # step 6, select pj from Q
            j = 0
            while j < P.shape[0]:
                pk = P[j] # step 8, select pk from P
                if torch.norm(pk - pj) < r: # step 9, check if pk is adjacent to pj
                    Q = torch.cat((Q, torch.unsqueeze(pk, 0)), dim=0) # step 10, move pk from P to Q
                    Cn = torch.cat((Cn, torch.unsqueeze(pk, 0)), dim=0) # add pk to Cn
                    P = torch.cat((P[:j], P[j+1:]), dim=0)
                else:
                    j += 1
            i += 1
        segmented_cloud.append(Cn.cpu().numpy()) # add cluster Cn to segmented point cloud
        colors.extend([color.cpu().numpy()]*Cn.shape[0]) # assign color to points in cluster

    # Convert segmented point cloud back to open3d point cloud
    segmented_pcd = o3d.geometry.PointCloud()
    segmented_pcd.points = o3d.utility.Vector3dVector(np.concatenate(segmented_cloud, axis=0))
    segmented_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    end_time = time.time()

    print("Number of clusters:", len(segmented_cloud))
    print("Processing time:", end_time - start_time, "seconds")
    
    return segmented_pcd

