
# K-means clustering

This is a Python script that performs the k-means clustering algorithm on an image. The code uses the numba library for optimizing performance and the opencv library for image processing. The resulting clustered colors are visualized using matplotlib. The code is easily optimisable for parallel computation, and numba provides an easy way to implement a CUDA kernel. This allows for using many more pixels from the source image to be rendered.
## Usage

To use this code, follow these steps:

 - Ensure that all the required dependencies are installed
 - Modify the image variable to point to the path of your desired input image.
 - Adjust the MEANS variable to set the number of desired color clusters.
 - Run the script.
 - The resulting clustered colors will be printed and visualized in a 3D color space.

Note: 
- The code includes both CPU and GPU versions of the k-means algorithm. By default, the CPU version is used. To switch to the GPU version, set the cuda parameter of the kmeans function to True.
- The matplotlib plot is limited to show approximately 2000 points to make the graph interactive.
## Example

K-means with `k=16`

![Alt Text](https://cdn.discordapp.com/attachments/888995529210617900/1126570652900524032/image.png)
