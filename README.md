# Oriented Bounding Box
calculate Oriented Bounding Box from point cloud

1. Use pcl::EuclideanClusterExtraction to cluster the point cloud
2. Use pcl::computeCovarianceMatrixNormalized to find object axes
3. transform the point cloud to global origin, then find its length width height

<center><img src="https://img-blog.csdnimg.cn/20210630112548970.gif" width="75%" /></center>
