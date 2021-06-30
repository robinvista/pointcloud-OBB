#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

typedef pcl::PointXYZI PointType;

int main(int argc, char* argv[])
{
	// 1. 读入点云文件
	pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    pcl::PLYReader reader;
    reader.read("pointcloud.ply", *cloud);

	//2. 开始欧式聚类
	pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
	tree->setInputCloud(cloud);
	std::vector<pcl::PointIndices> local_indices;
	pcl::EuclideanClusterExtraction<PointType> euclid;
	euclid.setInputCloud(cloud);
	float in_max_cluster_distance = 0.5;
	float MIN_CLUSTER_SIZE = 5;
	float MAX_CLUSTER_SIZE = 500000;
	euclid.setClusterTolerance(in_max_cluster_distance);
	euclid.setMinClusterSize(MIN_CLUSTER_SIZE);
	euclid.setMaxClusterSize(MAX_CLUSTER_SIZE);
	euclid.setSearchMethod(tree);
	euclid.extract(local_indices);

	//3. 取出第i个聚类，用户输入i的值
	std::cout << "Number of clusters = " << local_indices.size() << endl;
	size_t i = 0;
	pcl::PointCloud<PointType>::Ptr cloud_cluster(new pcl::PointCloud<PointType>());
	PointType point;
	for (auto pit = local_indices[i].indices.begin(); pit != local_indices[i].indices.end(); ++pit)
	{
		point.x = cloud->points[*pit].x;
		point.y = cloud->points[*pit].y;
		point.z = cloud->points[*pit].z;
		cloud_cluster->points.push_back(point);
	}

	//4. 保存第i个聚类
	pcl::PCLPointCloud2 blob;
	pcl::PLYWriter writer;
	std::string filename("cluster.ply");
	pcl::toPCLPointCloud2(*cloud_cluster, blob); // laserCloudFullResStacked
	writer.writeASCII(filename, blob);
	
	//5. 计算点云的几何中心和重心，注意这两个概念不一样，重心是所有点的平均值，中心是最大最小点的平均值
	Eigen::Vector4f centroid;     // 重心
	pcl::compute3DCentroid(*cloud_cluster, centroid);
	PointType min_pt, max_pt;
	Eigen::Vector3f center;       // 中心
	pcl::getMinMax3D(*cloud_cluster, min_pt, max_pt); 
	center = (max_pt.getVector3fMap() + min_pt.getVector3fMap())/2;
	
	// 6. 计算协方差矩阵的特征向量
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud_cluster, centroid, covariance); // 这里必须用重心
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA  = eigen_solver.eigenvalues();
	eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1)); //正交化
	eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
	eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));
	// 按照特征值从大到小排列（eigenvalues函数返回的特征值是从小到大排列的）
	Eigen::Matrix3f eigenVectorsPCA1;
	eigenVectorsPCA1.col(0) = eigenVectorsPCA.col(2);
	eigenVectorsPCA1.col(1) = eigenVectorsPCA.col(1);
	eigenVectorsPCA1.col(2) = eigenVectorsPCA.col(0);
	eigenVectorsPCA = eigenVectorsPCA1;

	// 7. 计算变换矩阵，只考虑绕全局坐标系Z轴的转动
	Eigen::Vector3f ea = (eigenVectorsPCA).eulerAngles(2, 1, 0); // 分别对应 yaw pitch roll
	Eigen::AngleAxisf keep_Z_Rot(ea[0], Eigen::Vector3f::UnitZ());
	Eigen::Affine3f transform = Eigen::Affine3f::Identity();
	transform.translate(center);  // translate与rotate书写的顺序不可搞反
	transform.rotate(keep_Z_Rot); // radians
	
	// 8. 计算包围盒的尺寸
	pcl::PointCloud<PointType>::Ptr transformedCloud(new pcl::PointCloud<PointType>);
	pcl::transformPointCloud(*cloud_cluster, *transformedCloud, transform.inverse());
	// 重新计算变换后的点云的中心，因为变换前的点云的中心不是其包围盒的几何中心
	PointType min_pt_T, max_pt_T;
	pcl::getMinMax3D(*transformedCloud, min_pt_T, max_pt_T);
	Eigen::Vector3f center_new = (max_pt_T.getVector3fMap() + min_pt_T.getVector3fMap()) / 2;
	Eigen::Vector3f box_dim;
	box_dim = max_pt_T.getVector3fMap() - min_pt_T.getVector3fMap();
	Eigen::Affine3f transform2 = Eigen::Affine3f::Identity();
	transform2.translate(center_new);
	Eigen::Affine3f transform3 = transform * transform2;

	//9. 显示
	pcl::visualization::PCLVisualizer viewer;
	pcl::visualization::PointCloudColorHandlerCustom<PointType> handler_1(cloud_cluster, 0, 0, 255);	//输入的初始点云
	pcl::visualization::PointCloudColorHandlerCustom<PointType> handler_2(transformedCloud, 255, 0, 0);	//转换到原点的点云
	viewer.addPointCloud(cloud_cluster, handler_1, "cloud1");
	viewer.addPointCloud(transformedCloud, handler_2, "cloud2");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud1");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
	const Eigen::Quaternionf bboxQ(keep_Z_Rot);
	const Eigen::Vector3f    bboxT(transform3.translation()); // 这里用新的"中心"，因为要显示的包围盒采用几何中心作为参考点
	viewer.addCube(bboxT, bboxQ, box_dim(0), box_dim(1), box_dim(2), "bbox");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "bbox");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "bbox");
	viewer.addCoordinateSystem(1.0);
	viewer.setBackgroundColor(1, 1, 1);
    viewer.spin();
    return 0;
}
