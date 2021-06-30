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
	// 1. ��������ļ�
	pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    pcl::PLYReader reader;
    reader.read("pointcloud.ply", *cloud);

	//2. ��ʼŷʽ����
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

	//3. ȡ����i�����࣬�û�����i��ֵ
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

	//4. �����i������
	pcl::PCLPointCloud2 blob;
	pcl::PLYWriter writer;
	std::string filename("cluster.ply");
	pcl::toPCLPointCloud2(*cloud_cluster, blob); // laserCloudFullResStacked
	writer.writeASCII(filename, blob);
	
	//5. ������Ƶļ������ĺ����ģ�ע�����������һ�������������е��ƽ��ֵ�������������С���ƽ��ֵ
	Eigen::Vector4f centroid;     // ����
	pcl::compute3DCentroid(*cloud_cluster, centroid);
	PointType min_pt, max_pt;
	Eigen::Vector3f center;       // ����
	pcl::getMinMax3D(*cloud_cluster, min_pt, max_pt); 
	center = (max_pt.getVector3fMap() + min_pt.getVector3fMap())/2;
	
	// 6. ����Э����������������
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud_cluster, centroid, covariance); // �������������
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA  = eigen_solver.eigenvalues();
	eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1)); //������
	eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
	eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));
	// ��������ֵ�Ӵ�С���У�eigenvalues�������ص�����ֵ�Ǵ�С�������еģ�
	Eigen::Matrix3f eigenVectorsPCA1;
	eigenVectorsPCA1.col(0) = eigenVectorsPCA.col(2);
	eigenVectorsPCA1.col(1) = eigenVectorsPCA.col(1);
	eigenVectorsPCA1.col(2) = eigenVectorsPCA.col(0);
	eigenVectorsPCA = eigenVectorsPCA1;

	// 7. ����任����ֻ������ȫ������ϵZ���ת��
	Eigen::Vector3f ea = (eigenVectorsPCA).eulerAngles(2, 1, 0); // �ֱ��Ӧ yaw pitch roll
	Eigen::AngleAxisf keep_Z_Rot(ea[0], Eigen::Vector3f::UnitZ());
	Eigen::Affine3f transform = Eigen::Affine3f::Identity();
	transform.translate(center);  // translate��rotate��д��˳�򲻿ɸ㷴
	transform.rotate(keep_Z_Rot); // radians
	
	// 8. �����Χ�еĳߴ�
	pcl::PointCloud<PointType>::Ptr transformedCloud(new pcl::PointCloud<PointType>);
	pcl::transformPointCloud(*cloud_cluster, *transformedCloud, transform.inverse());
	// ���¼���任��ĵ��Ƶ����ģ���Ϊ�任ǰ�ĵ��Ƶ����Ĳ������Χ�еļ�������
	PointType min_pt_T, max_pt_T;
	pcl::getMinMax3D(*transformedCloud, min_pt_T, max_pt_T);
	Eigen::Vector3f center_new = (max_pt_T.getVector3fMap() + min_pt_T.getVector3fMap()) / 2;
	Eigen::Vector3f box_dim;
	box_dim = max_pt_T.getVector3fMap() - min_pt_T.getVector3fMap();
	Eigen::Affine3f transform2 = Eigen::Affine3f::Identity();
	transform2.translate(center_new);
	Eigen::Affine3f transform3 = transform * transform2;

	//9. ��ʾ
	pcl::visualization::PCLVisualizer viewer;
	pcl::visualization::PointCloudColorHandlerCustom<PointType> handler_1(cloud_cluster, 0, 0, 255);	//����ĳ�ʼ����
	pcl::visualization::PointCloudColorHandlerCustom<PointType> handler_2(transformedCloud, 255, 0, 0);	//ת����ԭ��ĵ���
	viewer.addPointCloud(cloud_cluster, handler_1, "cloud1");
	viewer.addPointCloud(transformedCloud, handler_2, "cloud2");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud1");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud2");
	const Eigen::Quaternionf bboxQ(keep_Z_Rot);
	const Eigen::Vector3f    bboxT(transform3.translation()); // �������µ�"����"����ΪҪ��ʾ�İ�Χ�в��ü���������Ϊ�ο���
	viewer.addCube(bboxT, bboxQ, box_dim(0), box_dim(1), box_dim(2), "bbox");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "bbox");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "bbox");
	viewer.addCoordinateSystem(1.0);
	viewer.setBackgroundColor(1, 1, 1);
    viewer.spin();
    return 0;
}
