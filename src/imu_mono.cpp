#include <geometry_msgs/msg/detail/transform_stamped__struct.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl/cloud_iterator.h>
#include <pcl/common/centroid.h>
#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/callback_group.hpp>
#include <rclcpp/logging.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/empty.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "nav2_map_server/map_io.hpp"

#include <chrono>
#include <filesystem>

#include <sstream>
#include <mutex>
#include <queue>

#include <cv_bridge/cv_bridge.h>

// this is orb_slam3
#include "System.h"

#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;
using std::placeholders::_1, std::placeholders::_2;

class ImuMono : public rclcpp::Node {
public:
    ImuMono()
        : Node("imu_mono_node_cpp"),
          vocabulary_file_path(std::string(PROJECT_PATH) + "/ORB_SLAM3/Vocabulary/ORBvoc.txt"),
          inertial_ba1_(false), inertial_ba2_(false), tImage_(0.0)
    {
        // declare parameters
        declare_parameter("sensor_type", "imu-monocular");
        declare_parameter("use_pangolin", true);

        // get parameters
        sensor_type_param = get_parameter("sensor_type").as_string();
        use_pangolin = get_parameter("use_pangolin").as_bool();

        // define callback groups
        image_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        imu_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        slam_service_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        timer_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions image_options;
        image_options.callback_group = image_callback_group_;
        rclcpp::SubscriptionOptions imu_options;
        imu_options.callback_group = imu_callback_group_;

        // set the sensor type based on parameter
        ORB_SLAM3::System::eSensor sensor_type;
        if (sensor_type_param == "monocular") {
            sensor_type = ORB_SLAM3::System::MONOCULAR;
            settings_file_path = std::string(PROJECT_PATH) + "/config/Monocular/TUM-VI.yaml";
        } else if (sensor_type_param == "imu-monocular") {
            sensor_type = ORB_SLAM3::System::IMU_MONOCULAR;
            settings_file_path = std::string(PROJECT_PATH) + "/config/Monocular-Inertial/TUM-VI.yaml";
        } else {
            RCLCPP_ERROR(get_logger(), "Sensor type not recognized");
            rclcpp::shutdown();
        }

        RCLCPP_INFO_STREAM(get_logger(), "vocabulary_file_path: " << vocabulary_file_path);
        RCLCPP_INFO_STREAM(get_logger(), "settings_file_path: " << settings_file_path);

        // setup orb slam object
        orb_slam3_system_ = std::make_shared<ORB_SLAM3::System>(
            vocabulary_file_path, settings_file_path, sensor_type, use_pangolin, 0);

        // create publishers
        live_point_cloud_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("live_point_cloud", 10);
        pose_array_publisher_ = create_publisher<geometry_msgs::msg::PoseArray>("pose_array", 10);
        live_occupancy_grid_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>("live_occupancy_grid", 10);
        odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("orb_odom", 10);
        orb_image_publisher_ = create_publisher<sensor_msgs::msg::Image>("/orb_camera/image", 10);
        imu_publisher_ = create_publisher<sensor_msgs::msg::Imu>("/orb_camera/imu", 10);

        // create subscriptions
        rclcpp::QoS image_qos(rclcpp::KeepLast(10));
        image_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
        image_qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
        image_sub = create_subscription<sensor_msgs::msg::Image>(
            "/cam0/image_raw", image_qos,
            std::bind(&ImuMono::image_callback, this, _1), image_options);

        rclcpp::QoS imu_qos(rclcpp::KeepLast(100));
        imu_qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
        imu_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
        imu_sub = create_subscription<sensor_msgs::msg::Imu>(
            "/imu0", imu_qos,
            std::bind(&ImuMono::imu_callback, this, _1), imu_options);

        // tf broadcaster
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // create timer
        timer = create_wall_timer(500ms, std::bind(&ImuMono::timer_callback, this), timer_callback_group_);

        timestamp_ = generate_timestamp_string();

        std::string path = std::string(PROJECT_PATH) + "/output/" + timestamp_;
        if (!std::filesystem::create_directory(path)) {
            std::cout << "Failed to create output directory" << std::endl;
            return;
        }
        if (!std::filesystem::create_directory(path + "/cloud")) {
            std::cout << "Failed to create cloud directory" << std::endl;
            return;
        }
        if (!std::filesystem::create_directory(path + "/grid")) {
            std::cout << "Failed to create grid directory" << std::endl;
            return;
        }

        initialize_variables();
        RCLCPP_INFO(get_logger(), "ORB_SLAM3 node initialized successfully");
    }

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_point_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        // statistical outlier removal
        pcl::PointCloud<pcl::PointXYZ>::Ptr sor_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(100);
        sor.setStddevMulThresh(0.1);
        sor.filter(*sor_cloud);

        // radius outlier removal
        pcl::PointCloud<pcl::PointXYZ>::Ptr radius_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> radius_outlier;
        radius_outlier.setInputCloud(sor_cloud);
        radius_outlier.setRadiusSearch(0.1); 
        radius_outlier.setMinNeighborsInRadius(5); 
        radius_outlier.filter(*radius_cloud);

        return radius_cloud;
    }

    nav_msgs::msg::OccupancyGrid::SharedPtr point_cloud_to_occupancy_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        Eigen::Matrix<float, 4, 1> centroid;
        pcl::ConstCloudIterator<pcl::PointXYZ> cloud_iterator(*cloud);
        pcl::compute3DCentroid(cloud_iterator, centroid);

        float max_x = -std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        float min_x = std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();

        for (const auto &point : cloud->points) {
            if (point.x > max_x) max_x = point.x;
            if (point.y > max_y) max_y = point.y;
            if (point.x < min_x) min_x = point.x;
            if (point.y < min_y) min_y = point.y;
        }

        nav_msgs::msg::OccupancyGrid::SharedPtr occupancy_grid = std::make_shared<nav_msgs::msg::OccupancyGrid>();
        cloud->width = cloud->points.size();
        occupancy_grid->header.frame_id = "live_map";
        occupancy_grid->header.stamp = get_clock()->now();
        occupancy_grid->info.resolution = 0.05;
        occupancy_grid->info.width = std::abs(max_x - min_x) / occupancy_grid->info.resolution + 1;
        occupancy_grid->info.height = std::abs(max_y - min_y) / occupancy_grid->info.resolution + 1;
        occupancy_grid->info.origin.position.x = min_x;
        occupancy_grid->info.origin.position.y = min_y;
        occupancy_grid->info.origin.position.z = 0;
        occupancy_grid->info.origin.orientation.x = 0;
        occupancy_grid->info.origin.orientation.y = 0;
        occupancy_grid->info.origin.orientation.z = 0;
        occupancy_grid->info.origin.orientation.w = 1;
        occupancy_grid->data.resize(occupancy_grid->info.width * occupancy_grid->info.height, 0);
        
        for (const auto &point : cloud->points) {
            int x = (point.x - min_x) / occupancy_grid->info.resolution;
            int y = (point.y - min_y) / occupancy_grid->info.resolution;
            int index = y * occupancy_grid->info.width + x;
            occupancy_grid->data.at(index) = 100;
        }
        return occupancy_grid;
    }

    void initialize_variables()
    {
        pose_array_ = geometry_msgs::msg::PoseArray();
        pose_array_.header.frame_id = "live_map";
        live_pcl_cloud_msg_ = sensor_msgs::msg::PointCloud2();
        live_pcl_cloud_msg_.header.frame_id = "live_map";
        live_occupancy_grid_ = std::make_shared<nav_msgs::msg::OccupancyGrid>();
    }

    std::string generate_timestamp_string()
    {
        std::time_t now = std::time(nullptr);
        std::tm *ptm = std::localtime(&now);
        std::ostringstream oss;
        oss << std::put_time(ptm, "%Y-%m-%d_%H-%M-%S");
        return oss.str();
    }

    cv::Mat get_image(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImageConstPtr cv_ptr;
        cv::Mat imageFrame;
        try {
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return imageFrame;
        }
        if (cv_ptr && !cv_ptr->image.empty()) {
            imageFrame = cv_ptr->image.clone();
        } else {
            RCLCPP_ERROR(get_logger(), "Imagen convertida es nula o vacía.");
        }
        return imageFrame;
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        rclcpp::Time dataset_time = msg->header.stamp;
        sensor_msgs::msg::Image::SharedPtr msg_out = std::make_shared<sensor_msgs::msg::Image>(*msg);
        msg_out->header.stamp = dataset_time;
        msg_out->header.frame_id = "base_link";
        orb_image_publisher_->publish(*msg_out);

        img_buf_.push(msg);

        while (!img_buf_.empty()) {
            auto imgPtr = img_buf_.front();
            img_buf_.pop();

            cv::Mat imageFrame = get_image(imgPtr);
            double tImage_local = imgPtr->header.stamp.sec + imgPtr->header.stamp.nanosec * 1e-9;

            vector<ORB_SLAM3::IMU::Point> vImuMeas;

            buf_mutex_imu_.lock();
            while (!imu_buf_.empty()) {
                auto imuPtr = imu_buf_.front();
                double tIMU = imuPtr->header.stamp.sec + imuPtr->header.stamp.nanosec * 1e-9;
                if (tIMU <= tImage_local) {
                    cv::Point3f acc(imuPtr->linear_acceleration.x, imuPtr->linear_acceleration.y, imuPtr->linear_acceleration.z);
                    cv::Point3f gyr(imuPtr->angular_velocity.x, imuPtr->angular_velocity.y, imuPtr->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, tIMU));
                    imu_buf_.pop();
                } else {
                    break;
                }
            }
            buf_mutex_imu_.unlock();

            if (vImuMeas.empty() && sensor_type_param == "imu-monocular") {
                RCLCPP_WARN(get_logger(), "No valid IMU data available for the current frame at time %.6f.", tImage_local);
                return;
            }

            orbslam3_mutex_.lock();
            try {
                if (sensor_type_param == "imu-monocular") {
                    if (!vImuMeas.empty()) {
                        auto Tcw = orb_slam3_system_->TrackMonocular(imageFrame, tImage_local, vImuMeas);
                        Tcw_.translation() = Tcw.translation();
                        Tcw_.setQuaternion(Tcw.unit_quaternion());
                    } else {
                        RCLCPP_WARN(get_logger(), "IMU data missing when required.");
                        orbslam3_mutex_.unlock();
                        return;
                    }
                } else {
                    auto Tcw = orb_slam3_system_->TrackMonocular(imageFrame, tImage_local);
                    Tcw_.translation() = Tcw.translation();
                    Tcw_.setQuaternion(Tcw.unit_quaternion());
                }
                this->tImage_ = tImage_local;
            } catch (const std::exception &e) {
                RCLCPP_ERROR(get_logger(), "SLAM processing exception: %s", e.what());
            }
            orbslam3_mutex_.unlock();
        }
    }

    void imu_callback(const sensor_msgs::msg::Imu &msg)
    {
        buf_mutex_imu_.lock();
        sensor_msgs::msg::Imu msg_out = msg;
        msg_out.header.frame_id = "base_link";
        if (!std::isnan(msg.linear_acceleration.x) && !std::isnan(msg.linear_acceleration.y) && !std::isnan(msg.linear_acceleration.z) &&
            !std::isnan(msg.angular_velocity.x) && !std::isnan(msg.angular_velocity.y) && !std::isnan(msg.angular_velocity.z)) {
            const sensor_msgs::msg::Imu::SharedPtr msg_ptr = std::make_shared<sensor_msgs::msg::Imu>(msg);
            imu_buf_.push(msg_ptr);
        } else {
            RCLCPP_ERROR(get_logger(), "Invalid IMU data - nan");
        }
        buf_mutex_imu_.unlock();
    }

    void timer_callback()
    {
        std::unique_lock<std::mutex> lock(orbslam3_mutex_, std::defer_lock);
        if (!lock.try_lock()) {
            return;
        }

        Sophus::SE3f Tcw_copy;
        double tImage_copy = 0.0;
        bool imu_initialized = false;
        bool ba1_status = false;
        bool ba2_status = false;
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZ>);

        
          imu_initialized = orb_slam3_system_-> isImuInitialized();
          if (imu_initialized) {
            Tcw_copy = Tcw_;
            tImage_copy = tImage_;
            
            ba1_status = orb_slam3_system_->GetInertialBA1();
            ba2_status = orb_slam3_system_->GetInertialBA2();

            if (orb_slam3_system_->GetTrackingState() == 2) {

                pcl::PointCloud<pcl::PointXYZ> map_data = orb_slam3_system_->GetMapPCL();
        
                if (!map_data.empty()) {
                    pcl::copyPointCloud(map_data, *cloud_copy);
                }
            }

            RCLCPP_INFO(this->get_logger(), "Puntos en la nube: %zu", cloud_copy->points.size());

            //printf("size of map_data %ld \n", sizeof(map_data)/sizeof(map_data[0]));
          }

          lock.unlock();
        

        if (!imu_initialized) {
          RCLCPP_INFO_STREAM(get_logger(), "IMU not initialized");
          initialize_variables();
          return;
        }

        int64_t total_nanoseconds = static_cast<int64_t>(tImage_ * 1e9);
        rclcpp::Time dataset_time(total_nanoseconds, RCL_ROS_TIME);

        auto Twc = Tcw_copy.inverse();

        tf2::Quaternion q_orig(Twc.unit_quaternion().x(), Twc.unit_quaternion().y(), Twc.unit_quaternion().z(), Twc.unit_quaternion().w());
        tf2::Matrix3x3 m(q_orig);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        tf2::Quaternion q_yaw;
        q_yaw.setRPY(0, 0, yaw);
        tf2::Quaternion q_rot_z;
        q_rot_z.setRPY(0, 0, M_PI / 2.0);

        tf2::Quaternion q_combined = q_rot_z * q_yaw;
        q_combined.normalize();

        geometry_msgs::msg::TransformStamped odom_tf;
        odom_tf.header.stamp = dataset_time;
        odom_tf.header.frame_id = "odom";
        odom_tf.child_frame_id = "base_link";
        odom_tf.transform.translation.x = Twc.translation().x();
        odom_tf.transform.translation.y = Twc.translation().y();
        odom_tf.transform.translation.z = Twc.translation().z();
        odom_tf.transform.rotation.x = Twc.unit_quaternion().x();
        odom_tf.transform.rotation.y = Twc.unit_quaternion().y();
        odom_tf.transform.rotation.z = Twc.unit_quaternion().z();
        odom_tf.transform.rotation.w = Twc.unit_quaternion().w();
        tf_broadcaster->sendTransform(odom_tf);

        nav_msgs::msg::Odometry odom;
        odom.header.stamp = dataset_time;
        odom.header.frame_id = "odom";
        odom.child_frame_id = "base_link";
        odom.pose.pose.position.x = Twc.translation().x();
        odom.pose.pose.position.y = Twc.translation().y();
        odom.pose.pose.position.z = Twc.translation().z();
        odom.pose.pose.orientation.x = Twc.unit_quaternion().x();
        odom.pose.pose.orientation.y = Twc.unit_quaternion().y();
        odom.pose.pose.orientation.z = Twc.unit_quaternion().z();
        odom.pose.pose.orientation.w = Twc.unit_quaternion().w();
        odom_publisher_->publish(odom);

        geometry_msgs::msg::Pose pose;
        pose.position.x = Twc.translation().x();
        pose.position.y = Twc.translation().y();
        pose.orientation.x = q_combined.x();
        pose.orientation.y = q_combined.y();
        pose.orientation.z = q_combined.z();
        pose.orientation.w = q_combined.w();
        pose_array_.header.stamp = dataset_time;
        pose_array_.poses.push_back(pose);
        pose_array_publisher_->publish(pose_array_);

        geometry_msgs::msg::TransformStamped point_cloud_tf;
        point_cloud_tf.header.stamp = dataset_time;
        point_cloud_tf.header.frame_id = "map";
        point_cloud_tf.child_frame_id = "point_cloud";
        tf_broadcaster->sendTransform(point_cloud_tf);

        geometry_msgs::msg::TransformStamped live_map_tf;
        live_map_tf.header.stamp = dataset_time;
        live_map_tf.header.frame_id = "map";
        live_map_tf.child_frame_id = "live_map";
        tf_broadcaster->sendTransform(live_map_tf);
        printf("hola");
        {
          if (!cloud_copy->empty()) {
            printf("\nif");
            try {
                printf("dentro del try");
                printf("sizeof %ld \n", sizeof cloud_copy); 
              auto filtered_cloud_ptr = filter_point_cloud(cloud_copy);
            //   filtered_cloud_ptr->width = filtered_cloud_ptr->points.size();

            //   printf("previo sensor");

            //   sensor_msgs::msg::PointCloud2 cloud_msg_;
            //   pcl::toROSMsg(*filtered_cloud_ptr, cloud_msg_);
            //   cloud_msg_.header.stamp = dataset_time;
            //   cloud_msg_.header.frame_id = "map";
            //   live_point_cloud_publisher_->publish(cloud_msg_);
                
            //   printf("after sensor");
            //   // this can go
            //   // auto grid = point_cloud_to_occupancy_grid(filtered_cloud_ptr);
            //   // grid->header.stamp = dataset_time;
            //   // grid->header.frame_id = "map";
            //   // live_occupancy_grid_publisher_->publish(*grid);


                // 
                
            } 
                catch (const std::exception &e) {
                    RCLCPP_ERROR(get_logger(), "Error procesando PCL: %s", e.what());
                }
          }
        }

        {
          std::lock_guard<std::mutex> lock(orbslam3_mutex_);
          if (!inertial_ba1_ && ba1_status) {
              inertial_ba1_ = true;
              initialize_variables();
              RCLCPP_INFO(get_logger(), "Inertial BA1 complete");
          }
          if (!inertial_ba2_ && ba2_status) {
              inertial_ba2_ = true;
              initialize_variables();
              RCLCPP_INFO(get_logger(), "Inertial BA2 complete");
          }
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr live_point_cloud_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_array_publisher_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr live_occupancy_grid_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr orb_image_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
    rclcpp::TimerBase::SharedPtr timer;

    rclcpp::CallbackGroup::SharedPtr image_callback_group_;
    rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
    rclcpp::CallbackGroup::SharedPtr slam_service_callback_group_;
    rclcpp::CallbackGroup::SharedPtr timer_callback_group_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

    sensor_msgs::msg::Imu imu_msg;
    geometry_msgs::msg::PoseArray pose_array_;

    std::string sensor_type_param;
    bool use_pangolin;

    std::vector<geometry_msgs::msg::Vector3> vGyro;
    std::vector<double> vGyro_times;
    std::vector<geometry_msgs::msg::Vector3> vAccel;
    std::vector<double> vAccel_times;

    queue<sensor_msgs::msg::Imu::SharedPtr> imu_buf_;
    queue<sensor_msgs::msg::Image::SharedPtr> img_buf_;
    std::mutex buf_mutex_imu_, buf_mutex_img_, orbslam3_mutex_, live_pcl_cloud_mutex_;

    std::shared_ptr<ORB_SLAM3::System> orb_slam3_system_;
    std::string vocabulary_file_path;
    std::string settings_file_path;

    sensor_msgs::msg::PointCloud2 live_pcl_cloud_msg_;
    pcl::PointCloud<pcl::PointXYZ> live_pcl_cloud_;
    nav_msgs::msg::OccupancyGrid::SharedPtr live_occupancy_grid_;

    bool inertial_ba1_;
    bool inertial_ba2_;
    Sophus::SE3f Tcw_;

    std::string timestamp_;
    double tImage_;
  };
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImuMono>());
    rclcpp::shutdown();
    return 0;
}
