#include "KalmanFilter.h"
#include "ros/ros.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <nav_msgs/Odometry.h>
//#include <geometry_msgs/Quaternion.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/Marker.h>

//Main Filtre de Kalman odometrie + pose aprilTag


// time instants
ros::Time t;
ros::Time tPrec;
bool firstIteration = true;

// filters
KalmanFilter* positionFilter;
KalmanFilter* orientationFilter;

// publishers
ros::Publisher pubFilteredOdom;
ros::Publisher rvizFilteredPub;
ros::Publisher rvizWOdomPub; // wheel odom
ros::Publisher rvizCPosePub; // camera

// messages to be published
nav_msgs::Odometry filteredOdomMsg;
visualization_msgs::Marker filteredMarker;
visualization_msgs::Marker wOdomMarker; // wheel odom
visualization_msgs::Marker cPoseMarker; // camera


// parameters
//double essaiVar;
double positionProcNoiseCov;
double positionMeasNoiseCov;
double positionXInit;
double positionYInit;
double positionInitCov;
double orientationProcNoiseCov;
double orientationMeasNoiseCov;
double orientationInit;
double orientationInitCov;


// init markers for visu in RViz
void initVisuMarkers(){
	
	// filtered pose
	filteredMarker.header.frame_id = "world"; //base_link";
	//filteredMarker.header.stamp = ros::Time();
	filteredMarker.ns = "";//"my_namespace";
	filteredMarker.id = 0;
	filteredMarker.type = visualization_msgs::Marker::ARROW;
	filteredMarker.action = visualization_msgs::Marker::ADD;
	filteredMarker.pose.position.x = 0;
	filteredMarker.pose.position.y = 0;
	filteredMarker.pose.position.z = 0;
	filteredMarker.pose.orientation.x = 0.0;
	filteredMarker.pose.orientation.y = 0.0;
	filteredMarker.pose.orientation.z = 0.0;
	filteredMarker.pose.orientation.w = 1.0;
	filteredMarker.scale.x = 1;
	filteredMarker.scale.y = 0.1;
	filteredMarker.scale.z = 0.1;
	filteredMarker.color.a = 1.0; // Don't forget to set the alpha!
	filteredMarker.color.r = 1.0;
	filteredMarker.color.g = 0.0;
	filteredMarker.color.b = 0.0;
	
	
	// wheel odometry pose
	wOdomMarker.header.frame_id = "world"; //base_link";
	//filteredMarker.header.stamp = ros::Time();
	wOdomMarker.ns = "";//"my_namespace";
	wOdomMarker.id = 0;
	wOdomMarker.type = visualization_msgs::Marker::ARROW;
	wOdomMarker.action = visualization_msgs::Marker::ADD;
	wOdomMarker.pose.position.x = 0;
	wOdomMarker.pose.position.y = 0;
	wOdomMarker.pose.position.z = 0;
	wOdomMarker.pose.orientation.x = 0.0;
	wOdomMarker.pose.orientation.y = 0.0;
	wOdomMarker.pose.orientation.z = 0.0;
	wOdomMarker.pose.orientation.w = 1.0;
	wOdomMarker.scale.x = 1;
	wOdomMarker.scale.y = 0.1;
	wOdomMarker.scale.z = 0.1;
	wOdomMarker.color.a = 1.0; // Don't forget to set the alpha!
	wOdomMarker.color.r = 0.0;
	wOdomMarker.color.g = 0.0;
	wOdomMarker.color.b = 1.0;
	
	
	// pose from camera
	cPoseMarker.header.frame_id = "world"; //base_link";
	//filteredMarker.header.stamp = ros::Time();
	cPoseMarker.ns = "";//"my_namespace";
	cPoseMarker.id = 0;
	cPoseMarker.type = visualization_msgs::Marker::ARROW;
	cPoseMarker.action = visualization_msgs::Marker::ADD;
	cPoseMarker.pose.position.x = 0;
	cPoseMarker.pose.position.y = 0;
	cPoseMarker.pose.position.z = 0;
	cPoseMarker.pose.orientation.x = 0.0;
	cPoseMarker.pose.orientation.y = 0.0;
	cPoseMarker.pose.orientation.z = 0.0;
	cPoseMarker.pose.orientation.w = 1.0;
	cPoseMarker.scale.x = 1;
	cPoseMarker.scale.y = 0.1;
	cPoseMarker.scale.z = 0.1;
	cPoseMarker.color.a = 1.0; // Don't forget to set the alpha!
	cPoseMarker.color.r = 0.0;
	cPoseMarker.color.g = 1.0;
	cPoseMarker.color.b = 0.0;
	
}


// init filters
void initFilters(){
	
	// **** position filter   ****
	positionFilter = new KalmanFilter(2,2,2);
	positionFilter->setSys( Eigen::MatrixXd::Identity(2,2), Eigen::MatrixXd::Identity(2,2), Eigen::MatrixXd::Identity(2,2) );
	double q_pos= positionProcNoiseCov;
	double r_pos = positionMeasNoiseCov;
	positionFilter->setNoiseCovMatrices( q_pos*Eigen::MatrixXd::Identity(2,2), r_pos*Eigen::MatrixXd::Identity(2,2) );
	double p0_pos = positionInitCov;
	Eigen::VectorXd pos0 = Eigen::VectorXd::Zero(2);
	pos0(0) = positionXInit;
	pos0(1) = positionYInit;
	positionFilter->initFilter(pos0, p0_pos*Eigen::MatrixXd::Identity(2,2));


	// **** orientation filter ****
	orientationFilter = new KalmanFilter(1,1,1);
	orientationFilter->setSys( Eigen::MatrixXd::Identity(1,1), Eigen::MatrixXd::Identity(1,1), Eigen::MatrixXd::Identity(1,1) );
	double q_orient = orientationProcNoiseCov; 
	double r_orient = orientationMeasNoiseCov;
	orientationFilter->setNoiseCovMatrices( q_orient*Eigen::MatrixXd::Identity(1,1), r_orient*Eigen::MatrixXd::Identity(1,1) );
	double p0_yaw = orientationInitCov;
	// TO DO : position initiale à récupérer en argument au lancement
	Eigen::VectorXd yaw0 = Eigen::VectorXd::Zero(1);
	yaw0(0) = orientationInit;
	orientationFilter->initFilter( yaw0, p0_yaw*Eigen::MatrixXd::Identity(1,1) );
	
}



// call back for wheel odom
void callbackWOdom(const nav_msgs::Odometry::ConstPtr& msg)
{

	// get time duration from last message
	t = msg->header.stamp;
	if (firstIteration) {
		tPrec = t;
	   firstIteration = false;
	}
	ros::Duration TeRos = t - tPrec;
	double Te = TeRos.toNSec() / 1.0e9;


	// get speed from wheel odom	
	Eigen::VectorXd speedOdom = Eigen::VectorXd::Zero(1);
	speedOdom(0) = msg->twist.twist.linear.x;


	// get angular speed form wheel odom
	Eigen::VectorXd rotSpeedOdom = Eigen::VectorXd::Zero(1);
	rotSpeedOdom(0) = msg->twist.twist.angular.z;

	// orientation filter prediction  
	orientationFilter->setU_k(Te*rotSpeedOdom);
	orientationFilter->predictOneStep();

	// get predicted orientation
	double yaw = orientationFilter->getX_k()(0);

	// *** TO DO : get covariance


	// position filter prediction
	Eigen::VectorXd input = Eigen::VectorXd::Zero(2);
	input(0) = Te*speedOdom(0) * cos(yaw);
	input(1) = Te*speedOdom(0) * sin(yaw);
	positionFilter->setU_k(input);
	positionFilter->predictOneStep();

	// get predicted position
	Eigen::VectorXd position = Eigen::VectorXd::Zero(2);
	position = positionFilter->getX_k();

	// *** TO DO : get covariance


	// build odom message to be published
	filteredOdomMsg.header.stamp = msg->header.stamp;//ros::Time::now();

	filteredOdomMsg.pose.pose.position.x = position(0);
	filteredOdomMsg.pose.pose.position.y = position(1);
	filteredOdomMsg.pose.pose.position.z = 0.0;

	tf::Quaternion quat(0.0, 0.0, yaw);
	filteredOdomMsg.pose.pose.orientation.x = quat.x();
	filteredOdomMsg.pose.pose.orientation.y = quat.y();
	filteredOdomMsg.pose.pose.orientation.z = quat.z();
	filteredOdomMsg.pose.pose.orientation.w = quat.w();

	// publish odom
	pubFilteredOdom.publish(filteredOdomMsg);


	// publish filtered marker in Rviz
	filteredMarker.pose.position.x = position(0);
	filteredMarker.pose.position.y = position(1);
	filteredMarker.pose.position.z = 0.0;
	filteredMarker.pose.orientation.x = quat.x();
	filteredMarker.pose.orientation.y = quat.y();
	filteredMarker.pose.orientation.z = quat.z();
	filteredMarker.pose.orientation.w = quat.w();
	rvizFilteredPub.publish( filteredMarker );

	// publish wheel odom marker in Rviz
    wOdomMarker.pose.position.x = msg->pose.pose.position.x;
	wOdomMarker.pose.position.y = msg->pose.pose.position.y;
	wOdomMarker.pose.position.z = 0.0;
	wOdomMarker.pose.orientation = msg->pose.pose.orientation;
	rvizWOdomPub.publish( wOdomMarker );

	// update time
	tPrec = t;
   
   
}


// call back for Pose from camera
void callbackCPose(const nav_msgs::Odometry::ConstPtr& msg)
{

	ROS_INFO("  measurement received");

	// get position measurement
	Eigen::VectorXd positionOdom = Eigen::VectorXd::Zero(2);
	positionOdom(0) = msg->pose.pose.position.x;
	positionOdom(1) = msg->pose.pose.position.y;

	// position filter update
	positionFilter->setY_k(positionOdom);
	positionFilter->updateOneStep();

	// get orientation measurement
	tf::Quaternion quaternion(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
	tf::Matrix3x3 orientationMatrix(quaternion);
	double yaw = 0.0;
	double roll, pitch;
	orientationMatrix.getRPY(roll, pitch, yaw);
	Eigen::VectorXd angleOdom = Eigen::VectorXd::Zero(1);
	angleOdom(0) = yaw;

	//orientation filter update
	orientationFilter->setY_k(angleOdom); 
	orientationFilter->updateOneStep();


	// get updated orientation
	double yawF = orientationFilter->getX_k()(0);
    
    // get predicted position
	Eigen::VectorXd position = Eigen::VectorXd::Zero(2);
	position = positionFilter->getX_k();


	// publish pose from camera marker in Rviz
	cPoseMarker.pose.position.x = msg->pose.pose.position.x;
	cPoseMarker.pose.position.y = msg->pose.pose.position.y;
	cPoseMarker.pose.position.z = 0.0;
	cPoseMarker.pose.orientation = msg->pose.pose.orientation;
	rvizCPosePub.publish( cPoseMarker );

}


int main(int argc, char **argv)
{

	// node
	ros::init(argc, argv, "epuck_filter");
	ros::NodeHandle n;
	ROS_INFO("Epuck filter node started");

	// parameters
	//n.getParam("/epuck_filter/essai", essaiVar);
	//n.param("/epuck_filter/essai", essaiVar, 23.01);
	n.param("/epuck_filter/positionProcNoiseCov",positionProcNoiseCov, 0.05*0.05);
	n.param("/epuck_filter/positionMeasNoiseCov",positionMeasNoiseCov, 0.01*0.01);
	n.param("/epuck_filter/positionXInit",positionXInit, 0.0);
	n.param("/epuck_filter/positionYInit",positionYInit, 0.0);
	n.param("/epuck_filter/positionInitCov",positionInitCov, 0.01*0.01);
	n.param("/epuck_filter/orientationProcNoiseCov",orientationProcNoiseCov, pow(2*3.14159265359/180.0,2) );
	n.param("/epuck_filter/orientationMeasNoiseCov",orientationMeasNoiseCov, pow(1*3.14159265359/180.0,2) );
	n.param("/epuck_filter/orientationInit",orientationInit, 0.0);
	n.param("/epuck_filter/orientationInitCov",orientationInitCov, pow(5*3.14159265359/180.0,2) );


	// published topics
	pubFilteredOdom = n.advertise<nav_msgs::Odometry>("filteredOdom", 1000);
	rvizFilteredPub = n.advertise<visualization_msgs::Marker>( "filteredMarker", 10 );
	rvizWOdomPub = n.advertise<visualization_msgs::Marker>("wOdomMarker",10);
	rvizCPosePub = n.advertise<visualization_msgs::Marker>("cPoseMarker", 10);
	

	// init visualisation markers
	initVisuMarkers();

	// init filters
	initFilters();
	ROS_INFO("  filters initialized");
	

	//ros::Duration(2).sleep();


	// subscribed topics
	ros::Subscriber subOdom = n.subscribe("odom", 100, callbackWOdom);
	ros::Subscriber subPose = n.subscribe("pose", 100, callbackCPose);


	// main loop
	ros::spin();


	return 0;
}
