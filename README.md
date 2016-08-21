# epuck_filter

ROS node implementing a simplified Kalman filter for fusion between: 
- wheel odometry of epuck robot
- pose of epuck robot provided by AR tag detection from an extern camera


TO DO :
- make use of covariances included in input topcis
- publish covariance of filtered odometry in output topic
