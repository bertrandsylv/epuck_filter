#ifndef H_INCLUDED_KALMAN_FILTER
#define H_INCLUDED_KALMAN_FILTER

#include <Eigen/Dense>
#include <Eigen/LU>



class KalmanFilter{

private:
	// état estimé X(k) et matrice de covariance P(k)
	unsigned int StateDim;
	Eigen::VectorXd X_k;
	Eigen::MatrixXd P_k;

	// matrice de covariance Q(k) du bruit d'état
	Eigen::MatrixXd Q_k;

	// mesure Y(k) et matrice de covariance R(k) du bruit de mesure
	unsigned int MeasurementDim;
	Eigen::VectorXd Y_k;
	Eigen::MatrixXd R_k;

	// entrée U(k)
	unsigned int InputDim;
	Eigen::VectorXd U_k;

	
	// système  : 
	//	X(k+1) = A(k)*X(k) + B(k)*U(k) + w(k)
	//	Y(k) = C(k)*X(k) + v(k)
	Eigen::MatrixXd A_k;
	Eigen::MatrixXd B_k;
	Eigen::MatrixXd C_k;

	
	// affichage objet KalmanFilter
	friend std::ostream& operator<<(std::ostream& os, const KalmanFilter& KF);


public:
	// constructeurs
	KalmanFilter();
	KalmanFilter(unsigned int StateDim, unsigned int InputDim, unsigned int MeasurementDim);

	// desctructeur
	~KalmanFilter();
	
	// initialisations rapides
	void setSys(Eigen::MatrixXd A_k, Eigen::MatrixXd B_k, Eigen::MatrixXd C_k);
	void setNoiseCovMatrices(Eigen::MatrixXd Q_k, Eigen::MatrixXd R_k);
	void initFilter(Eigen::VectorXd X_0, Eigen::MatrixXd P_0);

	// prédiction
	void predictOneStep(); // état X(k|k-1) et matrice de covariance P(k|k-1)

	// mise à jour
	void updateOneStep(); // état X(k|k) et matrice de covariance P(k|k)

	// renvoi de l'innovation et de sa covariance
	Eigen::VectorXd getInnov();
	Eigen::MatrixXd getInnovCovMatrix();	

	// accesseurs
	void setStateDim(unsigned int StateDim);
	unsigned int getStateDim();
	void setX_k(Eigen::VectorXd X_k);
	Eigen::VectorXd getX_k();
	void setP_k(Eigen::MatrixXd P_k);
	Eigen::MatrixXd getP_k();

	void setQ_k(Eigen::MatrixXd Q_k);
	Eigen::MatrixXd getQ_k();

	void setMeasurementDim(unsigned int MeasurementDim);
	unsigned int getMeasurementDim();
	void setY_k(Eigen::VectorXd Y_k);
	Eigen::VectorXd getY_k();
	void setR_k(Eigen::MatrixXd R_k);
	Eigen::MatrixXd getR_k();

	void setInputDim(unsigned int InputDim);
	unsigned int getInputDim();
	void setU_k(Eigen::VectorXd U_k);
	Eigen::VectorXd getU_k();

	void setA_k(Eigen::MatrixXd A_k);
	Eigen::MatrixXd getA_k();
	void setB_k(Eigen::MatrixXd B_k);
	Eigen::MatrixXd getB_k();
	void setC_k(Eigen::MatrixXd C_k);
	Eigen::MatrixXd getC_k();	
	
};



#endif
