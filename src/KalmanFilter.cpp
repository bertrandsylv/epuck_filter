#include "KalmanFilter.h"


// constructeurs

KalmanFilter::KalmanFilter(){
}


KalmanFilter::KalmanFilter(unsigned int StateDim, unsigned int InputDim, unsigned int MeasurementDim){
	this->StateDim = StateDim;
	this->X_k.resize(StateDim);
	this->P_k.resize(StateDim, StateDim);

	this->Q_k.resize(StateDim, StateDim);

	this->MeasurementDim = MeasurementDim;
	this->Y_k.resize(MeasurementDim);
	this->R_k.resize(MeasurementDim, MeasurementDim);

	this->InputDim = InputDim;
	this->U_k.resize(InputDim);

	this->A_k.resize(StateDim, StateDim);
	this->B_k.resize(StateDim, InputDim);
	this->C_k.resize(MeasurementDim, StateDim);	
}


// destructeur
KalmanFilter::~KalmanFilter(){
}

// affichage
std::ostream& operator<<(std::ostream& os, const KalmanFilter& KF){
	os << "Sys: " << std::endl;
	os << "  Ak = [" << KF.A_k << "]" << std::endl;
	os << "  Bk = [" << KF.B_k << "]" << std::endl;
	os << "  Ck = [" << KF.C_k << "]" << std::endl;
	os << "State noise cov matrix: " << std::endl;
	os << "  Qk = [" << KF.Q_k << "]" << std::endl;
	os << "State and cov matrix: " << std::endl;
	os << "  Xk = [" << KF.X_k.transpose() << "]^t" << std::endl;
	os << "  Pk = [" << KF.P_k << "]" << std::endl;
	os << "Measurement and cov matrix:" <<  std::endl;
	os << "  Yk = [" << KF.Y_k.transpose() << "]^t" << std::endl;
	os << "  Rk = [" << KF.R_k << "]" << std::endl;
	os << "Input:" << std::endl;
	os << "  Uk = [" << KF.U_k.transpose() << "]^t" << std::endl;
	return os;
}


// initialisations rapides
void KalmanFilter::setSys(Eigen::MatrixXd A_k, Eigen::MatrixXd B_k, Eigen::MatrixXd C_k){
	this->A_k = A_k;
	this->B_k = B_k;
	this->C_k = C_k;
}

void KalmanFilter::setNoiseCovMatrices(Eigen::MatrixXd Q_k, Eigen::MatrixXd R_k){
	this->Q_k = Q_k;
	this->R_k = R_k;
}

void KalmanFilter::initFilter(Eigen::VectorXd X_0, Eigen::MatrixXd P_0){
	this->X_k = X_0;
	this->P_k = P_0;
}


// prédiction
void KalmanFilter::predictOneStep(){

	// prédiction état
	if (this->InputDim>0){
		this->X_k = this->A_k * this->X_k + this->B_k * this->U_k;
	}
	else{
		this->X_k = this->A_k * this->X_k;
	}

	// prédiction matrice de covariance
	this->P_k = this->A_k * this->P_k * this->A_k.transpose() + this->Q_k;
}


// mise à jour
void KalmanFilter::updateOneStep(){
	// innovation et covariance
	Eigen::VectorXd	Ytilde_k = this->Y_k - this->C_k * this->X_k;
	Eigen::MatrixXd S_k = this->C_k * this->P_k * this->C_k.transpose() + this->R_k;

	// gain de Kalman
	Eigen::MatrixXd K_k = this->P_k * this->C_k.transpose() * S_k.inverse();

	// maj état
	this->X_k = this->X_k + K_k * Ytilde_k;
	
	// maj matrice de covariance
	this->P_k = (Eigen::MatrixXd::Identity(this->StateDim, this->StateDim) - K_k * this->C_k) * this->P_k;
}


// renvoi de l'innovation
Eigen::VectorXd KalmanFilter::getInnov(){
	Eigen::VectorXd	Ytilde_k = this->Y_k - this->C_k * this->X_k;
	return Ytilde_k;	
}


// renvoi de la matrice de covariance de l'innovation
Eigen::MatrixXd KalmanFilter::getInnovCovMatrix(){
	Eigen::MatrixXd S_k = this->C_k * this->P_k * this->C_k.transpose() + this->R_k;
	return S_k;
}


// accesseurs
void KalmanFilter::setStateDim(unsigned int StateDim){
	this->StateDim = StateDim;
	this->P_k.resize(StateDim, StateDim);
	this->Q_k.resize(StateDim, StateDim);
	this->A_k.resize(StateDim, StateDim);
	this->B_k.resize(StateDim, this->InputDim);
}

unsigned int KalmanFilter::getStateDim(){
	return this->StateDim;
}

void KalmanFilter::setX_k(Eigen::VectorXd X_k){
	// gérer problème de dimension
	this->X_k = X_k;
}

Eigen::VectorXd KalmanFilter::getX_k(){
	return this->X_k;
}

void KalmanFilter::setP_k(Eigen::MatrixXd P_k){
	// gérer problème de dimension
	this->P_k = P_k;
}

Eigen::MatrixXd KalmanFilter::getP_k(){
	return this->P_k;
}

void KalmanFilter::setQ_k(Eigen::MatrixXd Q_k){
	// gérer problème de dimension
	this->Q_k = Q_k;
}

Eigen::MatrixXd KalmanFilter::getQ_k(){
	return this->Q_k;
}

void KalmanFilter::setMeasurementDim(unsigned int MeasurementDim){
	this->MeasurementDim = MeasurementDim;
	this->Y_k.resize(MeasurementDim);
	this->R_k.resize(MeasurementDim, MeasurementDim);
	this->C_k.resize(MeasurementDim, this->StateDim);
}

unsigned int KalmanFilter::getMeasurementDim(){
	return this->MeasurementDim;
}

void KalmanFilter::setY_k(Eigen::VectorXd Y_k){
	this->Y_k.resize(Y_k.rows());
	this->Y_k = Y_k;
}

Eigen::VectorXd KalmanFilter::getY_k(){
	return this->Y_k;
}


void KalmanFilter::setR_k(Eigen::MatrixXd R_k){
	this->R_k.resize(R_k.rows(),R_k.cols());
	this->R_k = R_k;
}

Eigen::MatrixXd KalmanFilter::getR_k(){
	return this->R_k;
}


void KalmanFilter::setInputDim(unsigned int InputDim){
	this->InputDim = InputDim;
	this->U_k.resize(InputDim);
	this->B_k.resize(this->StateDim, InputDim);
}

unsigned int KalmanFilter::getInputDim(){
	return this->InputDim;
}

void KalmanFilter::setU_k(Eigen::VectorXd U_k){
	this->U_k.resize(U_k.rows(),U_k.cols());
	this->U_k = U_k;
}

Eigen::VectorXd KalmanFilter::getU_k(){
	return this->U_k;
}

void KalmanFilter::setA_k(Eigen::MatrixXd A_k){
	this->A_k.resize(A_k.rows(),A_k.cols());
	this->A_k = A_k;
}

Eigen::MatrixXd KalmanFilter::getA_k(){
	return this->A_k;
}


void KalmanFilter::setB_k(Eigen::MatrixXd B_k){
	this->B_k.resize(B_k.rows(),B_k.cols());
	this->B_k = B_k;
}

Eigen::MatrixXd KalmanFilter::getB_k(){
	return this->B_k;
}

void KalmanFilter::setC_k(Eigen::MatrixXd C_k){
	this->C_k.resize(C_k.rows(),C_k.cols());
	this->C_k = C_k;
}

Eigen::MatrixXd KalmanFilter::getC_k(){
	return this->C_k;
}








