Eigen::VectorXd eigenGauss(Eigen::MatrixXd matrix_A, Eigen::VectorXd vector_b) {
    return matrix_A.householderQr().solve(vector_b);
}
