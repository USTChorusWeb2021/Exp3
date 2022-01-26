#include <iostream>
#include "../Eigen/Dense"
#include "../Eigen/Sparse"
#include <vector>
#include <list>
#include <fstream>
#include <string>
#include <array>
#include <algorithm>
#include <set>
#include <thread>
#include <random>
#include <cmath>

using namespace Eigen;

#define VERBOSE(str) std::cout << str << '\n'

const int USER_NUMBER = 23599;
const int ITEM_NUMBER = 21602;
const int EPOCHS = 30;
const int N_FACTORS = 100;

int main()
{
    initParallel();

    std::ifstream data_input("../DoubanMusic.txt");
    std::ofstream out_res("./result.txt");

    std::vector<Triplet<int>> input_triplets;

    for (int i = 0; i < USER_NUMBER; ++i)
    {
        int user_id;
        data_input >> user_id;
        while (data_input.peek() != '\n' && data_input.peek() != '\r')
        {
            int item_id;
            int item_rating;

            data_input >> item_id;
            data_input.get();
            data_input >> item_rating;
            input_triplets.emplace_back(user_id, item_id, 1.0);
        }
    }
    
    SparseMatrix<double> r(USER_NUMBER, ITEM_NUMBER);
    r.setFromTriplets(input_triplets.begin(), input_triplets.end());

    MatrixXd p(USER_NUMBER, N_FACTORS);
    MatrixXd q(N_FACTORS, ITEM_NUMBER);

    std::default_random_engine e;
    e.seed(127429);
    std::uniform_real_distribution<double> uniform(0,1);
    for (int i = 0; i < USER_NUMBER; ++i)
        for (int j = 0; j < N_FACTORS; ++j)
            p(i, j) = uniform(e);
    for (int i = 0; i < N_FACTORS; ++i)
        for (int j = 0; j < ITEM_NUMBER; ++j)
            q(i, j) = uniform(e);

    for (int k = 0; k < EPOCHS; ++k)
    {
        VERBOSE("Epoch " << k);

        MatrixXd rhat = p * q;
        double loss = (rhat - r).norm();
        VERBOSE("    loss: " << loss);

        // Update Pi's
        // #pragma omp parallel for
        // for (int i = 0; i < USER_NUMBER; ++i)
        //     p.row(i) = p.row(i).cwiseProduct((r.row(i) * q.transpose()).cwiseQuotient(rhat.row(i) * q.transpose()));
        p = p.cwiseProduct(r * q.transpose()).cwiseQuotient(rhat * q.transpose());

        rhat = p * q;
        loss = (rhat - r).norm();
        VERBOSE("    loss: " << loss);

        // Update Qj's
        // #pragma omp parallel for
        // for (int j = 0; j < ITEM_NUMBER; ++j)
        //     q.col(j) = q.col(j).cwiseProduct((p.transpose() * r.col(j)).cwiseQuotient(p.transpose() * rhat.col(j)));
        q = q.cwiseProduct(p.transpose() * r).cwiseQuotient(p.transpose() * rhat);
    }

    std::ofstream p_output("./predict_matrix_p.txt");
    std::ofstream q_output("./predict_matrix_q.txt");

    for (int i = 0; i < USER_NUMBER; ++i)
    {
        for (int j = 0; j < N_FACTORS; ++j)
            p_output << p(i, j) << " ";
    }
    for (int i = 0; i < N_FACTORS; ++i)
    {
        for (int j = 0; j < ITEM_NUMBER; ++j)
            q_output << q(i, j) << " ";
    }


    return 0;
}