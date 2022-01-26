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
const int MAX_THREAD = 16;
const int EPOCHS = 400;
const int N_FACTORS = 30;
const double REG_ALL = 0.05, LR_ALL = 0.01;

int main()
{

    std::ifstream data_input("../train.txt");
    std::ofstream out_res("./result.txt");

    std::vector<Triplet<int>> input_triplets;
    std::vector<std::set<int>> user_interact(USER_NUMBER);
    double global_mean;
    int count = 0;

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
            // user_interact[user_id].insert(item_id);
            // if (item_rating < 0) continue;
            // count++;
            // global_mean += item_rating;
            input_triplets.emplace_back(user_id, item_id, 1.0);

            // VERBOSE(item_id << "," << item_rating << ',' << data_input.peek());
        }
    }

    global_mean /= count;

    MatrixXd pu(USER_NUMBER, N_FACTORS);
    MatrixXd qi(ITEM_NUMBER, N_FACTORS);

    // random distribution
    std::default_random_engine e;
    std::uniform_real_distribution<double> uniform(0,1);
    for (int i = 0; i < USER_NUMBER; ++i)
    {
        for (int j = 0; j < N_FACTORS; ++j)
            pu(i, j) = uniform(e);
    }
    for (int i = 0; i < ITEM_NUMBER; ++i)
    {
        for (int j = 0; j < N_FACTORS; ++j)
            qi(i, j) = uniform(e);
    }

    // VectorXd bu(USER_NUMBER);
    // VectorXd bi(ITEM_NUMBER);

    for (int k = 0; k < EPOCHS; ++k)
    {

        double total_error = 0;
        for (int iti = 0; iti < input_triplets.size(); ++iti)
        {
            auto it = input_triplets[iti];
            int u, i;
            double r;
            u = it.row();
            i = it.col();
            r = it.value();
            double inner_product = qi.row(i).dot(pu.row(u));
            // for (int j = 0; j < N_FACTORS; ++j)
            //     inner_product += qi(i, j) * pu(u, j);
            // double err = r - (global_mean + bu[u] + bi[i] + inner_product);
            double err = r - inner_product;
            total_error += err * err;
            // bu(u) += LR_ALL * (err - REG_ALL * bu(u));
            // bi(i) += LR_ALL * (err - REG_ALL * bi(i));
            // if (uniform(e) > 0.2) continue;
            pu.row(u) += LR_ALL * (err * qi.row(i) - REG_ALL * pu.row(u));
            qi.row(i) += LR_ALL * (err * pu.row(u) - REG_ALL * qi.row(i));
            // for (int j = 0; j < N_FACTORS; ++j)
            // {
            //     pu(u, j) += LR_ALL * (err * qi(i, j) - REG_ALL * pu(u, j));
            //     qi(i, j) += LR_ALL * (err * pu(u, j) - REG_ALL * qi(i, j));
            // }
            // VERBOSE(pu);

        }
        std::cout << k << " " << total_error << std::endl;


    }

    std::ofstream p_output("./predict_matrix_p.txt");
    std::ofstream q_output("./predict_matrix_q.txt");

    for (int i = 0; i < USER_NUMBER; ++i)
    {
        for (int j = 0; j < N_FACTORS; ++j)
            p_output << pu(i, j) << " ";
    }
    // Transposed output
    for (int j = 0; j < N_FACTORS; ++j)
    {
        for (int i = 0; i < ITEM_NUMBER; ++i)
            q_output << qi(i, j) << " ";
    }


    return 0;
}