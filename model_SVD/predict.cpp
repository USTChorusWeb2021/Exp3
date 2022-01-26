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

using namespace Eigen;

#define VERBOSE(str) std::cout << str << '\n'

const int USER_NUMBER = 23599;
const int ITEM_NUMBER = 21602;
const int N_FACTOR = 30;
const int MAX_THREAD = 16;

int main()
{
    std::ifstream raw_input("../train.txt");
    std::ifstream p_input("./predict_matrix_p.txt");
    std::ifstream q_input("./predict_matrix_q.txt");
    std::ifstream bias_input("./item_bias.txt");
    std::ofstream r_predict_out("./rating_predict.txt");
    std::ofstream out_res("./result.txt");

    VERBOSE("input");

    std::vector<std::vector<int>> user_interact(USER_NUMBER);
    for (int i = 0; i < USER_NUMBER; ++i)
    {
        int user_id;
        raw_input >> user_id;
        while (raw_input.peek() != '\n' && raw_input.peek() != '\r')
        {
            int item_id;
            int item_rating;

            raw_input >> item_id;
            raw_input.get();
            raw_input >> item_rating;
            user_interact[user_id].push_back(item_id);
        }
        user_interact[user_id].push_back(ITEM_NUMBER); // Sentry
        std::sort(user_interact[user_id].begin(), user_interact[user_id].end());
    }
    raw_input.close();

    MatrixXd p(USER_NUMBER, N_FACTOR);
    for (int i = 0; i < USER_NUMBER; ++i)
        for (int k = 0; k < N_FACTOR; ++k)
            p_input >> p(i, k);

    p_input.close();

    MatrixXd q(N_FACTOR, ITEM_NUMBER);
    for (int k = 0; k < N_FACTOR; ++k)
        for (int j = 0; j < ITEM_NUMBER; ++j)
            q_input >> q(k, j);
    q_input.close();

    // MatrixXd b(1, ITEM_NUMBER);
    // for (int j = 0; j < N_FACTOR; ++j)
    //     bias_input >> q(1, j);
    // bias_input.close();

    VERBOSE("mul");

    MatrixXd r = p * q;

    for (int i = 0 ; i < 3; ++i)
    {
        for (int j = 0; j < ITEM_NUMBER; ++j)
            r_predict_out << r(i, j) << " ";
        r_predict_out << std::endl;
    }

    VERBOSE("user");

    std::vector<std::ofstream> output_temp_files(MAX_THREAD);
    for (int i = 0; i < MAX_THREAD; ++i)
    {
        std::stringstream ss;
        ss << "./temp" << i << ".txt";
        output_temp_files[i].open(ss.str());
    }

    auto userTask = [&user_interact, &r, &output_temp_files](int id, int begin, int end)
    {
        for (int i = begin; i < end; ++i)
        {
            auto it = user_interact[i].begin();
            std::vector<std::pair<double, int>> user_recommendation;
            user_recommendation.reserve(ITEM_NUMBER);
            for (int j = 0; j < ITEM_NUMBER; ++j)
            {
                while (*it < j) ++it;
                if (*it == j) continue; // Already interacted

                double rating_predict = r(i, j);
                user_recommendation.emplace_back(rating_predict, j);
                // if (*it == j) VERBOSE(rating_predict);
            }

            std::sort(user_recommendation.rbegin(), user_recommendation.rend());
            output_temp_files[id] << i << "\t";
            for (int l = 0; l < 100; ++l)
            {
                if (l != 99)
                    output_temp_files[id] << user_recommendation[l].second << ",";
                else
                    output_temp_files[id] << user_recommendation[l].second << std::endl;
            }
        }
    };

    std::vector<std::thread> userThreads;
    for (int t = 0; t < MAX_THREAD; ++t)
    {
        int begin = USER_NUMBER / MAX_THREAD * t;
        int end = USER_NUMBER / MAX_THREAD * (t + 1);
        if (t == MAX_THREAD - 1) end = USER_NUMBER;

        userThreads.emplace_back(userTask, t, begin, end);
    }

    for (auto & t : userThreads) t.join();
    
    for (int i = 0; i < MAX_THREAD; ++i)
    {
        output_temp_files[i].close();
    }

    for (int i = 0; i < MAX_THREAD; ++i)
    {
        std::stringstream ss;
        ss << "./temp" << i << ".txt";
        std::ifstream temp_read_file(ss.str());
        while (!temp_read_file.eof())
        {
            std::string tmp_string;
            std::getline(temp_read_file, tmp_string);
            temp_read_file >> std::ws;
            out_res << tmp_string << std::endl;
        }
    }
    out_res.close();
    
    return 0;
}
