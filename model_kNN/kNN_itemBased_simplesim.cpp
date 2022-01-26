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
const int MAX_THREAD = 16;
// const int USER_NUMBER = 13;
// const int ITEM_NUMBER = 7;
// const int MAX_THREAD = 1;

const int k = 40;

auto filterLhs(const SparseMatrix<double> &lhs, const SparseMatrix<double> &rhs)
{
    struct DoubleAndReserveLhs
    {
        double operator()(double lhs, double rhs) const
        {
            if (lhs && rhs) return lhs;
            return 0.0;
        };
    };
    return CwiseBinaryOp<DoubleAndReserveLhs, const SparseMatrix<double>, const SparseMatrix<double>>(lhs.derived(), rhs.derived());
}

int main()
{

    std::ifstream data_input("../train.txt");
    std::ofstream out_res("./result.txt");

    std::list<Triplet<double>> input_triplets;
    std::vector<std::set<int>> user_interact(USER_NUMBER);
    for (int i = 0; i < USER_NUMBER; ++i)
    {
        int user_id;
        data_input >> user_id;
        while (data_input.peek() != '\n')
        {
            int item_id, item_rating;

            data_input >> item_id;
            data_input.get();
            data_input >> item_rating;
            user_interact[user_id].insert(item_id);
            if (item_rating < 0) input_triplets.emplace_back(user_id, item_id, 0.5);
            else input_triplets.emplace_back(user_id, item_id, double(item_rating));
        }
    }

    VERBOSE("ratings");

    SparseMatrix<double> ratings(USER_NUMBER, ITEM_NUMBER);
    ratings.setFromTriplets(input_triplets.begin(), input_triplets.end());

    RowVector<double, Dynamic> item_average(ITEM_NUMBER);
    for (int j = 0; j < ITEM_NUMBER; ++j)
    {
        item_average(j) = ratings.col(j).sum() / ratings.col(j).nonZeros();
        // if (j < 10)
        //     std::cout << item_average(j) << std::endl;
    }

    for (int j = 0; j < ITEM_NUMBER; ++j)
    {
        for (SparseMatrix<double>::InnerIterator it(ratings, j); it; ++it)
        {
            it.valueRef() -= item_average(j);
        }
    }

    VERBOSE("simi");

    std::vector<std::vector<int>> similarity(ITEM_NUMBER);
    for (auto & row : similarity) row.resize(ITEM_NUMBER);

    auto simTask = [&ratings, &similarity](int id, int begin, int end)
    {
        for (int j = begin; j < end; ++j)
        {
            for (int l = 0; l < j; ++l)
            {
                // if (l == j) continue;
                SparseMatrix<double> intersect = filterLhs(ratings.col(j), ratings.col(l));
                // double norm2 = filterLhs(ratings.col(l), ratings.col(j)).norm();
                // if (norm1 == 0.0) continue;
                // double sim = ratings.col(j).dot(ratings.col(l)) / (norm1 * norm2);
                // if (sim == 1 || sim == -1) continue;
                similarity[j][l] = similarity[l][j] = intersect.nonZeros();
                // VERBOSE(intersect.nonZeros());
            }
            VERBOSE(id << ' ' << j);
        }
    };



    std::vector<std::thread> simThreads;
    for (int t = 0; t < MAX_THREAD; ++t)
    {
        int begin = ITEM_NUMBER / MAX_THREAD * t;
        int end = ITEM_NUMBER / MAX_THREAD * (t + 1);
        if (t == MAX_THREAD - 1) end = ITEM_NUMBER;

        simThreads.emplace_back(simTask, t, begin, end);
    }

    for (auto & t : simThreads) t.join();

    // for (int i = 0; i < USER_NUMBER; ++i)
    // {
    //     for (int j = 0; j < ITEM_NUMBER; ++j)
    //         std::cout << similarity[i][j] << "\t";
    //     std::cout << std::endl;
    // }

    // for (int j = 0; j < ITEM_NUMBER; ++j)
    // {
    //     for (int l = 0; l < j; ++l)
    //     {
    //         // if (l == j) continue;
    //         double norm1 = filterLhs(ratings.col(j), ratings.col(l)).norm();
    //         double norm2 = filterLhs(ratings.col(l), ratings.col(j)).norm();
    //         double sim = ratings.col(j).dot(ratings.col(l)) / (norm1 * norm2);
    //         similarity[j][l] = similarity[l][j] = sim;
    //     }
    //     VERBOSE(j);
    // }

    VERBOSE("user");

    std::vector<std::ofstream> output_temp_files(MAX_THREAD);
    for (int i = 0; i < MAX_THREAD; ++i)
    {
        std::stringstream ss;
        ss << "./temp" << i << ".txt";
        output_temp_files[i].open(ss.str());
    }


    auto userTask = [&user_interact, &similarity, &ratings, &item_average, &output_temp_files](int id, int begin, int end)
    {
        for (int i = begin; i < end; ++i)
        {
            std::vector<std::pair<double, int>> user_recommendation;
            user_recommendation.reserve(ITEM_NUMBER);
            for (int j = 0; j < ITEM_NUMBER; ++j)
            {
                if (user_interact[i].find(j) != user_interact[i].end())
                    continue;

                std::vector<std::pair<int, int>> most_similar_items;
                for (int l : user_interact[i])
                {
                    int sim = similarity[l][j];
                    // if (sim <= 1E-6) continue;
                    most_similar_items.emplace_back(sim, l);
                }
                // VERBOSE("sim > 0 items: " << most_similar_items.size());
                std::sort(most_similar_items.rbegin(), most_similar_items.rend());
                if (most_similar_items.size() > k)
                    most_similar_items.resize(k);
                
                double predict_rating = 0;
                int sim_sum = 0;
                for (auto [sim, l] : most_similar_items)
                {
                    sim_sum += sim;
                    predict_rating += sim * ratings.coeff(i, l);
                }
                if (sim_sum == 0) predict_rating = item_average[j];
                else predict_rating = predict_rating / sim_sum + item_average[j];
                // VERBOSE(predict_rating);
                user_recommendation.emplace_back(predict_rating, j);
            }
            std::sort(user_recommendation.rbegin(), user_recommendation.rend());
            output_temp_files[id] << i << "\t";
            for (int r = 0; r < 100; ++r)
            {
                if (r != 99)
                    output_temp_files[id] << user_recommendation[r].second << ",";
                else
                    output_temp_files[id] << user_recommendation[r].second << std::endl;
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
            out_res << tmp_string << '\n';
        }
    }
    

    return 0;
}