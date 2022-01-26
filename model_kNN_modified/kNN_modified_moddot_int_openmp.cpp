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

const int k = 100;
const double NO_RATING_PLACEHOLDER = 0.5;

auto mask(const SparseMatrix<int> &lhs, const SparseMatrix<int> &rhs)
{
    struct DoubleAndReserveLhs
    {
        double operator()(int lhs, int rhs) const
        {
            if (lhs == 0 || rhs == 0) return 0;
            if (lhs == 2 && rhs == 2) return 2;
            return 1;
        };
    };
    return CwiseBinaryOp<DoubleAndReserveLhs, const SparseMatrix<int>, const SparseMatrix<int>>(lhs.derived(), rhs.derived());
}

int main()
{
    std::ifstream data_input("../DoubanMusic.txt");
    std::ofstream out_res("./result.txt");

    std::list<Triplet<int>> input_triplets;
    std::vector<std::set<int>> user_interact(USER_NUMBER);
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
            user_interact[user_id].insert(item_id);
            // if (item_rating < 0) 
            //     item_rating = NO_RATING_PLACEHOLDER;
            // else 
            //     item_rating = RATING_PLACEHOLDER;
            input_triplets.emplace_back(user_id, item_id, item_rating);

            // VERBOSE(item_id << "," << item_rating << ',' << data_input.peek());
            // if (item_rating == 2135) return 0;
        }
    }

    VERBOSE("ratings");

    SparseMatrix<int> ratings(USER_NUMBER, ITEM_NUMBER);
    ratings.setFromTriplets(input_triplets.begin(), input_triplets.end());

    // RowVector<double, Dynamic> item_average(ITEM_NUMBER);
    // for (int j = 0; j < ITEM_NUMBER; ++j)
    // {
    //     item_average(j) = ratings.col(j).sum() / ratings.col(j).nonZeros();
    //     if (j < 10)
    //         std::cout << item_average(j) << std::endl;
    // }

    // for (int j = 0; j < ITEM_NUMBER; ++j)
    // {
    //     for (SparseMatrix<double>::InnerIterator it(ratings, j); it; ++it)
    //     {
    //         if (it.value() == 0)
    //         {
    //             it.valueRef() = item_average(j);
    //             // VERBOSE(it.valueRef());
    //         }
    //     }
    // }

    VERBOSE("simi");

    std::vector<std::vector<int>> similarity(ITEM_NUMBER);
    for (auto & row : similarity) row.resize(ITEM_NUMBER);

    #pragma omp parallel for
    for (int j = 0; j < ITEM_NUMBER; ++j)
    {
        for (int l = 0; l < j; ++l)
        {
            // if (l == j) continue;
            int sim = mask(ratings.col(j), ratings.col(l)).sum();
            // if (sim == 1 || sim == -1) continue;
            similarity[j][l] = similarity[l][j] = sim;
            // VERBOSE(sim);
        }
        // VERBOSE(id << ' ' << j);
    }

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
    //         double norm1 = mask(ratings.col(j), ratings.col(l)).norm();
    //         double norm2 = mask(ratings.col(l), ratings.col(j)).norm();
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


    auto userTask = [&user_interact, &similarity, &ratings, &output_temp_files](int id, int begin, int end)
    {
        for (int i = begin; i < end; ++i)
        {
            std::vector<std::pair<double, int>> user_recommendation;
            user_recommendation.reserve(ITEM_NUMBER);
            for (int j = 0; j < ITEM_NUMBER; ++j)
            {
                if (user_interact[i].find(j) != user_interact[i].end())
                    continue;

                std::vector<std::pair<double, int>> most_similar_items;
                for (int l : user_interact[i])
                {
                    double sim = similarity[l][j];
                    most_similar_items.emplace_back(sim, l);
                }
                // VERBOSE("sim > 0 items: " << most_similar_items.size());
                // std::sort(most_similar_items.rbegin(), most_similar_items.rend());
                // if (most_similar_items.size() > k)
                //     most_similar_items.resize(k);
                
                double predict_rating = 0;
                for (auto [sim, l] : most_similar_items)
                {
                    predict_rating += sim;
                }
                // if (sim_sum == 0) predict_rating = item_average[j];
                // else predict_rating = predict_rating / sim_sum + item_average[j];
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
            out_res << tmp_string << "\n";
        }
    }
    

    return 0;
}
