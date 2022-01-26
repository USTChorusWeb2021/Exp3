#include <iostream>
#include "Eigen/Dense"
#include <fstream>
#include <string>
#include <cmath>

const int USER_NUMBER = 23599;

int main()
{

    std::ifstream test_input("./result.txt");
    std::ifstream valid_dataset("../valid_dataset.txt");
    std::ofstream test_res("./test_score.txt");

    int user_id, predict_count, predict_place;
    int predict_item;
    int valid_item;
    double NDCG[2], HR[2];

    for (int i = 0; i < USER_NUMBER; ++i)
    {
        int user_id;
        test_input >> user_id;
        // std::cout << user_id << "\t";
        valid_dataset >> valid_item;
        predict_place = 0;
        while (true)
        {
            test_input >> predict_item;
            // std::cout << predict_item << " ";
            predict_place++;
            if (predict_item == valid_item)
            {
                if (predict_place <= 20)
                {
                    HR[0]++;
                    NDCG[0] += 1 / (log2(predict_place + 1));
                }
                if (predict_place <= 100)
                {
                    HR[1]++;
                    NDCG[1] += 1 / (log2(predict_place + 1));
                }
            }
            if (test_input.get() == '\n') break;
        }
        // std::cout << std::endl;
        // std::cout.flush();

    }
    HR[0] /= USER_NUMBER;
    HR[1] /= USER_NUMBER;
    NDCG[0] /= USER_NUMBER;
    NDCG[1] /= USER_NUMBER;
    std::cout << "HR@20\tHR@100\tNDCG@20\tNDCG@100" << std::endl;
    std::cout << HR[0] << "\t" << HR[1] << "\t" << NDCG[0] << "\t" << NDCG[1] << std::endl;
    test_res << "HR@20\tHR@100\tNDCG@20\tNDCG@100" << std::endl;
    test_res << HR[0] << "\t" << HR[1] << "\t" << NDCG[0] << "\t" << NDCG[1] << std::endl;


    return 0;
}