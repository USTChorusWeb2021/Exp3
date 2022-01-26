#include <iostream>
#include <fstream>

const int USER_NUMBER = 23599;
const int ITEM_NUMBER = 21602;

int main()
{

    std::ifstream data_input("../train.txt");
    std::ofstream reader_out("./surprise_read_data.txt");

    for (int i = 0; i < USER_NUMBER; ++i)
    {
        int user_id;
        data_input >> user_id;
        while (data_input.peek() != '\n')
        {
            int item_id;
            double item_rating;

            data_input >> item_id;
            data_input.get();
            data_input >> item_rating;
            reader_out << user_id << ";" << item_id << ";" << item_rating << std::endl;
        }
    }

    return 0;
}