import csv
csv_file = "TEST_RAP_pandas_frame_data_format_1.csv"
txt_file = "TEST_RAP_pandas_frame_data_format_1.txt"
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()