input_file = open("output.txt", "r")
output_elapsed_file = open("batch_elapsed_output.txt", "w")
output_rate_file = open("batch_rate_output.txt", "w")

for line in input_file.readlines():
    if "    Elapsed Time: " in line:
        output_elapsed_file.write(line[18:])
    elif "    Throughput: " in line:
        output_rate_file.write(line[16:])

input_file.close()
output_elapsed_file.close()
output_rate_file.close()