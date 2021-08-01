import csv

input_file = csv.reader(open('chronograf-data.csv', 'r'))
output_file = open('processed_chronograf_data.txt', 'w')

for row in input_file:
    if row[2] == 'tempf':
        if int(row[0][8:10]) >= 5 and int(row[0][8:10]) <= 10:
            print(row)
            output_file.write(row[1] + '\n')
