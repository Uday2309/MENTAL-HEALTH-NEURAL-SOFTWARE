import csv


def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader) 
        data = list(reader)     
    return headers, data


def find_s_algorithm(data):
    
    hypothesis = ['0'] * (len(data[0]) - 1)

    for row in data:
        if row[-1].lower() == 'yes':  
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':
                    hypothesis[i] = row[i]
                elif hypothesis[i] != row[i]:
                    hypothesis[i] = '?'
    return hypothesis


if __name__ == "__main__":
   
    headers, training_data = load_data("training_data.csv")
    hypothesis = find_s_algorithm(training_data)

    print("Attributes:", headers[:-1])
    print("Final Hypothesis:", hypothesis)
