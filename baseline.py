from utility import read_conll

if __name__ == "__main__":
    data_lines = read_conll("./AnnotatedData/export_42758_project-42758-at-2023-10-20-18-39-a9525692.conll")
    print(data_lines)