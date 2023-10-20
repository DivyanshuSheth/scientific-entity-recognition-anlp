from utility import read_conll, train_val_split

if __name__ == "__main__":
    list_of_paper_lines = read_conll("./AnnotatedData/export_42758_project-42758-at-2023-10-20-18-39-a9525692.conll")
    train_lines, val_lines = train_val_split(list_of_paper_lines)

    print(train_lines)