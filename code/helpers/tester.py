import os

def count_items_in_folder(folder_path):
    num_items = len(os.listdir(folder_path))
    return num_items

def fuck(val_path, test_path, train_path):

    num_items2 = count_items_in_folder(val_path)
    num_items3 = count_items_in_folder(test_path)
    num_items1 = count_items_in_folder(train_path)

    if 3883 == num_items2+num_items1+num_items3:
        print("success!")
        return True
    else:
        print(num_items2+num_items1+num_items3)
        print("FUCK!")
        return False