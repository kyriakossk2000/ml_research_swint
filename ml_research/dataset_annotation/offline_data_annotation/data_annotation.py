from PIL import Image

def load_groups_pictures(file_path):
    images = []
    groups = []
    with open(file_path) as f:
        next(f)
        for line in f:
            line = line.strip().split(",")
            image = line[0]
            group = line[2]
            if group not in groups:
                groups.append(group)
                images.append(image)
    
    return (images, groups)



if __name__ == "__main__":
    # labels_for_groups_so_far = []
    # images, groups = load_groups_pictures("../data/train.csv")
    # with open("results.csv", "w+") as w:
    #     for image_index, image_path in enumerate(images):
    #         print(f"Processing group {groups[image_index]}")
    #         s = ""
    #         for index, custom_label in enumerate(labels_for_groups_so_far):
    #             s += f"- [{index}] {custom_label} \n"
    #         print(f"Current labels available {len(labels_for_groups_so_far)}")
    #         print(s)
    #         test_img = Image.open(f"../data/train/{image_path}")
    #         test_img.show()
    #         inp = input("Specify class:")
    #         if inp.isnumeric():
    #             w.write(f"{groups[image_index]},{labels_for_groups_so_far[int(inp)]}")
    #             w.write("\n")
    #         else:
    #             labels_for_groups_so_far.append(inp)
    #             w.write(f"{groups[image_index]},{labels_for_groups_so_far[-1]}")
    #             w.write("\n")

    #         test_img.close()
    new_labels = {}
    with open("results.csv") as f:
        for line in f:
            group, label = line.strip().split(",")
            new_labels[group] = label

    class_labels_map = {}
    with open("../data/train.csv") as f:
        with open("../data/train_adjusted.csv", "w")  as w:
            next(f)
            for line in f:
                line = line.strip()
                image, class_label, group = line.split(",")
                w.write(f"{image},{class_label},{group},{new_labels[group]}\n")
                if class_label not in class_labels_map:
                    class_labels_map[class_label] = group
    

    with open("../data/test_kaggletest.csv") as f:
        with open("../data/test_kaggletest_adjusted.csv", "w")  as w:
            next(f)
            for line in f:
                line = line.strip()
                image, class_label, visibility = line.split(",")
                w.write(f"{image},{class_label},{visibility},{new_labels[class_labels_map[class_label]]}\n")
            
        



