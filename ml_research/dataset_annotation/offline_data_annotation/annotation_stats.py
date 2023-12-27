import pprint as pp
if __name__ == "__main__":
    train_stats = {}
    test_stats = {}

    with open("../data/train_adjusted.csv") as f:
        for line in f: 
            _,_,_,macro_group = line.strip().split(",")
            if macro_group not in train_stats:
                train_stats[macro_group] = 1
            else:
                train_stats[macro_group] += 1

    with open("../data/test_kaggletest_adjusted.csv") as f:
        for line in f: 
            _,_,_,macro_group = line.strip().split(",")
            if macro_group not in test_stats:
                test_stats[macro_group] = 1
            else:
                test_stats[macro_group] += 1

    print("Training Stats")
    pp.pprint(train_stats)
    print("Testing Stats")
    pp.pprint(test_stats)
    

