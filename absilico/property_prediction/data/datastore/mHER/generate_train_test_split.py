import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test_size", type=float, default=0.2)
args = parser.parse_args()

pos = pd.read_csv("mHER_H3_AgPos.csv")
neg = pd.read_csv("mHER_H3_AgNeg.csv")

pos["binding"] = True
neg["binding"] = False

pos_train, pos_test = train_test_split(
    pos, test_size=args.test_size, random_state=0
)
neg_train, neg_test = train_test_split(
    neg, test_size=args.test_size, random_state=0
)

train = pd.concat((pos_train, neg_train))
test = pd.concat((pos_test, neg_test))

test_porportion = int(args.test_size * 100)

train.to_csv(f"mHER_H3_train_{test_porportion}.csv")
test.to_csv(f"mHER_H3_test_{test_porportion}.csv")
