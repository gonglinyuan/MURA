import os
import sys

import numpy as np
import pandas
import yaml

import predict


def main():
    keys, results = [], []
    for path_config in ["config01.yaml", "config02.yaml", "config03.yaml", "config04.yaml", "config05.yaml"]:
        with open(os.path.join("configs", path_config), "r") as f:
            config = yaml.load(f)
        keys, result = predict.predict(sys.argv[1], config)
        results.append(result)
    results = np.stack(results, axis=1)
    # score = np.mean(results, axis=1)
    # label = np.array(score >= 0.0, dtype=np.int32)
    pandas.DataFrame(results, index=keys).to_csv(sys.argv[2], header=False)


# --------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
