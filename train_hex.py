# train_hex.py
import numpy as np
from hex_gtm.selfplay import make_dataset
from hex_gtm.model import HexGTM
from hex_gtm.hex_rules import X, O

if __name__ == "__main__":
    N = 7
    data = make_dataset(n=N, games=3000, seed=42)

    # train/test split
    rng = np.random.default_rng(0)
    rng.shuffle(data)
    cut = int(0.8 * len(data))
    train, test = data[:cut], data[cut:]

    model = HexGTM(n=N, clauses=400, T=80, s=5.0, depth=2, hv_size=1024, hv_bits=4)
    model.fit(train, epochs=25)

    # quick test accuracy
    correct = 0
    for b, to_move, y in test:
        p = model.predict_proba(b, to_move)
        pred = 1 if p >= 0.5 else 0
        correct += int(pred == y)
    print(f"Test acc: {correct/len(test):.3f}")

    # save via pickle (GraphTM objects are Python-level)
    import pickle
    with open("hex_gtm_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Saved model to hex_gtm_model.pkl")
