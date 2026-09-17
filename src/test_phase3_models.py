import torch

from model_phase3 import (
    build_phase3_model,
    count_parameters,
)


EXPECTED = {
    "M1": {
        "hidden_dims": [16, 8],
        "parameters": 289,
    },
    "M0": {
        "hidden_dims": [32, 16],
        "parameters": 833,
    },
    
    "M2": {
    "hidden_dims": [64, 32, 16],
    "parameters": 3201,
},
}


def test_model(model_id):

    print("=" * 70)
    print(f"Testing {model_id}")
    print("=" * 70)

    model = build_phase3_model(model_id)

    print(model)

    # ----------------------------------
    # Architecture
    # ----------------------------------

    assert model.model_id == model_id

    assert (
        model.hidden_dims
        == EXPECTED[model_id]["hidden_dims"]
    )

    # ----------------------------------
    # Parameter count
    # ----------------------------------

    n_params = count_parameters(model)

    print(f"\nTrainable parameters: {n_params}")

    assert (
        n_params
        == EXPECTED[model_id]["parameters"]
    ), (
        f"{model_id}: expected "
        f"{EXPECTED[model_id]['parameters']} "
        f"parameters but got {n_params}"
    )

    # ----------------------------------
    # Input/output shape
    # ----------------------------------

    x = torch.randn(32, 8)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    assert x.shape == (32, 8)
    assert y.shape == (32, 1)

    # ----------------------------------
    # Numerical sanity
    # ----------------------------------

    assert torch.isfinite(y).all()

    print(f"\n{model_id}: PASS\n")


def main():

    print("\n")
    print("=" * 70)
    print("PHASE 3 MODEL ARCHITECTURE AUDIT")
    print("=" * 70)

    for model_id in ["M1", "M0", "M2"]:
        test_model(model_id)

    print("=" * 70)
    print("PHASE 3 MODEL ARCHITECTURE AUDIT: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()