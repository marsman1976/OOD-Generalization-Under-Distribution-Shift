import torch

from model import BaselineMLP
from model_phase3 import build_phase3_model, count_parameters


def main():

    print("=" * 70)
    print("PHASE 3 — LEGACY M0 EQUIVALENCE AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Build both models
    # ---------------------------------------------------------

    torch.manual_seed(12345)

    old_m0 = BaselineMLP()
    new_m0 = build_phase3_model("M0")

    print("\nOLD PHASE 1/2 M0")
    print("-" * 70)
    print(old_m0)

    print("\nNEW PHASE 3 M0")
    print("-" * 70)
    print(new_m0)

    # ---------------------------------------------------------
    # 2. Compare parameter counts
    # ---------------------------------------------------------

    old_params = count_parameters(old_m0)
    new_params = count_parameters(new_m0)

    print("\nPARAMETER COUNTS")
    print("-" * 70)
    print(f"Old M0: {old_params}")
    print(f"New M0: {new_params}")

    assert old_params == 833, (
        f"Old M0 should have 833 parameters, got {old_params}"
    )

    assert new_params == 833, (
        f"New M0 should have 833 parameters, got {new_params}"
    )

    assert old_params == new_params

    print("Parameter count check: PASS")

    # ---------------------------------------------------------
    # 3. Compare state-dict structure
    # ---------------------------------------------------------

    old_state = old_m0.state_dict()
    new_state = new_m0.state_dict()

    print("\nSTATE-DICT STRUCTURE")
    print("-" * 70)

    old_keys = list(old_state.keys())
    new_keys = list(new_state.keys())

    print("Old keys:")
    for key in old_keys:
        print(" ", key, tuple(old_state[key].shape))

    print("\nNew keys:")
    for key in new_keys:
        print(" ", key, tuple(new_state[key].shape))

    assert old_keys == new_keys, (
        "Old and new M0 state-dict keys differ."
    )

    for key in old_keys:

        assert old_state[key].shape == new_state[key].shape, (
            f"Shape mismatch for {key}: "
            f"{old_state[key].shape} vs "
            f"{new_state[key].shape}"
        )

    print("State-dict structure check: PASS")

    # ---------------------------------------------------------
    # 4. Copy EXACT old weights into new model
    # ---------------------------------------------------------

    new_m0.load_state_dict(old_state)

    print("\nWEIGHT TRANSFER")
    print("-" * 70)
    print("Old M0 weights copied into new Phase 3 M0.")

    for key in old_state:

        assert torch.equal(
            old_m0.state_dict()[key],
            new_m0.state_dict()[key]
        ), f"Weight mismatch after copying: {key}"

    print("Weight equality check: PASS")

    # ---------------------------------------------------------
    # 5. Functional equivalence test
    # ---------------------------------------------------------

    torch.manual_seed(999)

    x = torch.randn(100, 8)

    old_m0.eval()
    new_m0.eval()

    with torch.no_grad():

        y_old = old_m0(x)
        y_new = new_m0(x)

    print("\nFUNCTIONAL OUTPUT TEST")
    print("-" * 70)

    print("Input shape :", x.shape)
    print("Old output :", y_old.shape)
    print("New output :", y_new.shape)

    assert y_old.shape == (100, 1)
    assert y_new.shape == (100, 1)

    max_difference = torch.max(
        torch.abs(y_old - y_new)
    ).item()

    print(
        f"Maximum absolute output difference: "
        f"{max_difference:.12e}"
    )

    assert torch.allclose(
        y_old,
        y_new,
        rtol=1e-7,
        atol=1e-8
    ), (
        "Old M0 and new Phase 3 M0 do not "
        "produce equivalent outputs."
    )

    print("Functional equivalence check: PASS")

    # ---------------------------------------------------------
    # 6. Final audit
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LEGACY M0 EQUIVALENCE AUDIT: PASS")
    print("=" * 70)

    print(
        "\nConclusion:\n"
        "Phase 3 M0 is architecturally and functionally "
        "equivalent to the Phase 1/2 baseline M0."
    )


if __name__ == "__main__":
    main()