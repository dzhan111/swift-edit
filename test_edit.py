from inference.edit_image import SwiftEditPipeline
import os

print("Loading SwiftEdit pipeline...")

weights_root = os.path.join(os.path.dirname(
    __file__), "swiftedit_weights")
inversion_path = os.path.join(weights_root, "inverse_ckpt-120k")
sbv2_path = os.path.join(weights_root, "sbv2_0.5")
ip_adapter_path = os.path.join(
    weights_root, "ip_adapter_ckpt-90k", "ip_adapter.bin")

pipeline = SwiftEditPipeline(
    inversion_model_path=inversion_path,
    sbv2_model_path=sbv2_path,
    ip_adapter_path=ip_adapter_path,
    device="cpu",
    dtype="fp32",
)

print("Editing image...")
edited_image = pipeline.edit(
    source_image="test_images/woman_face.jpg",
    source_prompt="woman",
    edit_prompt="taylor swift",
)


edited_image.save("test_images/taylor_swift.png")
print("Edited image saved to taylor_swift.png")
