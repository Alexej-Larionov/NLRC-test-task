from common import load_model, load_data, ASSETS

print("loading model:")
tok, model = load_model()
model.save_pretrained(ASSETS / "model")
tok.save_pretrained(ASSETS / "model")

print("loading dataset:")
ds = load_data()
try:
    ds.save_to_disk(ASSETS / "openwebtext_100k")
except Exception:
    pass

print("done")
