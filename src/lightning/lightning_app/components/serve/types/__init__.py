from lightning.lightning_app.components.serve.types.image import Image

_SERIALIZER = {"image": Image.serialize}
_DESERIALIZER = {"image": Image.deserialize}
