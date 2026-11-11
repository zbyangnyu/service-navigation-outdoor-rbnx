"""SegFormer adapter: contract, size, threshold — no weight download needed."""
from pathlib import Path
import numpy as np
import pytest
from outdoor_nav.segmentation import predict_mask, validate_segmentation

torch = pytest.importorskip('torch')
pytest.importorskip('cv2')
from outdoor_nav import segformer_backend as sb  # noqa: E402


class Fake(torch.nn.Module):
    """Logits at 1/4 resolution; class = left/right half, sharpness by column."""
    def __init__(self, n=19):
        super().__init__()
        self.n, self.seen = n, []
        self.config = type('C', (), {'num_labels': n})()

    def forward(self, pixel_values):
        self.seen.append(tuple(pixel_values.shape))
        b, _, h, w = pixel_values.shape
        y = torch.zeros(b, self.n, h // 4, w // 4)
        y[:, 0, :, : w // 8] = 10.   # road on the left
        y[:, 2, :, w // 8:] = 10.    # building on the right
        return type('O', (), {'logits': y})()


def test_contract_shape_and_padding():
    m = Fake()
    b = sb.SegformerRoad(m, 'cpu')
    rgb = np.random.randint(0, 255, (100, 150, 3), np.uint8)
    mask, conf = b.predict(rgb)
    assert m.seen[0][2:] == (128, 160)  # padded to multiple of 32
    assert mask.shape == conf.shape == (100, 150) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 1}
    assert mask[:, :20].all() and not mask[:, -20:].any()
    assert (conf >= 0).all() and (conf <= 1).all()
    out = predict_mask(b, rgb, .7)
    assert set(np.unique(out)) <= {0, 1, 255}


def test_input_scale_and_threshold():
    m = Fake()
    b = sb.SegformerRoad(m, 'cpu', input_scale=.5)
    rgb = np.zeros((480, 640, 3), np.uint8)
    mask, conf = b.predict(rgb)
    assert m.seen[0][2:] == (256, 320) and mask.shape == (480, 640)
    out = predict_mask(b, rgb, 1.0)  # nothing is fully certain -> unknown
    assert (out == 255).all()


def test_options_validation():
    assert sb._options({}) == (1.0, [0, 1], [])
    assert sb._options({'input_scale': .5, 'road_class_ids': [0]}) == (.5, [0], [])
    assert sb._options({'road_class_ids': [1], 'stairs_class_ids': [3]}) == (1.0, [1], [3])
    for bad in ({'foo': 1}, {'input_scale': 0}, {'input_scale': True}, {'road_class_ids': []}, {'road_class_ids': [-1]},
                {'stairs_class_ids': 3}, {'stairs_class_ids': [0]}):
        with pytest.raises(ValueError):
            sb._options(bad)
    with pytest.raises(ValueError):
        sb.SegformerRoad(Fake(3), 'cpu', road_class_ids=[5])
    with pytest.raises(ValueError):
        sb.SegformerRoad(Fake(3), 'cpu', road_class_ids=[1], stairs_class_ids=[3])
    with pytest.raises(ValueError):
        sb.SegformerRoad(Fake(), 'cpu').predict(np.zeros((4, 4), np.uint8))


def test_stairs_output():
    """4-class fine-tuned layout: 1 paved = road, 3 stairs = nonroad + last_stairs."""
    class Four(Fake):
        def forward(self, pixel_values):
            b, _, h, w = pixel_values.shape
            y = torch.zeros(b, 4, h // 4, w // 4)
            y[:, 1, :, : w // 8] = 10.   # paved on the left
            y[:, 3, :, w // 8:] = 10.    # stairs on the right
            return type('O', (), {'logits': y})()
    rgb = np.zeros((64, 128, 3), np.uint8)
    plain = sb.SegformerRoad(Four(4), 'cpu', road_class_ids=[1])
    assert not plain.has_stairs
    plain.predict(rgb)
    assert plain.last_stairs is None
    b = sb.SegformerRoad(Four(4), 'cpu', road_class_ids=[1], stairs_class_ids=[3])
    assert b.has_stairs and b.last_stairs is None  # known before any image is run
    mask, _ = b.predict(rgb)
    assert b.last_stairs.shape == mask.shape and b.last_stairs.dtype == bool
    assert mask[:, :20].all() and not mask[:, -20:].any()
    assert b.last_stairs[:, -20:].all() and not b.last_stairs[:, :20].any()


def test_config_accepts_model_directory(tmp_path: Path):
    (tmp_path / 'm').mkdir()
    cfg = validate_segmentation({'enabled': True, 'backend': 'outdoor_nav.segformer_backend:create_backend',
                                 'weights': 'm', 'options': {'input_scale': .5}}, tmp_path, True)
    assert cfg['weights'] == str((tmp_path / 'm').resolve())


def test_real_tiny_segformer(tmp_path):
    tr = pytest.importorskip('transformers')
    cfg = tr.SegformerConfig(num_labels=19, hidden_sizes=[8, 16, 16, 16], depths=[1, 1, 1, 1],
                             num_attention_heads=[1, 1, 1, 1], decoder_hidden_size=16)
    tr.SegformerForSemanticSegmentation(cfg).save_pretrained(tmp_path)
    b = sb.create_backend(weights=str(tmp_path), device='cpu', options={'input_scale': .5})
    rgb = np.random.randint(0, 255, (48, 70, 3), np.uint8)
    out = predict_mask(b, rgb, .0)
    assert out.shape == (48, 70) and set(np.unique(out)) <= {0, 1}
