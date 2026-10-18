"""量子化用 extra だけで実行できる動的 INT8 変換の回帰テスト。"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


class QuantizeTests(unittest.TestCase):
    """量子化コマンドの依存解決と変換結果を確認する。"""

    def test_dynamic_quantization_preserves_predictions(self):
        """INT8 の演算が生成され、浮動小数点モデルの出力に近いことを確認する。"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.onnx"
            target = Path(directory) / "quantized.onnx"
            weights = onnx.numpy_helper.from_array(
                np.eye(4, dtype=np.float32), name="weight"
            )
            graph = onnx.helper.make_graph(
                [onnx.helper.make_node("MatMul", ["input", "weight"], ["output"])],
                "quantize-test",
                [
                    onnx.helper.make_tensor_value_info(
                        "input", onnx.TensorProto.FLOAT, [1, 4]
                    )
                ],
                [
                    onnx.helper.make_tensor_value_info(
                        "output", onnx.TensorProto.FLOAT, [1, 4]
                    )
                ],
                [weights],
            )
            model = onnx.helper.make_model(
                graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10
            )
            onnx.save(model, source)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ocrus_scripts.quantize",
                    "--input",
                    str(source),
                    "--output",
                    str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            quantized = onnx.load(target)
            onnx.checker.check_model(quantized)
            self.assertIn(
                "MatMulInteger", [node.op_type for node in quantized.graph.node]
            )
            session = ort.InferenceSession(
                str(target), providers=["CPUExecutionProvider"]
            )
            data = np.array([[1.0, -1.0, 0.5, 0.0]], dtype=np.float32)
            actual = session.run(None, {"input": data})[0]
            np.testing.assert_allclose(actual, data, atol=0.02)


if __name__ == "__main__":
    unittest.main()
