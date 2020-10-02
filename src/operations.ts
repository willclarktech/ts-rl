import * as tf from "@tensorflow/tfjs-node";

// Grad function for gatherND is not implemented in TFJS
// See https://github.com/tensorflow/tfjs/issues/1795
export function gatherND(x: tf.Tensor, indices: tf.Tensor): tf.Tensor {
	const grad = (dy: tf.Tensor, saved: tf.Tensor[]): { x: () => tf.Tensor } => {
		return { x: (): tf.Tensor => tf.scatterND(saved[0], dy, x.shape) };
	};
	return tf.engine().runKernelFunc(
		(backend, save) => {
			save && save([indices]);
			return backend.gatherND(x, indices);
		},
		{ x },
		grad,
	);
}
