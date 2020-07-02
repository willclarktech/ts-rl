import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";

import carsData from "./data/cars.json";

type PreparedData = {
	readonly inputs: tf.Tensor;
	readonly labels: tf.Tensor;
	readonly inputMax: tf.Tensor;
	readonly inputMin: tf.Tensor;
	readonly labelMax: tf.Tensor;
	readonly labelMin: tf.Tensor;
};

type NormalizationParameters = {
	readonly inputMax: tf.Tensor;
	readonly inputMin: tf.Tensor;
	readonly labelMax: tf.Tensor;
	readonly labelMin: tf.Tensor;
};

export const prepareData = (
	data: readonly { readonly mpg: number; readonly horsepower: number }[],
): PreparedData => {
	const mutableData = Array.from(data);
	return tf.tidy(() => {
		tf.util.shuffle(mutableData);

		const inputs = mutableData.map((datum) => datum.horsepower);
		const labels = mutableData.map((datum) => datum.mpg);

		const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
		const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

		const inputMax = inputTensor.max();
		const inputMin = inputTensor.min();
		const labelMax = labelTensor.max();
		const labelMin = labelTensor.min();

		const normalizedInputs = inputTensor
			.sub(inputMin)
			.div(inputMax.sub(inputMin));
		const normalizedLabels = labelTensor
			.sub(labelMin)
			.div(labelMax.sub(labelMin));

		return {
			inputs: normalizedInputs,
			labels: normalizedLabels,
			inputMax,
			inputMin,
			labelMax,
			labelMin,
		};
	});
};

export const createNetwork = (): tf.Sequential => {
	const network = tf.sequential({
		name: "tutorial-2d",
		layers: [
			tf.layers.dense({
				inputShape: [1],
				units: 1,
			}),
			tf.layers.dense({
				units: 1,
			}),
		],
	});
	return network;
};

export const train = async (
	model: tf.Sequential,
	inputs: tf.Tensor,
	labels: tf.Tensor,
): Promise<tf.History> => {
	const compileArgs = {
		optimizer: tf.train.adam(),
		loss: tf.losses.meanSquaredError,
	};
	model.compile(compileArgs);

	const fitArgs = {
		batchSize: 32,
		epochs: 50,
		shuffle: true,
	};
	return model.fit(inputs, labels, fitArgs);
};

export const testModel = (
	model: tf.Sequential,
	{ inputMax, inputMin, labelMax, labelMin }: NormalizationParameters,
): readonly { readonly x: number; readonly y: number }[] => {
	const [xs, predictions] = tf.tidy(() => {
		const raw_xs = tf.linspace(0, 1, 100);
		const raw_predictions = model.predict(
			raw_xs.reshape([100, 1]),
		) as tf.Tensor;

		const unnormalized_xs = raw_xs
			.mul(inputMax.sub(inputMin))
			.add(inputMin)
			.dataSync();
		const unnormalized_predictions = raw_predictions
			.mul(labelMax.sub(labelMin))
			.add(labelMin)
			.dataSync();

		return [unnormalized_xs, unnormalized_predictions];
	});

	return Array.from(xs).map((x, i) => ({ x, y: predictions[i] }));
};

const main = async (): Promise<void> => {
	const model = createNetwork();
	const preparedData = prepareData(carsData);
	await train(model, preparedData.inputs, preparedData.labels);
	const results = testModel(model, preparedData);
	fs.writeFileSync("./results/data/xxx.json", JSON.stringify(results));
};

if (process.env.NODE_ENV !== "TEST") {
	main();
}
