import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";

import carsData from "./data/cars.json";

type PreparedData = {
	readonly inputs: readonly number[];
	readonly labels: readonly number[];
	readonly normalizedInputs: tf.Tensor;
	readonly normalizedLabels: tf.Tensor;
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

type Point = {
	readonly x: number;
	readonly y: number;
};

const normalizeData = <T extends tf.Tensor>(
	tensor: T,
	min = tensor.min(),
	max = tensor.max(),
): T => tensor.sub(min).div(max.sub(min));

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

		const normalizedInputs = normalizeData(inputTensor, inputMin, inputMax);
		const normalizedLabels = normalizeData(labelTensor, labelMin, labelMax);

		return {
			inputs: inputs,
			labels: labels,
			normalizedInputs: normalizedInputs,
			normalizedLabels: normalizedLabels,
			inputMax,
			inputMin,
			labelMax,
			labelMin,
		};
	});
};

export const createNetwork = (
	widths: readonly number[],
	activation: "sigmoid",
): tf.Sequential => {
	const network = tf.sequential({
		name: "tutorial-2d",
		layers: widths.slice(1).map((width, i) =>
			i === 0
				? tf.layers.dense({
						inputShape: [widths[0]],
						units: width,
						activation,
				  })
				: i !== widths.length - 2
				? tf.layers.dense({ units: width, activation })
				: tf.layers.dense({ units: width }),
		),
	});
	return network;
};

export const train = async (
	model: tf.Sequential,
	inputs: tf.Tensor,
	labels: tf.Tensor,
	hyperparameters: tf.ModelFitArgs & { readonly learningRate: number },
): Promise<tf.History> => {
	const compileArgs = {
		optimizer: tf.train.adam(hyperparameters.learningRate),
		loss: tf.losses.meanSquaredError,
	};
	model.compile(compileArgs);

	return model.fit(inputs, labels, hyperparameters);
};

export const testModel = (
	model: tf.Sequential,
	{ inputMax, inputMin, labelMax, labelMin }: NormalizationParameters,
): readonly tf.TypedArray[] => {
	return tf.tidy(() => {
		const rawXs = tf.linspace(0, 1, 100);
		const rawPredictions = model.predict(rawXs.reshape([100, 1])) as tf.Tensor;

		const unnormalizedXs = rawXs
			.mul(inputMax.sub(inputMin))
			.add(inputMin)
			.dataSync();
		const unnormalizedPredictions = rawPredictions
			.mul(labelMax.sub(labelMin))
			.add(labelMin)
			.dataSync();

		return [unnormalizedXs, unnormalizedPredictions];
	});
};

const createPlotData = (
	xs: readonly number[],
	ys: readonly number[],
): readonly Point[] =>
	Array.from(xs).map((x: number, i: number): Point => ({ x, y: ys[i] }));

const main = async (): Promise<void> => {
	const preparedData = prepareData(carsData);

	const activationFunction = "sigmoid";
	const model = createNetwork([1, 4, 4, 1], activationFunction);

	const hyperparameters = {
		batchSize: 32,
		epochs: 20,
		shuffle: true,
		learningRate: 0.03,
	};
	await train(
		model,
		preparedData.normalizedInputs,
		preparedData.normalizedLabels,
		hyperparameters,
	);
	const predictions = testModel(model, preparedData);

	const dataDir = "./results/data";
	const experimentName = "cars";
	const fileName = `${dataDir}/${experimentName}.json`;

	const originalData = createPlotData(preparedData.inputs, preparedData.labels);
	const predictionData = createPlotData(
		Array.from(predictions[0]),
		Array.from(predictions[1]),
	);

	fs.writeFileSync(
		fileName,
		JSON.stringify({
			name: "Horsepower v MPG",
			xLabel: "Horsepower",
			yLabel: "MPG",
			height: 300,
			originalData,
			predictionData,
		}),
	);
};

if (process.env.NODE_ENV !== "TEST") {
	main();
}
