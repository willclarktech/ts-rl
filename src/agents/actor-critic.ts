import * as tf from "@tensorflow/tfjs-node";

import { Environment, Observation, Sample } from "../environments";
import {
	calculateDiscountedNormalizedRewards,
	createNetwork,
	sum,
} from "../util";
import { Agent } from "./core";

export interface ActorCriticOptions {
	readonly alphaActor: number;
	readonly alphaCritic: number;
	readonly gamma: number;
	readonly hiddenWidths: readonly number[];
	readonly seed?: number;
}

export class ActorCritic implements Agent {
	public readonly name: string;

	private readonly options: ActorCriticOptions;
	private readonly sharedNetwork: tf.Sequential;
	private readonly actorHead: tf.layers.Layer;
	private readonly actor: tf.LayersModel;
	private readonly criticHead: tf.layers.Layer;
	private readonly critic: tf.LayersModel;
	private readonly actorOptimizer: tf.Optimizer;
	private readonly criticOptimizer: tf.Optimizer;

	private steps: number;

	public constructor(environment: Environment, options: ActorCriticOptions) {
		this.name = "ActorCritic";
		this.options = options;
		this.steps = 0;

		const widths = [
			environment.numObservationDimensions,
			...options.hiddenWidths,
		];
		this.sharedNetwork = createNetwork(widths, "relu");

		this.actorHead = tf.layers.dense({
			activation: "softmax",
			units: environment.numActions,
		});
		this.actor = tf.model({
			inputs: this.sharedNetwork.inputs,
			outputs: this.actorHead.apply(
				this.sharedNetwork.layers[this.sharedNetwork.layers.length - 1].output,
			) as tf.SymbolicTensor,
		});
		this.actorOptimizer = tf.train.adam(options.alphaActor);

		this.criticHead = tf.layers.dense({
			units: 1,
		});
		this.critic = tf.model({
			inputs: this.sharedNetwork.inputs,
			outputs: this.criticHead.apply(
				this.sharedNetwork.layers[this.sharedNetwork.layers.length - 1].output,
			) as tf.SymbolicTensor,
		});
		this.criticOptimizer = tf.train.adam(options.alphaCritic);
	}

	private getSample(
		environment: Environment,
		observation: Observation,
		_steps: number,
	): Sample {
		const { seed } = this.options;

		const processedObservation = tf.tensor2d(
			[...observation],
			[1, observation.length],
		);
		const output = this.actor.predictOnBatch(
			processedObservation,
		) as tf.Tensor2D;
		const squeezedOutput = output.squeeze<tf.Tensor1D>();
		const logProbababilities = squeezedOutput.log();
		const action = tf.multinomial(logProbababilities, 1, seed).dataSync()[0];
		const sample = environment.step(action);

		return sample;
	}

	private calculateActorLoss(observations: readonly Observation[]): tf.Scalar {
		const { seed } = this.options;

		const logProbabilities = observations.reduce((lps, observation) => {
			const processedObservation = tf.tensor2d(
				[...observation],
				[1, observation.length],
			);
			const output = this.actor.predictOnBatch(
				processedObservation,
			) as tf.Tensor2D;
			const squeezedOutput = output.squeeze<tf.Tensor1D>();
			const logits = squeezedOutput.log();
			const action = tf.multinomial(logits, 1, seed).dataSync()[0];
			const logProbability = logits.gather([action]);
			return lps.concat(logProbability);
		}, tf.tensor1d([]));
		const processedObservations = tf.tensor2d(
			observations.map((observation) => [...observation]),
		);
		const vEstimates = this.critic.predictOnBatch(
			processedObservations,
		) as tf.Tensor2D;
		const vEstimatesSqueezed = vEstimates.squeeze();
		return logProbabilities.mul(vEstimatesSqueezed).sum().mul(-1).asScalar();
	}

	private calculateCriticLoss(
		observations: readonly Observation[],
		rewards: readonly number[],
	): tf.Scalar {
		const { gamma } = this.options;
		const discountedReturns = calculateDiscountedNormalizedRewards(
			rewards,
			gamma,
		);
		const processedObservations = tf.tensor2d(
			observations.map((observation) => [...observation]),
		);
		const vEstimates = this.critic.predictOnBatch(
			processedObservations,
		) as tf.Tensor2D;

		return tf.losses.meanSquaredError(
			discountedReturns,
			vEstimates.reshape([discountedReturns.length]),
		);
	}

	public runEpisode(environment: Environment): number {
		let observations: readonly Observation[] = [environment.reset()];
		let done = false;
		let rewards: readonly number[] = [];

		tf.tidy(() => {
			while (!done) {
				this.actorHead.trainable = false;
				this.criticHead.trainable = false;
				this.steps += 1;

				const observation = observations[observations.length - 1];
				const {
					observation: nextObservation,
					reward,
					done: nextDone,
				} = this.getSample(environment, observation, this.steps);

				if (!nextDone) {
					observations = [...observations, nextObservation];
				}
				rewards = [...rewards, reward];
				done = nextDone;
			}

			this.actorHead.trainable = true;
			this.criticHead.trainable = false;
			this.actorOptimizer.minimize(() => {
				return this.calculateActorLoss(observations);
			});

			this.actorHead.trainable = false;
			this.criticHead.trainable = true;
			this.criticOptimizer.minimize(() => {
				return this.calculateCriticLoss(observations, rewards);
			});
		});

		return sum(rewards);
	}
}
