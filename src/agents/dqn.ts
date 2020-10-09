import * as tf from "@tensorflow/tfjs-node";

import { Environment, Observation } from "../environments";
import { gatherND } from "../operations";
import { BasicReplayMemory, ReplayMemory, Transition } from "../replay-memory";
import { createNetwork, sampleUniform, sum } from "../util";
import { Agent } from "./core";

export interface DQNOptions {
	readonly hiddenWidths: readonly number[];
	readonly alpha: number; // learning rate
	readonly gamma: number; // discount rate
	readonly epsilonInitial: number; // initial exploration rate, set to 0 to use greedy action policy
	readonly epsilonMinimum: number; // minimum exploration rate
	readonly epsilonDecay: number; // exploration rate decay per action, set to 0 to use epsilon-greedy not epsilon-decreasing
	readonly tau: number; // smooth target network weight update rate
	readonly targetNetworkUpdatePeriod: number; // set to 1 to use same parameters between q and target networks at all times
	readonly shouldClipLoss: boolean;
	readonly warmup: number;
	readonly replayMemoryCapacity: number;
	readonly minibatchSize: number;
}

export class DQN implements Agent {
	public readonly name: string;

	private readonly gamma: number;
	private readonly epsilonMinimum: number;
	private readonly epsilonDecay: number;
	private readonly tau: number;
	private readonly targetNetworkUpdatePeriod: number;
	private readonly numActions: number;
	private readonly warmup: number;
	private readonly replayMemory: ReplayMemory;
	private readonly minibatchSize: number;
	private readonly shouldClipLoss: boolean;
	private readonly qNetwork: tf.Sequential;
	private readonly optimizer: tf.Optimizer;

	private epsilon: number;
	private targetNetwork: tf.Sequential;
	private steps: number;

	public constructor(
		{ numObservationDimensionsProcessed, numActions }: Environment,
		{
			hiddenWidths,
			alpha,
			gamma,
			epsilonInitial,
			epsilonMinimum,
			epsilonDecay,
			tau,
			targetNetworkUpdatePeriod,
			shouldClipLoss,
			warmup,
			replayMemoryCapacity,
			minibatchSize,
		}: DQNOptions,
	) {
		this.name = "DQN";
		this.gamma = gamma;
		this.epsilon = epsilonInitial;
		this.epsilonMinimum = epsilonMinimum;
		this.epsilonDecay = epsilonDecay;
		this.numActions = numActions;
		this.warmup = warmup;
		this.replayMemory = new BasicReplayMemory(replayMemoryCapacity);
		this.minibatchSize = minibatchSize;
		this.optimizer = tf.train.sgd(alpha);
		this.shouldClipLoss = shouldClipLoss;
		this.tau = tau;
		this.targetNetworkUpdatePeriod = targetNetworkUpdatePeriod;
		const widths = [
			numObservationDimensionsProcessed,
			...hiddenWidths,
			numActions,
		];
		this.qNetwork = createNetwork(widths, "relu");
		this.targetNetwork = createNetwork(widths, "relu");
		this.synchroniseTargetNetwork();
		this.steps = 0;
	}

	private synchroniseTargetNetwork(): void {
		const newTargetWeights = this.qNetwork.weights.map((weight, i) =>
			weight
				.read()
				.mul(this.tau)
				.add(this.targetNetwork.weights[i].read().mul(1 - this.tau)),
		);
		this.targetNetwork.setWeights(newTargetWeights);
	}

	private act(observation: Observation): number {
		const isWarmup = this.steps < this.warmup;
		const shouldActRandom = isWarmup || Math.random() < this.epsilon;

		if (!isWarmup) {
			this.epsilon = Math.max(
				this.epsilonMinimum,
				this.epsilon * this.epsilonDecay,
			);
		}

		if (shouldActRandom) {
			return sampleUniform(this.numActions);
		}

		const output = this.qNetwork.predict(
			tf.tensor2d([...observation], [1, observation.length]),
		) as tf.Tensor2D;
		return output.argMax(-1).dataSync()[0];
	}

	private getTargetsFromTransitions(
		samples: readonly Transition[],
	): tf.Tensor1D {
		return tf.tensor1d(
			samples.map((transition) => {
				if (transition.done) {
					return transition.reward;
				}
				const output = this.targetNetwork.predict(
					tf.tensor2d(
						[...transition.nextObservation],
						[1, transition.nextObservation.length],
					),
				) as tf.Tensor2D;

				return output
					.squeeze()
					.gather([transition.action])
					.mul(this.gamma)
					.add(transition.reward)
					.dataSync()[0];
			}),
		);
	}

	private getLoss(
		transitions: readonly Transition[],
		targets: tf.Tensor1D,
	): tf.Scalar {
		const observations = transitions.map(
			(transition) => transition.observation,
		);
		const actions = transitions.map((transition) => transition.action);
		const output = this.qNetwork.predict(
			tf.tensor(observations),
		) as tf.Tensor2D;
		const predictions = gatherND(
			output,
			tf.tensor2d(
				actions.map((a, i) => [i, a]),
				[transitions.length, 2],
				"int32",
			),
		).squeeze();

		const loss = tf.losses.meanSquaredError(targets, predictions) as tf.Scalar;
		return this.shouldClipLoss ? (loss.clipByValue(-1, 1) as tf.Scalar) : loss;
	}

	private learn(): void {
		const transitions = this.replayMemory.sample(this.minibatchSize);
		const targets = this.getTargetsFromTransitions(transitions);

		this.optimizer.minimize(() => this.getLoss(transitions, targets));

		if (
			this.steps >= this.warmup &&
			(this.steps + 1) % this.targetNetworkUpdatePeriod === 0
		) {
			this.synchroniseTargetNetwork();
		}
	}

	public runEpisode(env: Environment): number {
		let observation = env.resetProcessed();
		let done = false;
		let baseRewards: readonly number[] = [];
		let rewards: readonly number[] = [];

		tf.tidy(() => {
			while (!done) {
				this.steps += 1;
				const action = this.act(observation);
				const sample = env.step(action);
				const processedSample = env.processSample(sample, this.steps);
				const {
					observation: nextObservation,
					reward,
					done: nextDone,
				} = processedSample;

				this.replayMemory.store({
					observation,
					action,
					reward,
					done: nextDone,
					nextObservation,
				});

				if (
					this.steps >= this.warmup &&
					this.replayMemory.size >= this.minibatchSize
				) {
					this.learn();
				}

				baseRewards = [...baseRewards, sample.reward];
				rewards = [...rewards, reward];
				done = nextDone;
				observation = nextObservation;
			}
		});

		return sum(baseRewards);
	}
}
