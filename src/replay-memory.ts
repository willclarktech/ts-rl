import { Observation } from "./environments";
import { sampleUniform } from "./util";

export interface Transition {
	readonly observation: Observation;
	readonly action: number;
	readonly reward: number;
	readonly done: boolean;
	readonly nextObservation: Observation;
}

export interface ReplayMemory {
	readonly size: number;
	readonly store: (transition: Transition) => void;
	readonly sample: (n: number) => readonly Transition[];
}

export class BasicReplayMemory implements ReplayMemory {
	private readonly capacity: number;
	private transitions: readonly Transition[];

	public constructor(capacity: number) {
		this.capacity = capacity;
		this.transitions = [];
	}

	public get size(): number {
		return this.transitions.length;
	}

	public store(transition: Transition): void {
		this.transitions = [...this.transitions, transition].slice(-this.capacity);
	}

	public sample(n: number): readonly Transition[] {
		const initialSample: readonly Transition[] = [];
		return Array.from({ length: n }).reduce(
			({
				sample,
				transitions,
			}): {
				readonly sample: readonly Transition[];
				readonly transitions: readonly Transition[];
			} => {
				const i = sampleUniform(transitions.length);
				return {
					sample: [...sample, transitions[i]],
					transitions: [
						...transitions.slice(0, i),
						...transitions.slice(i + 1),
					],
				};
			},
			{ sample: initialSample, transitions: this.transitions },
		).sample;
	}
}

export class BalancedReplayMemory implements ReplayMemory {
	private readonly positiveCapacity: number;
	private readonly negativeCapacity: number;
	private positiveTransitions: readonly Transition[];
	private negativeTransitions: readonly Transition[];

	public constructor(positiveCapacity: number, negativeCapacity: number) {
		this.positiveCapacity = positiveCapacity;
		this.negativeCapacity = negativeCapacity;
		this.positiveTransitions = [];
		this.negativeTransitions = [];
	}

	public get size(): number {
		return this.positiveTransitions.length + this.negativeTransitions.length;
	}

	public store(transition: Transition): void {
		if (transition.reward >= 0) {
			this.positiveTransitions = [
				...this.positiveTransitions,
				transition,
			].slice(-this.positiveCapacity);
		} else {
			this.negativeTransitions = [
				...this.negativeTransitions,
				transition,
			].slice(-this.negativeCapacity);
		}
	}

	public sample(n: number): readonly Transition[] {
		const initialSample: readonly Transition[] = [];
		return Array.from({ length: n }).reduce(
			({
				sample,
				transitions,
			}): {
				readonly sample: readonly Transition[];
				readonly transitions: readonly Transition[];
			} => {
				const i = sampleUniform(transitions.length);
				return {
					sample: [...sample, transitions[i]],
					transitions: [
						...transitions.slice(0, i),
						...transitions.slice(i + 1),
					],
				};
			},
			{
				sample: initialSample,
				transitions: [...this.positiveTransitions, ...this.negativeTransitions],
			},
		).sample;
	}
}
