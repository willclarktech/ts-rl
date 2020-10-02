import { Observation } from "./environments";
import { sampleUniform } from "./util";

export interface Transition {
	readonly observation: Observation;
	readonly action: number;
	readonly reward: number;
	readonly done: boolean;
	readonly nextObservation: Observation;
}

export class ReplayMemory {
	public size: number;
	private capacity: number;
	private transitions: readonly Transition[];

	public constructor(capacity: number) {
		this.size = 0;
		this.capacity = capacity;
		this.transitions = [];
	}

	public store(transition: Transition): void {
		this.transitions = [...this.transitions, transition].slice(-this.capacity);
		this.size = this.transitions.length;
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
